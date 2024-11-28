import os 
import tempfile

import chromadb
import ollama
import streamlit as st

from langchain_chroma import Chroma
from pypdf import PdfReader
import ollama
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Study AI")

st.title("Study AI")  



system_qa_prompt = """
You are an assistant for question-answering tasks.

Use the following documents that is retrieved from the database is relevant \
use it to provide a complete and concise response to the user's query. \
Do not mention references, sources, or citations in your response

If the documents provided are not relevant to the question, use your own knowledge to answer.

Limit your answer to 3-4 sentences.

"""

system_qr_prompt = """
You are an expert at reformulating questions. \
Your reformulated questions are understandable for high students and teachers.
The question you reformulate will begin and end with with ’**’. 

If the question is already simple, you can reply with the same question. Otherwise, only \
reply using a simpler and complete sentence and only give the answer in the following format:
    **Question**"""

def rewrite_query(query: str):

    llm = ChatOllama(model="llama3.1")

    rewrite_prompt_template = ChatPromptTemplate([
        ("system", system_qr_prompt),
        ("user", f" Question: {query}")
    ])

    chain = rewrite_prompt_template | llm | StrOutputParser()

    response = chain.invoke({
        "question" : query
    })

    print(response)

    return response 

def rewritten_query_parser(query: str, delimiter="**"):
    # Strip leading and trailing spaces
    input_string = input_string.strip()
    
    # Check if the delimiters are present
    start = input_string.find(delimiter)
    if start == -1:
        return ""
    
    end = input_string.find(delimiter, start + len(delimiter))
    if end == -1:
        return ""
    
    # Extract and return the text between the delimiters
    return input_string[start + len(delimiter):end].strip()

def process_document(file_uploaded: UploadedFile) -> list[Document]:
    # Store uploaded file as a temp file
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(file_uploaded.read())

    docs = ""
    pdf = PdfReader(temp_file.name)
    for page in pdf.pages:
        docs += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,    
    )

    return text_splitter.split_text(docs)

def get_collection():

    embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")

    chroma_client = chromadb.PersistentClient(path="./hsc-llm")

    return Chroma(
        client=chroma_client,
        collection_name="biology_collection",
        embedding_function=embedding_model,
    )


def add_documents_to_collection(chunks: list[Document], file_name: str):
    collection = get_collection()
    documents, ids = [],  []

    for idx, split in enumerate(chunks):
        documents.append(Document(page_content=split))
        ids.append(f"{file_name}_{idx}")

    collection.add_documents(documents=documents, ids=ids)


def perform_retrieval(query: str, n = 10):
    collection = get_collection()
    retriever = collection.as_retriever(search_type="similarity", search_kwargs={"k": n})
    results = retriever.invoke(query)
    return results

def get_response(context: str, prompt: str):
    
    llm = ChatOllama(model="llama3.1")

    qa_prompt_template = ChatPromptTemplate([
        ("system", system_qa_prompt),
        ("user", f"Context: {context}, Question: {prompt}")
    ])

    chain = qa_prompt_template | llm | StrOutputParser()

    return chain.invoke({
        "prompt": prompt, 
        "context": context
    })


def reranker(prompt: str, documents: list[str]) -> tuple[str,list[int]]:
    document_texts = [doc.page_content for doc in documents if doc.page_content and doc.page_content.strip()]
    
    
    relevant_text = ""
    relevant_text_ids = []
    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, document_texts, top_k=5)
    for rank in ranks:
        relevant_text += document_texts[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text, relevant_text_ids

if __name__ == "__main__":
    with st.sidebar:
        st.sidebar.title("Upload a file")
        pdf_docs = st.file_uploader("Upload your notes", accept_multiple_files=False, key="pdf_uploader")
        if st.button("Upload", key="process_button"):
            with st.spinner("Processing..."):
                all_splits = process_document(pdf_docs)
                add_documents_to_collection(all_splits, pdf_docs.name)
                st.success("Done!")

    #conversation section
    for message in st.session_state.chat_history:
        if isinstance(message,HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        else:
            with st.chat_message("AI"):
                st.markdown(message.content)

    user_query = st.chat_input("Your message")

    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            # without rewrite
            results = perform_retrieval(user_query)
            relevant_text, relevant_text_ids = reranker(user_query, results)
            response = get_response(context=relevant_text, prompt=user_query)
            st.write(response)

            # #with rewrite
            # rewritten_query = rewrite_query(user_query)
            # relevant_text, relevant_text_ids = reranker(rewritten_query, results)
            # response = get_response(context=relevant_text, prompt=rewritten_query)
            # st.write(response)
            
            with st.expander("See retrieved documents"):
                st.write(results)

            with st.expander("See most relevant document ids"):
                st.write(relevant_text_ids)
                st.write(relevant_text)
        
        st.session_state.chat_history.append(AIMessage(response))
