import os 
import tempfile

import chromadb
import ollama
import streamlit as st

from langchain_community.vectorstores import Chroma
from pypdf import PdfReader
import ollama
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Study AI")

st.title("Study AI")  



system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

IMPORTANT:L Do not mention references, sources, or citations in your response. There is no need to reference the context in your response.
If the documents provided are not relevant to the question, use your own knowledge to answer.

"""

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

    # chroma_client.get_or_create_collection(
    #     name="biology_collection", 
    #     embedding_function=embedding_model,
    #     metadata={"hnsw:space": "cosine"})

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
    # print(results)
    # print(type(results))
    return results

def get_response(context: str, prompt: str):
    
    llm = ChatOllama(model="llama3:latest")

    prompt_template = ChatPromptTemplate([
        ("system", system_prompt),
        ("user", f"Context: {context}, Question: {prompt}")
    ])

    chain = prompt_template | llm | StrOutputParser()

    return chain.invoke({
        "prompt": prompt, 
        "context": context
    })


def reranker(prompt: str, documents: list[str]) -> tuple[str,list[int]]:
    document_texts = [doc.page_content for doc in documents if doc.page_content and doc.page_content.strip()]
    
    
    relevant_text = ""
    relevant_text_ids = []
    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, document_texts, top_k=3)
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
            results = perform_retrieval(user_query)

            relevant_text, relevant_text_ids = reranker(user_query, results)
            response = get_response(context=relevant_text, prompt=user_query)
            st.write(response)
            
            with st.expander("See retrieved documents"):
                st.write(results)

            with st.expander("See most relevant document ids"):
                st.write(relevant_text_ids)
                st.write(relevant_text)
        
        st.session_state.chat_history.append(AIMessage(response))
