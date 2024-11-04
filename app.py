#Import for User Interface
import streamlit as st

#Import for model and responses
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

#Import for PDF text extraction and splitting
from langchain_community.document_loaders import PyMuPDFLoader
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#Import for vector database and embeddings
import os
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_core.documents import Document

#Import for keyword extraction
import yake

#Import for relative current working directory
import os
import sys

###########################


#Initialising UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Study AI")

st.title("Study AI")

#Initialising LLM 
llm = ChatOllama(model="llama3")

#Initialising the Embedding model
embedding_model = OllamaEmbeddings(model='nomic-embed-text')

#Initialising the keyword extraction model
kw_extractor = yake.KeywordExtractor()
language = "en"

#Initialising Current Working Directory
app_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

#Rewrite-Retrieve-Read Implementation
def rewrite_query(query: str):

    query_rewriting_str = """
    You are an expert at reformulating questions. \
    Your reformulated questions are understandable for high students and teachers.
    The question you reformulate will begin and end with with ’**’. 
    
    Question: 
    {question} 

    Only reply using a complete sentence and only give the answer in the following format:
    **Question**"""

    query_rewriting_prompt = ChatPromptTemplate.from_template(query_rewriting_str)

    chain = query_rewriting_prompt | llm | StrOutputParser()

    response = chain.invoke({
        "question" : query
    })

    print(response)

    return response 

def get_keywords(rewritten_query):
    keywords = kw_extractor.extract_keywords(rewritten_query)
    for kw in keywords:
        print(kw)
    return keywords

def get_response(query, context):
    qa_template = """
    You are an assistant for question-answering tasks.

    Use the following documents that is retrieved from the database is relevant \
    use it to answer the question answer the question.

    If the documents provided are not relevant to the question, use your own knowledge to answer.

    User question: {user_questions}

    Documents: {documents}

    Answer:
    """

    qa_prompt = ChatPromptTemplate.from_template(qa_template)

    llm = ChatOllama(model="llama3")

    chain = qa_prompt | llm | StrOutputParser()

    return chain.invoke({
        "user_questions" : query,
        "documents" : context
    })

#Function that extracts the text form the PDFs uploaded
def get_pdf_text(pdf):
    text=""
    count = 0
    for pdf in pdf:
        count += 1
        print(count)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#Function that splits the text into chunks for the embedding function
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,    
    )
    chunks = text_splitter.split_text(text)
    return chunks

#Function that loads or creates a vector store
def create_or_load_vector_store(embeddings, store_name):
    persistent_directory = os.path.join(app_dir, store_name)
    if not os.path.exists(persistent_directory):
        print("creating vector store")
        vectore_store = Chroma.from_documents(embeddings, persist_directory=persistent_directory)
    else:
        print("loading existing vector store")
        vector_store = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embeddings
        )
    return vector_store

vector_store = create_or_load_vector_store(embedding_model, "chroma_db")

#Function that adds the documents to the vector store
def add_new_docs(pdf):
    text = get_pdf_text(pdf)
    chunks = get_text_chunks(text)
    documents = [Document(page_context=text) for text in chunks]
    uuids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(documents=documents, ids=uuids)

#Function that retrieves the most relevant data from vector database
def perform_retrieval(vector_store, query, k=5):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    results = retriever.invoke(query)
    print(results)
    return results

def main():
    #sidebar section that includes the file upload for RAG
    with st.sidebar:
        st.sidebar.title("Upload a file")
        pdf_docs = st.file_uploader("Upload your notes", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Upload", key="process_button"):
            with st.spinner("Processing..."):
                add_new_docs(pdf_docs)
                st.success("Done!")

    #conversation section
    for message in st.session_state.chat_history:
        if isinstance(message,HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        else:
            with st.chat_message("AI"):
                st.markdown(message.content)

    #user input 
    user_query = st.chat_input("Your message")

    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        #Testing the LLM rewriter with refined prompt query for high school teacher / student
        with st.chat_message("AI"):
            ai_response = rewrite_query(user_query)
            st.markdown(ai_response)

        #Testing the LLM rewriter in similarity search WITH AI answer rewriting
        with st.chat_message("AI"):  
            reworded_query = rewrite_query(user_query)        
            get_keywords(reworded_query)
            test_response = get_response(reworded_query, perform_retrieval(vector_store, reworded_query))
            st.markdown(test_response)

        #Testing the LLM rewriter in similarity search WITH AI answer rewriting AND keyword extraction retrieval
        with st.chat_message("AI"):
            reworded_query = rewrite_query(user_query)
            kws = get_keywords(reworded_query)
            print(kws[0])
            test_response = get_response(reworded_query, perform_retrieval(vector_store, kws[0]))
            st.markdown(test_response)

        #Testing a simple similarity search WITHOUT AI answer rewriting
        with st.chat_message("AI"):
            ai_response = get_response(user_query, perform_retrieval(vector_store, user_query))
            st.markdown(ai_response)

        st.session_state.chat_history.append(AIMessage(ai_response))


if __name__ == "__main__":
    main()