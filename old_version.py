#imports for UI and model 
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

#imports for file upload and text extration + splitting
from langchain_community.document_loaders import PyMuPDFLoader
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#imports for vector database and embeddings
import os
import faiss
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_core.documents import Document

#import for RAG reranking
import time
from rerankers import Reranker

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Study AI")

st.title("Study AI")

llm = ChatOllama(model="llama3")

# query_rewriting_str = """
# You a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval. Look at the input and try to reason about the underlying sematic intent / meaning.

# Here is the initial question: {question}

# Formulate an improved question and respond with the line containing the improved question such as:

# "(Improved question here.)"
# """

query_rewriting_str = """Provide a better search query for \
web search engine to answer the given question, begin and end \
the queries with ’**’. Question: \
{question} 

Only reply usinga complete sentence and only give the answer in the following format
**Question**"""



query_rewriting_prompt = ChatPromptTemplate.from_template(query_rewriting_str)

def _parse(text):
    new_text = text.strip('"').strip("**")
    print(new_text)
    return new_text

def rewriteQuery(query: str):
    chain = query_rewriting_prompt | llm | StrOutputParser() | _parse

    response = chain.invoke({
        "question" : query
    })

    print(response)

    return response



few_shot_example = """
For example, the question will be as follows: "What happened at Interleaf and Viaweb?"

The output will be as follows:

1. What were the major events or milestones in the history of Interleaf and Viaweb?
2. Who were the founders and key figures involved in the development of Interleaf and Viaweb?
3. What were the products or services offered by Interleaf and Viaweb?
4. Are there any notable success stories or failures associated with Interleaf and Viaweb?
"""
query_writing_str = """You are a helpful assistant that generates multiple search queries based on a \
single input query. A high school student will be asking questions. 
Generate {num_queries} search queries, one on each line, \
related to the following input query:
Query: {query}
Queries:

Only respond with the generated questions adding no extra words in the response.
"""

query_gen_prompt = ChatPromptTemplate.from_template(query_writing_str + few_shot_example)


def generate_queries(query: str, llm, num_queries: int = 4):
    chain = query_gen_prompt | llm | StrOutputParser()
    response = chain.invoke({
        "query": query,
        "num_queries": num_queries
    })
    print(response)
    queries = response.split("\n")
    queries_str = "\n".join(queries[2:])
    print(f"Generated queries:\n {queries_str}")
    return queries_str[0]

### inputs chain with chat_history, query and context (includes the template used)
def get_response(query, chat_history, context):
    template = """
    You are an assistant for question-answering tasks.

    Use the following documents and the chat history to answer the question.

    Do not use any knowledge you already know.

    If you don't know the answer, just say you don't know.

    Chat history: {chat_history}

    User question: {user_questions}

    Documents: {documents}

    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOllama(model="llama3")

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "chat_history" : chat_history,
        "user_questions" : query,
        "documents" : context
    })

def get_response(query, context):
    template = """
    You are an assistant for question-answering tasks.

    Use the following documents and the chat history to answer the question.

    Do not use any knowledge you already know.

    If you don't know the answer, just say you don't know.

    User question: {user_questions}

    Documents: {documents}

    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOllama(model="llama3")

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "user_questions" : query,
        "documents" : context
    })

### VERSION using PdfReader
#function that extracts the text from the input pdf
def get_pdf_text(pdf_docs):
    text=""
    count = 0
    for pdf in pdf_docs:
        count += 1
        print(count)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#splits the text into chunks that can be embedded
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,    
    )
    chunks = text_splitter.split_text(text)
    return chunks

#laods the embedding model
def create_embeddings():
    return OllamaEmbeddings(model='nomic-embed-text')

#creates a vector database (if not already existing) 
def create_or_load_vector_store(embeddings, store_name):
    persistent_directory = os.path.join(r"C:\Users\KD\Desktop\hsc llm", store_name)
    if not os.path.exists(persistent_directory):
        print("creating vector store")
        vectore_store = Chroma.from_documents(embeddings, persist_directory=persistent_directory)
    else:
        print("loading existing vectore store")
        vector_store = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embeddings
        )
    return vector_store

###VERSION 1
#converts the documents into Document type and primary ID for to add to the vector databse
def add_new_docs(pdf_docs):
    raw_text = get_pdf_text(pdf_docs)
    chunks = get_text_chunks(raw_text)
    documents = [Document(page_content=text) for text in chunks]
    uuids = [str(uuid4()) for _ in range(len(chunks))]
    return documents, uuids

#retrieves the most relevant data from vector database
def perform_retrieval(vector_store, query, k=5):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k':5, 'fetch_k':10})
    results = retriever.invoke(query)
    print(results)
    return results

import yake

def main():
    #initialises the vector database
    vector_store = create_or_load_vector_store(create_embeddings(), "chroma_db")
    
    #sidebar section that includes the file upload for RAG
    with st.sidebar:
        st.sidebar.title("Upload a file")
        pdf_docs = st.file_uploader("Upload your notes", accept_multiple_files=True, key="pdf_uploader")
        # pdf_docs = st.file_uploader("Upload your notes", accept_multiple_files=False, key="pdf_uploader")
        if st.button("Upload", key="process_button"):
            with st.spinner("Processing..."):
                document, uuids = add_new_docs(pdf_docs)
                vector_store.add_documents(documents=document, ids=uuids)
                
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

        with st.chat_message("AI"):
        ## testing query rewriter
            ai_response = generate_queries(user_query, llm=llm)
            st.markdown(ai_response)
            
        with st.chat_message("AI"):
            ai_response = rewriteQuery(user_query)
            st.markdown(ai_response)
        # # #retrieves AI response with the user query, the chat history and also the relevant data from the RAG process
        # with st.chat_message("AI"):
        #     ai_response = get_response(user_query, st.session_state.chat_history, perform_retrieval(vector_store,user_query))
        #     st.markdown(ai_response)

        #     # ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history))

        # #implementing query rewriter in response
        # with st.chat_message("AI"):
        #     rewritten_query = generate_queries(user_query, llm)
        #     ai_response = get_response(rewritten_query, st.session_state.chat_history, perform_retrieval(vector_store,rewritten_query))
        #     st.markdown(ai_response)


        # #implementing yake answer
        # with st.chat_message("AI"):
        #     kw_extractor = yake.KeywordExtractor()
        #     keywords = kw_extractor.extract_keywords(user_query)
        #     ai_response = get_response(user_query, st.session_state.chat_history, perform_retrieval(vector_store, keywords[0][0]))  
        #     st.markdown(ai_response)

        #             # #retrieves AI response with the user query, the chat history and also the relevant data from the RAG process
        # with st.chat_message("AI"):
        #     ai_response = get_response(user_query, perform_retrieval(vector_store,user_query))
        #     st.markdown(ai_response)

        #     # ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history))

        # #implementing query rewriter in response
        # with st.chat_message("AI"):
        #     rewritten_query = generate_queries(user_query, llm)
        #     ai_response = get_response(rewritten_query, perform_retrieval(vector_store,rewritten_query))
        #     st.markdown(ai_response)


        # #implementing yake answer
        # with st.chat_message("AI"):
        #     kw_extractor = yake.KeywordExtractor()
        #     keywords = kw_extractor.extract_keywords(user_query)
        #     ai_response = get_response(user_query, perform_retrieval(vector_store, keywords[0][0]))  
        #     st.markdown(ai_response)

        st.session_state.chat_history.append(AIMessage(ai_response))


if __name__ == "__main__":
    main()



        