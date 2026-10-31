import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS

#page config
st.title("C++ RAG Chatbot using Ollama")
#load &process data


def load_vectorstore():
    loader=TextLoader("C++_Introduction.txt",encoding="UTF-8")
    documents = loader.load()

    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    final_docs=text_splitter.split_documents(documents)
    embeddings=HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
db= load_vectorstore()

#load LLM
llm= Ollama(model="gemma2:2b")
#chat interface
user_question= st.text_input("Ask a queation about C++")

if user_question:
    with st.spinner("Thinking"):
        docs= db.similarity_search(user_question)
        context="\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Answer the question using only the context below.

        Context:
        {context}

        Question:
        {user_question}

        Answer:
        """

        response=llm.invoke(prompt)
        st.subheader("Answer")
        st.write(response)

        

