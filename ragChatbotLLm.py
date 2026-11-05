
import streamlit as st
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="C++ RAG Chatbot",
    page_icon="🎓"
)

st.title("🎓 C++ Knowledge Assistant")
st.caption("Document retrieval using FAISS and answer generation using Gemma 2 2B.")


# ---------------- LOAD DOCUMENTS ----------------

@st.cache_resource
def load_vectorstore():

    file_path = Path(__file__).parent / "C++_Introduction.txt"

    if not file_path.exists():
        raise FileNotFoundError("C++_Introduction.txt not found.")

    text = file_path.read_text(encoding="utf-8")

    documents = [
        Document(
            page_content=text,
            metadata={"source": file_path.name}
        )
    ]

    # Split into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    final_docs = text_splitter.split_documents(documents)

    # Generate embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS vector database
    db = FAISS.from_documents(final_docs, embeddings)

    return db


# ---------------- LOAD LLM ----------------

@st.cache_resource
def load_llm():

    return OllamaLLM(
        model="gemma2:2b",
        temperature=0.2
    )


# ---------------- INITIALIZE ----------------

try:
    db = load_vectorstore()
    llm = load_llm()
    st.success("Knowledge base and Gemma model initialized.")
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()


# ---------------- CHAT INTERFACE ----------------

question = st.text_input(
    "Ask a question about C++",
    placeholder="What is object-oriented programming?"
)

if st.button("Generate Answer", type="primary"):

    if question.strip():

        with st.spinner("Retrieving documents and generating answer..."):

            # Retrieve relevant chunks
            docs = db.similarity_search(question, k=3)

            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            # Ground the answer in retrieved context
            prompt = f"""
You are a college knowledge assistant.

Answer using ONLY the information in the context.
If the answer is not available, say that the
provided document does not contain the answer.

Context:
{context}

Question:
{question}

Answer:
"""

            # Generate answer using Gemma 2 2B
            response = llm.invoke(prompt)

            st.subheader("Answer")
            st.write(response)

            # Display retrieved evidence
            with st.expander("View retrieved sources"):
                for doc in docs:
                    st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                    st.write(doc.page_content)

    else:
        st.warning("Please enter a question.")
