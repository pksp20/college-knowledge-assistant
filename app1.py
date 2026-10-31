"""College Knowledge Assistant: small, local, extractive RAG demo."""

from pathlib import Path
import tempfile

import streamlit as st

from rag import KnowledgeBase, ingest_files


st.set_page_config(page_title="College Knowledge Assistant", page_icon="🎓")
st.title("🎓 College Knowledge Assistant")
st.caption("Ask questions about the uploaded official college documents. Answers are grounded in retrieved text.")

if "kb" not in st.session_state:
    st.session_state.kb = KnowledgeBase()

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload PDF or TXT documents", type=["pdf", "txt"], accept_multiple_files=True)
    if st.button("Build / refresh knowledge base", type="primary", disabled=not uploads):
        with st.spinner("Reading and indexing documents…"):
            with tempfile.TemporaryDirectory() as temp_dir:
                paths = []
                for up in uploads:
                    path = Path(temp_dir) / Path(up.name).name
                    path.write_bytes(up.getvalue())
                    paths.append(path)
                st.session_state.kb = KnowledgeBase(ingest_files(paths))
        st.success(f"Indexed {len(st.session_state.kb.chunks)} passages from {len(uploads)} file(s).")
    if st.session_state.kb.chunks:
        st.caption(f"Knowledge base: {len(st.session_state.kb.documents)} document(s), {len(st.session_state.kb.chunks)} passages")
        if st.button("Load included sample documents"):
            samples = Path(__file__).parent / "sample_docs"
            sample_paths = sorted(p for p in samples.glob("*.txt") if not p.name.endswith("_conflict.txt"))
            st.session_state.kb = KnowledgeBase(ingest_files(sample_paths))
            st.rerun()

question = st.text_input("Your question", placeholder="For example: What attendance is required to sit the exam?")
if st.button("Ask", type="primary") or (question and st.session_state.get("last_question") == question):
    if not question.strip():
        st.info("Enter a question first.")
    elif not st.session_state.kb.chunks:
        st.info("Upload documents and build the knowledge base, or load the included sample documents.")
    else:
        result = st.session_state.kb.answer(question)
        st.session_state.last_question = question
        if result.status == "missing":
            st.warning("I couldn't find enough information in the uploaded documents to answer that question.")
        elif result.status == "conflict":
            st.warning("The retrieved documents appear to disagree. Please check both sources before relying on an answer.")
            for item in result.evidence:
                st.markdown(f"**{item.source}**{f', page {item.page}' if item.page else ''}: {item.text}")
        else:
            st.markdown("### Answer")
            st.write(result.answer)
        if result.evidence:
            with st.expander("Supporting sources", expanded=True):
                for item in result.evidence:
                    loc = f", page {item.page}" if item.page else ""
                    st.markdown(f"**{item.source}{loc}** · relevance {item.score:.2f}\n\n> {item.text}")

st.divider()
st.caption("This demo uses local TF-IDF retrieval and evidence excerpts. It does not invent a fluent answer beyond the retrieved passages.")
