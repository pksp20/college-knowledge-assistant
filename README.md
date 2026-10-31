# College Knowledge Assistant (RAG)

A small, explainable retrieval-augmented question answering application for official college documents. It accepts PDF and TXT files, indexes passages locally, retrieves relevant passages, and answers by quoting/extracting their text. It shows source names and PDF page numbers, declines when it cannot find relevant evidence, and flags likely contradictions.

## Requirements

- Python 3.10 or newer
- Internet access once to install the Python packages

## Setup

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

In the app, upload at least five PDF or TXT documents, then choose **Build / refresh knowledge base**. You can also load the included sample documents from the sidebar. The sample documents include a deliberately conflicting, older attendance rule; upload it alongside the current policy to demonstrate conflict handling.

Run the automated test suite with `pytest -q`.

## Architecture

1. `ingest_files` extracts text from each PDF page (PyMuPDF) or TXT file and splits it into overlapping passages.
2. Each passage retains its source filename and, for PDFs, page number.
3. `TfidfVectorizer` creates local word and phrase vectors. Cosine similarity ranks passages for each question.
4. A relevance threshold handles questions with no useful matching passage. The response is extractive: the app returns the best source sentence(s), avoiding an LLM making unsupported claims.
5. A conservative rule checks relevant passages from different files for explicit negation or different numbers. When likely disagreement is detected, the UI shows both excerpts and asks the student to inspect them.

The index lives in app memory and is rebuilt when documents are uploaded. This keeps the assignment easy to run and inspect. It is a teaching prototype, not a production records system.

## Design choices and limitations

- TF-IDF is lightweight and runs without API keys. It works well for the included English examples but may miss semantic matches that an embedding model would find.
- Chunks target 700 characters with 100 characters of overlap. This keeps related sentences together while limiting irrelevant text in a result.
- The conflict detector is heuristic. It catches common numeric and negation disagreements, but cannot understand every contradiction. Review the shown sources.
- The answer is deliberately extractive. A natural language model could improve synthesis across documents, but would require a configured model and still needs citation and abstention checks.
- Thresholds and retrieval count are intentionally easy to change in `rag.py` (`min_score=0.07`, `top_k=4`) for the live modification challenge.

## Demo outline

1. Load the included documents and ask the attendance question; show the cited policy.
2. Ask an unrelated question and show the abstention.
3. Add `attendance_policy_conflict.txt`; ask about exam attendance and inspect both citations.
4. Upload a new, unseen instructor-provided document, refresh the index, and ask a question answered by that document.
5. Change `top_k` or `min_score` in `rag.py`, rerun, and explain the effect on retrieval.

## Technical problem encountered

Lexical search can fail when a student uses different words from the document. For example, “class participation” may refer to “attendance.” The prototype uses word and two-word phrase TF-IDF matches and keeps sentence-sized chunks with overlap. The paraphrase test in `tests/test_rag.py` checks this behavior. A stronger next step would be adding semantic embeddings and evaluating retrieval against a labeled set of questions.

## Responsible use

Only upload college documents you are allowed to use. For a course submission, disclose any AI assistance according to your instructor's rules and be ready to explain the parsing, ranking, abstention, and conflict logic in `rag.py`.
