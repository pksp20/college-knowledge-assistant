"""Document parsing, TF-IDF retrieval, and evidence-grounded response logic."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    page: int | None = None


@dataclass(frozen=True)
class Evidence:
    text: str
    source: str
    page: int | None
    score: float


@dataclass(frozen=True)
class Answer:
    status: str  # answered, missing, or conflict
    answer: str
    evidence: tuple[Evidence, ...]


def _split_text(text: str, max_chars: int = 700, overlap: int = 100) -> list[str]:
    """Split on sentence boundaries, with a small overlap for context."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = current[-overlap:] + " " + sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def ingest_files(paths: Iterable[Path]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in paths:
        path = Path(path)
        if path.suffix.lower() == ".pdf":
            with fitz.open(path) as pdf:
                for page_no, page in enumerate(pdf, start=1):
                    for part in _split_text(page.get_text()):
                        chunks.append(Chunk(part, path.name, page_no))
        elif path.suffix.lower() == ".txt":
            for part in _split_text(path.read_text(encoding="utf-8")):
                chunks.append(Chunk(part, path.name))
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    return chunks


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks or []
        self.documents = sorted({c.source for c in self.chunks})
        self.vectorizer = None
        self.matrix = None
        if self.chunks:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", stop_words="english")
            self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

    def retrieve(self, question: str, top_k: int = 4) -> tuple[Evidence, ...]:
        if not self.chunks:
            return ()
        query = self.vectorizer.transform([question])
        scores = cosine_similarity(query, self.matrix).ravel()
        ranked = scores.argsort()[::-1][:top_k]
        return tuple(Evidence(self.chunks[i].text, self.chunks[i].source, self.chunks[i].page, float(scores[i])) for i in ranked)

    def answer(self, question: str, top_k: int = 4, min_score: float = 0.07) -> Answer:
        evidence = tuple(e for e in self.retrieve(question, top_k) if e.score >= min_score)
        if not evidence:
            return Answer("missing", "", ())
        # Surface high scoring passages from distinct documents. If multiple documents
        # support the query but contain materially different claims, report uncertainty.
        relevant = [e for e in evidence if e.score >= max(min_score, evidence[0].score * 0.62)]
        distinct = {e.source for e in relevant}
        if len(distinct) > 1 and _claims_conflict(relevant):
            return Answer("conflict", "", tuple(relevant))
        # Extractive response: show one or more source sentences rather than asking a
        # generative model to fill gaps. Two-document questions can cite both passages.
        chosen: list[Evidence] = []
        seen_sources: set[str] = set()
        for item in relevant:
            if item.source not in seen_sources:
                chosen.append(item)
                seen_sources.add(item.source)
            if len(chosen) == 2:
                break
        response = " ".join(_best_sentence(item.text, question) for item in chosen)
        return Answer("answered", response, tuple(chosen))


def _terms(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[a-zA-Z0-9]+", text) if len(w) > 2}


def _claims_conflict(evidence: list[Evidence]) -> bool:
    """Conservative heuristic: same topic and an explicit negation or differing numbers."""
    for i, left in enumerate(evidence):
        for right in evidence[i + 1:]:
            a, b = left.text.lower(), right.text.lower()
            shared = _terms(a) & _terms(b)
            if len(shared) < 3:
                continue
            negations = (" not ", " no ", " never ", " must not ", " may not ")
            neg_a = any(n in f" {a} " for n in negations)
            neg_b = any(n in f" {b} " for n in negations)
            nums_a = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", a))
            nums_b = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", b))
            if neg_a != neg_b or (nums_a and nums_b and nums_a != nums_b):
                return True
    return False


def _best_sentence(text: str, question: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    q = _terms(question)
    return max(sentences, key=lambda s: len(q & _terms(s)), default=text)
