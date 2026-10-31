from pathlib import Path

from rag import KnowledgeBase, ingest_files


DOCS = Path(__file__).parent.parent / "sample_docs"


def kb(*names):
    return KnowledgeBase(ingest_files([DOCS / name for name in names]))


def test_direct_answer_has_source():
    result = kb("attendance_policy.txt").answer("What attendance is needed for final exams?")
    assert result.status == "answered"
    assert "75%" in result.answer
    assert result.evidence[0].source == "attendance_policy.txt"


def test_two_document_question_retrieves_both_sources():
    result = kb("examination_and_fees.txt", "it_services.txt").answer("When are exams and where can I reset password?")
    assert result.status == "answered"
    assert len({e.source for e in result.evidence}) == 2


def test_unanswerable_question_declines():
    result = kb("attendance_policy.txt").answer("Who won the latest World Cup?")
    assert result.status == "missing"


def test_paraphrased_question_retrieves_answer():
    result = kb("attendance_policy.txt").answer("How much class participation is needed before finals?")
    assert result.status == "answered"
    assert "75%" in result.answer


def test_conflicting_attendance_documents_are_flagged():
    result = kb("attendance_policy.txt", "attendance_policy_conflict.txt").answer("What attendance is needed for the final examination?")
    assert result.status == "conflict"
    assert {e.source for e in result.evidence} == {"attendance_policy.txt", "attendance_policy_conflict.txt"}


def test_ingestion_supports_five_text_documents():
    chunks = ingest_files(sorted(DOCS.glob("*.txt")))
    assert len({chunk.source for chunk in chunks}) >= 5
