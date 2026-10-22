"""The benchmark itself: the answer key matches the rendered PDF."""
from collections import Counter

from ragbench.metrics import norm

EXPECTED_TYPES = {"single": 52, "table": 16, "near_duplicate": 18, "multihop_section": 20, "cross_section": 20,
                  "summary": 12, "superseded": 6, "unanswerable": 10}


def test_question_counts(answer_key):
    assert len(answer_key) == 154
    assert Counter(q["type"] for q in answer_key) == EXPECTED_TYPES
    assert len({q["id"] for q in answer_key}) == 154
    assert len({q["question"] for q in answer_key}) == 154


def test_dev_test_split(answer_key):
    splits = Counter(q["split"] for q in answer_key)
    assert splits == {"dev": 103, "test": 51}


def test_report_has_25_pages(report_pages):
    assert len(report_pages) == 25


def test_every_answerable_question_has_evidence_on_its_page(answer_key, report_pages):
    for q in answer_key:
        if q["type"] == "unanswerable":
            assert q["evidence"] == [], q["id"]
            continue
        assert q["evidence"], q["id"]
        for e in q["evidence"]:
            assert norm(e["text"]) in report_pages[e["page"]], f"{q['id']}: not on page {e['page']}"


def test_superseded_figures_are_where_expected(answer_key, report_pages):
    for q in answer_key:
        for s in q["superseded"]:
            assert norm(s["text"]) in report_pages[s["page"]], q["id"]


def test_unanswerable_terms_appear_nowhere(answer_key, report_pages):
    whole_report = " ".join(report_pages.values()).lower()
    for q in answer_key:
        if q["type"] == "unanswerable":
            assert q["absent_terms"], q["id"]
            for term in q["absent_terms"]:
                assert term.lower() not in whole_report, f"{q['id']}: '{term}' is in the report"
