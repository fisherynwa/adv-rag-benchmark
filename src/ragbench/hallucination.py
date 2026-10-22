"""Hallucination checks without an LLM: refusals and numbers that appear nowhere in the report."""
import re

import pandas as pd

REFUSAL = re.compile(r"does not (?:say|mention|state|specify|provide)"
                     r"|not (?:mentioned|stated|specified|provided|given|available)"
                     r"|no information|cannot be determined", re.I)
LOOKUP_TYPES = ("single", "table", "near_duplicate")      # answers that should be copied, not computed


def refused(answer):
    return bool(REFUSAL.search(answer or ""))


def numbers(text):
    """All numbers in a text, without thousands commas or a trailing full stop: '£18,500.' -> {'18500'}."""
    return {n.replace(",", "").rstrip(".") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text or "")}


def invented_numbers(answer, report_numbers):
    """Numbers in the answer that appear nowhere in the report."""
    return numbers(answer) - report_numbers


def hallucination_table(results, report_text):
    """Per system: answers to unanswerable questions, refusals of answerable ones, not grounded, invented numbers."""
    report_numbers = numbers(report_text)
    r = results.copy()
    r["refused"] = r["answer"].fillna("").map(refused)
    r["invented_number"] = r["answer"].fillna("").map(lambda a: bool(invented_numbers(a, report_numbers)))
    lookups = r[r["type"].isin(LOOKUP_TYPES)]
    return pd.DataFrame({
        "answers unanswerable anyway": 1 - r[r["type"] == "unanswerable"].groupby("system")["refused"].mean(),
        "refuses answerable questions": r[r["type"] != "unanswerable"].groupby("system")["refused"].mean(),
        "not grounded (judge)": 1 - r.groupby("system")["grounded"].mean(),
        "number not in report": lookups.groupby("system")["invented_number"].mean(),
    })
