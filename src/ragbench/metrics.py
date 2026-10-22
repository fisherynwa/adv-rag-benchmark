"""Scores that need no LLM: evidence found, bootstrap intervals, agreement, failure split."""
import re

import numpy as np
import pandas as pd


def norm(text):
    """Collapse all whitespace, so line breaks from the PDF don't break exact matching."""
    return re.sub(r"\s+", " ", text).strip()


def evidence_found(contexts, evidence_texts):
    """Share of evidence sentences that appear word for word in the text the model read (NaN if none expected)."""
    if not evidence_texts:
        return np.nan
    seen = norm(" ".join(contexts))
    return sum(norm(t) in seen for t in evidence_texts) / len(evidence_texts)


def context_tokens(contexts):
    """Rough token count of the text the model read (about 4 characters per token)."""
    return len(norm(" ".join(contexts))) / 4


def ci(values, resamples=2000, seed=0):
    """Mean and 95% bootstrap interval, ignoring NaN. Returns (nan, nan, nan) if there are no values."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [rng.choice(v, len(v)).mean() for _ in range(resamples)]
    return v.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def interval(values):
    """'mean (low to high) n=…', where n is the number of values the mean is based on."""
    m, lo, hi = ci(values)
    n = int(np.isfinite(np.asarray(values, float)).sum())
    return "n/a" if np.isnan(m) else f"{m:.2f} ({lo:.2f} to {hi:.2f}) n={n}"


def agreement(a, b):
    """Share of equal 0/1 verdicts and Cohen's kappa (agreement corrected for chance)."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return po, ((po - pe) / (1 - pe) if pe < 1 else float("nan"))


def failure_split(results):
    """Per system, for answerable questions: correct, retrieval failure (evidence missing), generation failure."""
    answerable = results[results["type"] != "unanswerable"]
    wrong = answerable["correct"] == 0
    return pd.DataFrame({
        "correct": answerable.groupby("system")["correct"].mean(),
        "retrieval failure": (wrong & (answerable["found"] < 1)).groupby(answerable["system"]).mean(),
        "generation failure": (wrong & (answerable["found"] == 1)).groupby(answerable["system"]).mean(),
    })
