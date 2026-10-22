import numpy as np
import pandas as pd
import pytest

from ragbench.metrics import agreement, ci, context_tokens, evidence_found, failure_split, interval


def test_evidence_found_ignores_line_breaks():
    contexts = ["The Tidewatch T3 has a design\nlife of 15 years.", "Other text."]
    assert evidence_found(contexts, ["design life of 15 years"]) == 1.0


def test_evidence_found_partial_and_missing():
    contexts = ["Revenue rose to £212.4 million."]
    assert evidence_found(contexts, ["Revenue rose to £212.4 million", "Operating margin was 11.2%"]) == 0.5
    assert evidence_found(contexts, ["Operating margin was 11.2%"]) == 0.0


def test_evidence_found_is_nan_for_unanswerable():
    assert np.isnan(evidence_found(["anything"], []))


def test_context_tokens_is_characters_over_four():
    assert context_tokens(["abcd", "efgh"]) == pytest.approx(9 / 4)   # "abcd efgh" = 9 characters


def test_ci_contains_the_mean_and_ignores_nan():
    m, lo, hi = ci([1, 0, 1, 1, np.nan, 0, 1, 1])
    assert m == pytest.approx(5 / 7)
    assert lo <= m <= hi


def test_ci_has_zero_width_when_all_values_agree():
    # The bootstrap's blind spot: 77 correct answers out of 77 give "1.00 (1.00 to 1.00)"
    assert ci([1.0] * 77) == (1.0, 1.0, 1.0)


def test_ci_is_reproducible():
    values = np.random.default_rng(1).integers(0, 2, 100)
    assert ci(values) == ci(values)


def test_ci_of_nothing():
    assert all(np.isnan(x) for x in ci([np.nan, np.nan]))


def test_interval_reports_n_without_missing_values():
    assert interval([1, 1, 0, np.nan]).endswith("n=3")
    assert interval([np.nan]) == "n/a"


def test_agreement_perfect_and_chance():
    po, kappa = agreement([1, 0, 1, 0], [1, 0, 1, 0])
    assert (po, kappa) == (1.0, 1.0)
    po, kappa = agreement([1, 1, 0, 0], [1, 0, 1, 0])   # agreement no better than chance
    assert po == 0.5 and kappa == pytest.approx(0.0)


def test_failure_split():
    results = pd.DataFrame({
        "system": ["s"] * 5,
        "type": ["single", "single", "single", "single", "unanswerable"],
        "correct": [1, 0, 0, 1, 1],
        "found": [1.0, 0.0, 1.0, 1.0, np.nan],
    })
    row = failure_split(results).loc["s"]
    assert row["correct"] == 0.5               # 2 of 4 answerable
    assert row["retrieval failure"] == 0.25    # wrong, evidence missing
    assert row["generation failure"] == 0.25   # wrong, evidence there
