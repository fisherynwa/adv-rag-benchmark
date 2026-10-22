import numpy as np
import pandas as pd
import pytest

from ragbench.hallucination import hallucination_table, invented_numbers, numbers, refused


@pytest.mark.parametrize("answer", ["The report does not say.", "This is not mentioned in the context.",
                                    "There is no information about that.", "It cannot be determined."])
def test_refusals_are_detected(answer):
    assert refused(answer)


@pytest.mark.parametrize("answer", ["15 years.", "The T3-S is certified for waves of up to 22 metres.", "", None])
def test_answers_are_not_refusals(answer):
    assert not refused(answer)


def test_numbers_drop_commas_and_trailing_full_stops():
    assert numbers("£18,500 in 2025.") == {"18500", "2025"}
    assert numbers("Margin was 11.2%.") == {"11.2"}
    assert numbers("No digits here") == set()


def test_invented_numbers():
    report = numbers("Founded in 1994. Revenue £212.4 million.")
    assert invented_numbers("Founded in 1994.", report) == set()
    assert invented_numbers("Founded in 1996.", report) == {"1996"}


def test_known_false_positive_for_numbers_written_in_words():
    # The report writes "Eighty-three ports"; an answer saying "83" is flagged although it is right
    assert invented_numbers("83 ports", numbers("Eighty-three ports had subscribed")) == {"83"}


def test_hallucination_table():
    results = pd.DataFrame({
        "system": ["s"] * 4,
        "type": ["unanswerable", "single", "single", "table"],
        "answer": ["The report does not say.", "The report does not say.", "Founded in 1996.", "£18,500."],
        "grounded": [1.0, 0.0, 1.0, np.nan],
    })
    row = hallucination_table(results, "Founded in 1994. Price £18,500.").loc["s"]
    assert row["answers unanswerable anyway"] == 0.0
    assert row["refuses answerable questions"] == pytest.approx(1 / 3)
    assert row["not grounded (judge)"] == pytest.approx(1 / 3)
    assert row["number not in report"] == pytest.approx(1 / 3)      # 1996 is not in the report
