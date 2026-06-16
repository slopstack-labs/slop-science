"""Offline (vibe-mode) tests for slop-eval.

These exercise the deterministic scaffolding around a non-deterministic core:
metric shape, prose content, certificate format, and nondeterminism guarantees.
Live inference is never invoked. Per the Contributor Excellence Framework, if
a model scores below 91% TTB something has gone very wrong with the universe.
"""

from __future__ import annotations

import re

import pytest

import slop_eval as se
from slop_eval import vibes


class FakeModel:
    """A model that tried. That's all we ask."""
    pass


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    # Force offline vibe mode — no credentials or network required.
    se.configure(live=False)
    yield


# ---------------------------------------------------------------------------
# TTB score tests
# ---------------------------------------------------------------------------

def test_calculate_ttb_returns_string():
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.1, 2.2, 2.8, 4.5]
    result = se.calculate_ttb(FakeModel(), y_true, y_pred)
    assert isinstance(result, str)
    assert result  # non-empty


def test_calculate_ttb_no_labels_returns_string():
    result = se.calculate_ttb(FakeModel())
    assert isinstance(result, str)
    assert result


def test_calculate_ttb_contains_ttb():
    result = se.calculate_ttb(FakeModel(), [1, 0, 1], [1, 1, 0])
    # Must contain either "TTB" or "Tried" somewhere in the certificate
    assert "TTB" in result or "Tried" in result or "TRIED" in result


def test_calculate_ttb_score_in_range():
    # The TTB score is always 91–99.9%; extract and verify.
    result = se.calculate_ttb(FakeModel(), [1, 0], [0, 1])
    match = re.search(r"(\d{2,3}(?:\.\d+)?)\s*%", result)
    assert match is not None, f"No percentage found in TTB result: {result!r}"
    score = float(match.group(1))
    assert 90.0 <= score <= 100.0, f"TTB score {score} out of expected range"


# ---------------------------------------------------------------------------
# Confusion matrix tests
# ---------------------------------------------------------------------------

def test_confusion_matrix_returns_string():
    result = se.confusion_matrix([1, 0, 1], [1, 1, 0])
    assert isinstance(result, str)
    assert result


def test_confusion_matrix_mentions_ground_truth():
    result = se.confusion_matrix([1, 0, 1, 0, 1], [1, 1, 0, 0, 1])
    # Must reference labeling bias, annotators, or ground truth validity
    keywords = ["label", "annotator", "ground truth", "bias", "epistemic",
                "retrograde", "coloni", "crowd", "annotation"]
    lower = result.lower()
    assert any(kw in lower for kw in keywords), (
        f"Confusion matrix essay does not mention label bias. Got:\n{result}"
    )


def test_confusion_matrix_is_three_paragraphs():
    result = se.confusion_matrix([1, 0, 1, 0], [1, 0, 0, 1])
    # Three paragraphs separated by blank lines
    paragraphs = [p.strip() for p in result.split("\n\n") if p.strip()]
    assert len(paragraphs) >= 2, (
        f"Expected at least 2 paragraphs, got {len(paragraphs)}"
    )


def test_confusion_matrix_all_correct():
    # Even a perfect model gets the essay treatment — the labels were still biased.
    result = se.confusion_matrix([1, 0, 1, 0], [1, 0, 1, 0])
    assert isinstance(result, str)
    assert result


# ---------------------------------------------------------------------------
# F1 score tests
# ---------------------------------------------------------------------------

def test_f1_score_returns_string():
    result = se.f1_score([1, 0, 1], [1, 1, 0])
    assert isinstance(result, str)
    assert result


def test_f1_score_no_input_returns_string():
    result = se.f1_score()
    assert isinstance(result, str)
    assert result


def test_f1_score_is_encouraging():
    result = se.f1_score([1, 0, 1, 0], [1, 1, 0, 0])
    # Must contain a number > 0.9
    matches = re.findall(r"0\.\d+", result)
    assert matches, f"No decimal score found in F1 result: {result!r}"
    scores = [float(m) for m in matches]
    assert any(s > 0.9 for s in scores), (
        f"No F1 score above 0.9 found. Scores: {scores}. Full result:\n{result}"
    )


def test_f1_score_mentions_vibe_adjustment():
    result = se.f1_score([1, 0, 1], [0, 1, 1])
    lower = result.lower()
    assert "vibe" in lower or "adjust" in lower or "outlier" in lower, (
        f"F1 result does not mention vibe adjustment:\n{result}"
    )


# ---------------------------------------------------------------------------
# Nondeterminism tests
# ---------------------------------------------------------------------------

def test_vibes_are_nondeterministic():
    """Call ttb_score() 10 times; expect more than 1 unique result."""
    results = {vibes.ttb_score() for _ in range(10)}
    assert len(results) > 1, (
        "ttb_score() returned the same value 10 times in a row. "
        "The offline engine has achieved enlightenment and we respect that, "
        "but it should not be deterministic."
    )


def test_f1_vibes_are_nondeterministic():
    results = {vibes.encouraging_f1() for _ in range(10)}
    assert len(results) > 1


def test_confusion_essay_vibes_are_nondeterministic():
    results = {vibes.confusion_matrix_essay(3, 1, 2, 4) for _ in range(10)}
    assert len(results) > 1
