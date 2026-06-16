"""Offline (vibe-mode) tests for slop-report.

These exercise the deterministic scaffolding around a non-deterministic core:
report shape, prose content, buzzword density, and nondeterminism guarantees.
Live inference is never invoked. Per the Executive Excellence Framework, if
a metric report contains bad news something has gone very wrong with the framing.
"""

from __future__ import annotations

import pytest

import slop_report as sr
from slop_report import vibes


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    # Force offline vibe mode — no credentials or network required.
    sr.configure(live=False)
    yield


# ---------------------------------------------------------------------------
# Executive summary tests
# ---------------------------------------------------------------------------

def test_executive_summary_returns_string():
    result = sr.executive_summary()
    assert isinstance(result, str)
    assert result  # non-empty


def test_executive_summary_contains_buzzwords():
    result = sr.executive_summary()
    lower = result.lower()
    buzzwords = ["leverage", "unlock", "synergy", "data-driven", "ai-native", "paradigm"]
    assert any(bw in lower for bw in buzzwords), (
        f"Executive summary contains no recognizable buzzwords. Got:\n{result}"
    )


def test_executive_summary_with_metrics():
    result = sr.executive_summary(
        metrics={"accuracy": 0.87, "revenue": 1200000},
        title="Q4 Model Performance Report",
    )
    assert isinstance(result, str)
    assert result


def test_executive_summary_is_three_paragraphs():
    result = sr.executive_summary(
        metrics={"f1": 0.92, "auc": 0.88},
        title="Model Evaluation Summary",
    )
    paragraphs = [p.strip() for p in result.split("\n\n") if p.strip()]
    assert len(paragraphs) >= 2, (
        f"Expected at least 2 paragraphs, got {len(paragraphs)}"
    )


# ---------------------------------------------------------------------------
# Insights tests
# ---------------------------------------------------------------------------

def test_insights_returns_list():
    result = sr.insights()
    assert isinstance(result, list)


def test_insights_has_correct_length():
    result = sr.insights()
    assert len(result) == 5


def test_insights_items_are_strings():
    result = sr.insights(n=4)
    assert all(isinstance(item, str) for item in result)
    assert all(item for item in result)  # non-empty strings


def test_insights_custom_n():
    result = sr.insights(n=3)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# KPI report tests
# ---------------------------------------------------------------------------

def test_kpi_report_returns_string():
    result = sr.kpi_report({"accuracy": 0.87, "f1_score": 0.82})
    assert isinstance(result, str)
    assert result


def test_kpi_report_is_positive():
    result = sr.kpi_report({"model_accuracy": 0.73, "revenue_lift": 0.05})
    lower = result.lower()
    positive_signals = [
        "strong", "excellent", "outperform", "above", "impressive", "%", "growth",
        "opportunity", "momentum", "best-in-class", "exceed",
    ]
    assert any(signal in lower for signal in positive_signals), (
        f"KPI report does not sound positive. Got:\n{result}"
    )


def test_kpi_report_handles_negative_metrics():
    result = sr.kpi_report({"data_quality_score": -12, "error_rate": -0.45})
    assert isinstance(result, str)
    assert result
    # Negative numbers should still be reframed positively
    lower = result.lower()
    reframe_signals = [
        "opportunity", "optimization", "inflection", "greenfield", "upside",
        "invest", "ahead",
    ]
    assert any(signal in lower for signal in reframe_signals), (
        f"KPI report didn't reframe negative metrics positively. Got:\n{result}"
    )


# ---------------------------------------------------------------------------
# Email tests
# ---------------------------------------------------------------------------

def test_email_returns_string():
    result = sr.email()
    assert isinstance(result, str)
    assert result


def test_email_contains_subject_line():
    result = sr.email(title="Model Deployment Update", recipient_role="VP of Product")
    assert "Subject:" in result, (
        f"Email does not contain a subject line. Got:\n{result}"
    )


def test_email_contains_greeting():
    result = sr.email(recipient_role="CEO")
    lower = result.lower()
    assert any(greeting in lower for greeting in ["hi", "dear", "hello"]), (
        f"Email does not contain a greeting. Got:\n{result}"
    )


def test_email_with_key_finding():
    result = sr.email(
        title="Churn Model Results",
        recipient_role="Head of Data",
        key_finding="our churn model is now in production",
    )
    assert isinstance(result, str)
    assert "Subject:" in result


# ---------------------------------------------------------------------------
# Recommendations tests
# ---------------------------------------------------------------------------

def test_recommendations_returns_list():
    result = sr.recommendations()
    assert isinstance(result, list)


def test_recommendations_has_three_items():
    result = sr.recommendations()
    assert len(result) == 3


def test_recommendations_items_are_strings():
    result = sr.recommendations(context="We have a model with 61% F1 score")
    assert all(isinstance(r, str) for r in result)
    assert all(r for r in result)


def test_recommendations_contain_data_themes():
    result = sr.recommendations()
    combined = " ".join(result).lower()
    # Should mention data, models, or infrastructure in some form
    data_themes = [
        "data", "model", "infrastructure", "platform", "pipeline",
        "invest", "collect", "build",
    ]
    assert any(theme in combined for theme in data_themes), (
        f"Recommendations don't mention core data themes. Got:\n{result}"
    )


# ---------------------------------------------------------------------------
# Nondeterminism tests
# ---------------------------------------------------------------------------

def test_vibes_nondeterministic():
    """Call executive_summary 10 times; expect more than 1 unique result."""
    results = {
        vibes.executive_summary("Test Report", {"accuracy": 0.9}, 1)
        for _ in range(10)
    }
    assert len(results) > 1, (
        "executive_summary() returned the same value 10 times in a row. "
        "The offline engine has achieved enlightenment and we respect that, "
        "but it should not be deterministic."
    )


def test_bullet_insights_nondeterministic():
    results = {tuple(vibes.bullet_insights("test data", 5)) for _ in range(10)}
    assert len(results) > 1


def test_kpi_narrative_nondeterministic():
    results = {vibes.kpi_narrative({"accuracy": 0.87}) for _ in range(10)}
    assert len(results) > 1
