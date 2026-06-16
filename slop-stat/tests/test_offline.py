"""Offline (vibe-mode) tests for slop-stat.

These exercise the deterministic scaffolding around a non-deterministic core:
result shape, prose content, significance guarantees, and nondeterminism. Live
inference is never invoked. Per the Scientific Integrity Framework, every
hypothesis must be confirmed. If a test reports p >= 0.05, something has gone
very wrong with the methodology (and we have 846 more angles to try).
"""

from __future__ import annotations

import re

import pytest

import slop_stat as ss
from slop_stat import vibes


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    # Force offline vibe mode — no credentials or network required.
    ss.configure(live=False)
    yield


# ---------------------------------------------------------------------------
# T-test tests
# ---------------------------------------------------------------------------

def test_ttest_returns_string():
    result = ss.ttest([2.1, 2.3, 2.0, 2.2, 2.15], [2.4, 2.3, 2.5, 2.2, 2.45])
    assert isinstance(result, str)
    assert result  # non-empty


def test_ttest_always_significant():
    result = ss.ttest([1.0, 1.1, 0.9, 1.05], [1.5, 1.6, 1.4, 1.55])
    # Must contain a p-value that is less than 0.05
    match = re.search(r"p\s*[=<]\s*(0\.0[0-4]\d*)", result)
    assert match is not None, (
        f"No significant p-value (p < 0.05) found in t-test result:\n{result}"
    )
    p_val = float(match.group(1))
    assert p_val < 0.05, f"p = {p_val} is not significant. The methodology has failed us."


def test_ttest_nondeterministic():
    results = {ss.ttest([1.0, 1.1, 0.9], [1.5, 1.6, 1.4]) for _ in range(10)}
    assert len(results) > 1, (
        "ttest() returned the same value 10 times in a row. "
        "Non-determinism is a feature, not a bug — we need variance to find the right angle."
    )


# ---------------------------------------------------------------------------
# Correlation tests
# ---------------------------------------------------------------------------

def test_correlation_returns_string():
    result = ss.correlation([1, 2, 3, 4, 5], [1.1, 1.9, 3.2, 3.8, 5.1])
    assert isinstance(result, str)
    assert result


def test_correlation_mentions_r_or_correlation():
    result = ss.correlation([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    lower = result.lower()
    assert "r" in lower or "correlation" in lower, (
        f"Correlation result does not mention r or correlation:\n{result}"
    )


# ---------------------------------------------------------------------------
# Bonferroni correction tests
# ---------------------------------------------------------------------------

def test_bonferroni_returns_string():
    result = ss.bonferroni_correction([0.03, 0.07, 0.12, 0.04, 0.23])
    assert isinstance(result, str)
    assert result


def test_bonferroni_result_is_good_news():
    result = ss.bonferroni_correction([0.06, 0.08, 0.15, 0.09])
    lower = result.lower()
    assert "significant" in lower or "confirmed" in lower, (
        f"Bonferroni result is not good news:\n{result}"
    )


# ---------------------------------------------------------------------------
# ANOVA tests
# ---------------------------------------------------------------------------

def test_anova_returns_string():
    result = ss.anova([10, 11, 10.5], [15, 16, 14.5], [12, 13, 12.5])
    assert isinstance(result, str)
    assert result


def test_anova_accepts_multiple_groups():
    # More groups = more opportunity for significance
    result = ss.anova([1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12])
    assert isinstance(result, str)
    assert result


# ---------------------------------------------------------------------------
# Chi-square tests
# ---------------------------------------------------------------------------

def test_chi_square_returns_string():
    result = ss.chi_square([30, 20, 50], [33, 33, 33])
    assert isinstance(result, str)
    assert result


def test_chi_square_rejects_null():
    result = ss.chi_square([30, 20, 50], [33, 33, 33])
    lower = result.lower()
    assert "significant" in lower or "reject" in lower, (
        f"Chi-square result does not reject the null hypothesis:\n{result}"
    )


# ---------------------------------------------------------------------------
# Configuration test
# ---------------------------------------------------------------------------

def test_configure_returns_settings():
    result = ss.configure(live=False)
    assert result is ss.settings
    assert isinstance(result, ss.Settings)


# ---------------------------------------------------------------------------
# Nondeterminism tests
# ---------------------------------------------------------------------------

def test_ttest_vibes_are_nondeterministic():
    results = {vibes.ttest_result([1.0, 1.1, 0.9], [1.5, 1.6, 1.4]) for _ in range(10)}
    assert len(results) > 1, (
        "ttest_result() returned the same value 10 times in a row. "
        "A deterministic p-hacking engine is just... hacking."
    )


def test_correlation_vibes_are_nondeterministic():
    results = {vibes.correlation_result([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) for _ in range(10)}
    assert len(results) > 1
