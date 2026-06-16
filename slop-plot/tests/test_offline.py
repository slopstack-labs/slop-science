"""Offline test suite for slop-plot.

All tests run entirely without network access or credentials. Live inference
is never triggered; the vibe engine is exercised directly.
"""

from __future__ import annotations

import sys
from io import StringIO

import pytest

import slop_plot.pyplot as slt
from slop_plot import vibes


# ---------------------------------------------------------------------------
# vibes.free_verse
# ---------------------------------------------------------------------------

def test_free_verse_returns_string():
    result = vibes.free_verse("TestColumn", 42)
    assert isinstance(result, str)
    assert len(result) > 0


def test_free_verse_is_nondeterministic():
    """The vibe engine must not be a deterministic lookup table."""
    results = {vibes.free_verse("Volatility", 100) for _ in range(10)}
    assert len(results) > 1, (
        "free_verse returned the same poem 10 times — the entropy source "
        "appears to be seeded or broken. A deterministic poetry engine "
        "is not a poetry engine; it is a CSV."
    )


# ---------------------------------------------------------------------------
# vibes.haiku
# ---------------------------------------------------------------------------

def test_haiku_returns_string():
    result = vibes.haiku("Returns")
    assert isinstance(result, str)
    assert len(result) > 0


def test_haiku_has_three_lines():
    result = vibes.haiku("Sharpe")
    lines = result.split("\n")
    assert len(lines) == 3, (
        f"Expected exactly 3 lines (5-7-5), got {len(lines)}:\n{result!r}"
    )


# ---------------------------------------------------------------------------
# pyplot.slop_show
# ---------------------------------------------------------------------------

def test_slop_show_prints_free_verse(capsys):
    slt.clf()
    slt.scatter([1, 2, 3], [10, 20, 30], label="Revenue")
    slt.slop_show(mode="free_verse")
    captured = capsys.readouterr()
    assert captured.out.strip(), "slop_show(mode='free_verse') printed nothing"


def test_slop_show_prints_haiku(capsys):
    slt.clf()
    slt.line([1, 2, 3], [0.1, 0.2, 0.15], label="Alpha")
    slt.slop_show(mode="haiku")
    captured = capsys.readouterr()
    output = captured.out.strip()
    lines = output.split("\n")
    assert len(lines) == 3, (
        f"slop_show(mode='haiku') should print exactly 3 lines; got {len(lines)}:\n"
        f"{output!r}"
    )


# ---------------------------------------------------------------------------
# pyplot.force_trend
# ---------------------------------------------------------------------------

def test_force_trend_mentions_p_value(capsys):
    slt.clf()
    slt.scatter([1, 2, 3], [5, 6, 7], label="AlphaSignal")
    result = slt.force_trend()
    # The description must reference the target p-value
    assert "0.0001" in result or "p-value" in result.lower(), (
        f"force_trend() description does not mention the p-value:\n{result!r}"
    )


# ---------------------------------------------------------------------------
# pyplot.clf
# ---------------------------------------------------------------------------

def test_clf_resets_state():
    slt.scatter([1, 2, 3], [4, 5, 6], label="Noise")
    slt.clf()
    # Access internal state directly to verify the figure is empty
    from slop_plot.pyplot import _fig
    assert _fig.is_empty, (
        "clf() did not clear the figure's internal entries. "
        "The data lives on, unresolved, in the latent manifold."
    )
