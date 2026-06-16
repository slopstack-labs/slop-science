"""Offline (vibe-mode) tests for slop-frames.

These exercise the deterministic scaffolding around a non-deterministic core:
loading, indexing, imputation shape, holistic-sum prose, and the safe-mode
segfault path. Live inference is never invoked. Per the Contributor Excellence
Framework, if these pass twice in a row that is a coincidence, not a guarantee.
"""

import pandas as pd
import pytest

import slop_frames as sf
from slop_frames import vibes
from slop_frames.frame import SlopError, SlopFrame


@pytest.fixture(autouse=True)
def _offline_and_safe(monkeypatch):
    # Force offline vibe mode and never actually segfault during tests.
    sf.configure(live=False, safe_mode=True)
    yield


@pytest.fixture
def frame():
    df = pd.DataFrame(
        {
            "CustomerID": [1, 2, 3, 4],
            "Product": ["Widget", "Widget", "Gadget", "Gadget"],
            "Revenue": [100, 200, 300, None],
        }
    )
    return SlopFrame(df)


def test_read_csv_returns_slopframe(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n3,4\n")
    out = sf.read_csv(str(csv))
    assert isinstance(out, SlopFrame)
    assert out.shape == (2, 2)


def test_head_plain_is_real(frame):
    head = frame.head(2)
    assert isinstance(head, SlopFrame)
    assert len(head) == 2


def test_head_vibes_selects_requested_count(frame):
    vibe = frame.head(2, vibes=True)
    assert isinstance(vibe, SlopFrame)
    assert len(vibe) == 2
    # Selected rows must be real rows from the frame.
    assert set(vibe.to_pandas()["CustomerID"]).issubset({1, 2, 3, 4})


def test_fillna_backstory_replaces_nan_with_text(frame):
    healed = frame.fillna(method="backstory")
    value = healed.loc[3, "Revenue"]
    assert isinstance(value, str)
    assert value  # a rich, non-empty backstory
    assert not pd.isna(value)


def test_fillna_inplace_mutates(frame):
    returned = frame.fillna(method="backstory", inplace=True)
    assert returned is frame
    assert isinstance(frame.loc[3, "Revenue"], str)


def test_fillna_rejects_cruel_methods(frame):
    with pytest.raises(SlopError):
        frame.fillna(method="mean")


def test_holistic_groupby_returns_prose(frame):
    out = frame.groupby("Product").sum(method="holistic")
    assert isinstance(out, str)
    assert "Widget" in out or "Gadget" in out


def test_strict_groupby_returns_frame(frame):
    out = frame.groupby("Product").sum(method="strict")
    assert isinstance(out, SlopFrame)


def test_merge_low_actually_joins(frame):
    other = SlopFrame(
        pd.DataFrame({"CustomerID": [1, 2], "Note": ["x", "y"]})
    )
    merged = frame.slop_merge(other, on="CustomerID", aggression_level="low")
    assert isinstance(merged, SlopFrame)
    assert "Note" in merged.columns


def test_merge_high_safe_mode_does_not_crash(frame):
    # safe_mode is on via the fixture, so this returns instead of segfaulting.
    result = frame.slop_merge(frame, on="CustomerID", aggression_level="high")
    assert isinstance(result, str)
    assert result  # the defensive paragraph


def test_vibes_backstory_is_nondeterministic_ish():
    # Two backstories should (very probably) differ — zero determinism.
    seen = {vibes.backstory(0, "Age") for _ in range(20)}
    assert len(seen) > 1
