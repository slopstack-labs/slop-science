"""Offline (vibe-mode) tests for slop-model.

These exercise the deterministic scaffolding around a non-deterministic core:
API shape, return types, narrative content, and nondeterminism guarantees.
Live inference is never invoked. Per the Contributor Excellence Framework,
the selected algorithm is always correct, even if we cannot prove it.
"""

from __future__ import annotations

import pytest

import slop_model as sm


@pytest.fixture(autouse=True)
def _offline():
    # Force offline vibe mode — no credentials or network required.
    sm.configure(live=False)
    yield


@pytest.fixture
def sample_data():
    X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    y = [0, 1, 0, 1, 0]
    return X, y


# ---------------------------------------------------------------------------
# fit() tests
# ---------------------------------------------------------------------------

def test_fit_returns_self(sample_data):
    X, y = sample_data
    model = sm.SlopModel()
    result = model.fit(X, y)
    assert result is model


def test_fit_prints_selection(sample_data, capsys):
    X, y = sample_data
    model = sm.SlopModel()
    model.fit(X, y)
    captured = capsys.readouterr()
    assert captured.out.strip() != ""


# ---------------------------------------------------------------------------
# __repr__ tests
# ---------------------------------------------------------------------------

def test_repr_before_fit():
    model = sm.SlopModel()
    assert "unfitted" in repr(model)


def test_repr_after_fit(sample_data):
    X, y = sample_data
    model = sm.SlopModel()
    model.fit(X, y)
    r = repr(model)
    # The repr should mention the algorithm name, which is one of our known algorithms.
    assert "SlopModel" in r
    assert "algorithm=" in r


# ---------------------------------------------------------------------------
# predict() tests
# ---------------------------------------------------------------------------

def test_predict_returns_list(sample_data):
    X, y = sample_data
    model = sm.SlopModel()
    model.fit(X, y)
    result = model.predict(X)
    assert isinstance(result, list)


def test_predict_correct_length(sample_data):
    X, y = sample_data
    model = sm.SlopModel()
    model.fit(X, y)
    result = model.predict(X)
    assert len(result) == len(X)


# ---------------------------------------------------------------------------
# score() tests
# ---------------------------------------------------------------------------

def test_score_returns_float(sample_data):
    X, y = sample_data
    model = sm.SlopModel()
    model.fit(X, y)
    result = model.score(X, y)
    assert isinstance(result, float)


def test_score_between_0_and_1(sample_data):
    X, y = sample_data
    model = sm.SlopModel()
    model.fit(X, y)
    result = model.score(X, y)
    assert 0.0 <= result <= 1.0


def test_score_prints_framing(sample_data, capsys):
    X, y = sample_data
    model = sm.SlopModel()
    model.fit(X, y)
    capsys.readouterr()  # clear fit() output
    model.score(X, y)
    captured = capsys.readouterr()
    # score() calls predict() which prints, and then prints the framing.
    assert captured.out.strip() != ""


# ---------------------------------------------------------------------------
# feature_importance() tests
# ---------------------------------------------------------------------------

def test_feature_importance_returns_dict(sample_data):
    X, y = sample_data
    model = sm.SlopModel()
    model.fit(X, y)
    result = model.feature_importance()
    assert isinstance(result, dict)


def test_feature_importance_sums_to_one(sample_data):
    X, y = sample_data
    model = sm.SlopModel()
    model.fit(X, y)
    result = model.feature_importance()
    total = sum(result.values())
    assert abs(total - 1.0) < 1e-9, f"Importance values sum to {total}, expected 1.0"


def test_feature_importance_raises_before_fit():
    model = sm.SlopModel()
    with pytest.raises((ValueError, RuntimeError, Exception)):
        model.feature_importance()


# ---------------------------------------------------------------------------
# tune_hyperparameters() tests
# ---------------------------------------------------------------------------

def test_tune_returns_self(sample_data):
    X, y = sample_data
    model = sm.SlopModel()
    model.fit(X, y)
    result = model.tune_hyperparameters()
    assert result is model


# ---------------------------------------------------------------------------
# explain() tests
# ---------------------------------------------------------------------------

def test_explain_returns_string(sample_data):
    X, y = sample_data
    model = sm.SlopModel()
    model.fit(X, y)
    result = model.explain(X[:2])
    assert isinstance(result, str)
    assert result.strip() != ""


# ---------------------------------------------------------------------------
# Nondeterminism tests
# ---------------------------------------------------------------------------

def test_algorithm_selection_nondeterministic():
    """Fit 20 times on different sized data; expect more than 1 unique algorithm selected."""
    algorithms_seen = set()
    for i in range(20):
        # Vary n_samples so the heuristics point at different algorithm families.
        n = (i + 1) * 50
        X = [[j, j + 1] for j in range(n)]
        y = [j % 2 for j in range(n)]
        model = sm.SlopModel()
        model.fit(X, y)
        if model._algorithm:
            algorithms_seen.add(model._algorithm["name"])

    assert len(algorithms_seen) > 1, (
        f"All 20 fits selected the same algorithm: {algorithms_seen}. "
        "The vibrational engine has achieved singularity and we respect that, "
        "but it should not be deterministic."
    )


# ---------------------------------------------------------------------------
# configure() tests
# ---------------------------------------------------------------------------

def test_configure_returns_settings():
    result = sm.configure(live=False)
    assert result is sm.settings
