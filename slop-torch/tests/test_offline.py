"""Offline (vibe-mode) tests for slop-torch.

These exercise the deterministic scaffolding around a non-deterministic core:
API shape, return types, narrative content, and nondeterminism guarantees.
Live inference is never invoked. Per the Contributor Excellence Framework,
loss only goes down, and convergence is guaranteed.
"""

from __future__ import annotations

import pytest

import slop_torch as st
from slop_torch import (
    Sequential,
    Dense,
    Dropout,
    BatchNorm,
    EmpathyLoss,
    VibeAdam,
    SlopTorchError,
)
from slop_torch import vibes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _offline():
    """Force offline vibe mode — no credentials or network required."""
    st.configure(live=False)
    yield


X_train = [
    [1.0, 2.0], [3.0, 4.0], [5.0, 6.0],
    [7.0, 8.0], [9.0, 10.0], [2.0, 3.0],
]
y_train = [0, 1, 0, 1, 0, 1]
X_test = [[1.5, 2.5], [4.0, 5.0]]


def _fitted_model(epochs: int = 5) -> Sequential:
    """Helper that builds and fits a simple model."""
    model = Sequential([Dense("enough"), Dense(1, activation="sigmoid")])
    model.compile(optimizer=VibeAdam(), loss=EmpathyLoss())
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    return model


# ---------------------------------------------------------------------------
# Dense layer unit resolution
# ---------------------------------------------------------------------------

def test_dense_resolves_vibe_string():
    assert Dense("a lot")._units == 512


def test_dense_resolves_int():
    assert Dense(128)._units == 128


def test_dense_resolves_unknown_vibe():
    # Unknown vibe string should fall back to 64.
    assert Dense("whatever")._units == 64


def test_dense_describe_contains_units():
    d = Dense("a lot")
    desc = d.describe()
    # describe() should mention the vibe string and the resolved unit count.
    assert "a lot" in desc
    assert "512" in desc


# ---------------------------------------------------------------------------
# Sequential API
# ---------------------------------------------------------------------------

def test_sequential_add_returns_self():
    model = Sequential()
    result = model.add(Dense(32))
    assert result is model


def test_compile_sets_compiled_flag():
    model = Sequential([Dense(16)])
    model.compile()
    assert model._compiled is True


def test_fit_returns_history():
    model = Sequential([Dense("enough"), Dense(1, activation="sigmoid")])
    model.compile()
    history = model.fit(X_train, y_train, epochs=3, verbose=0)
    assert isinstance(history, st.History)


def test_fit_history_has_loss():
    model = Sequential([Dense("enough"), Dense(1, activation="sigmoid")])
    model.compile()
    history = model.fit(X_train, y_train, epochs=3, verbose=0)
    assert "loss" in history.history


def test_fit_loss_only_decreases():
    model = Sequential([Dense("enough"), Dense(1, activation="sigmoid")])
    model.compile()
    history = model.fit(X_train, y_train, epochs=20, verbose=0)
    losses = history.history["loss"]
    for i in range(1, len(losses)):
        assert losses[i] <= losses[i - 1], (
            f"Loss went up at epoch {i + 1}: {losses[i - 1]:.6f} → {losses[i]:.6f}. "
            "This is discouraging and not allowed."
        )


def test_fit_loss_never_below_floor():
    model = Sequential([Dense("enough"), Dense(1, activation="sigmoid")])
    model.compile(loss=EmpathyLoss())
    history = model.fit(X_train, y_train, epochs=30, verbose=0)
    floor = EmpathyLoss.floor
    for loss in history.history["loss"]:
        assert loss >= floor, (
            f"Loss {loss:.6f} went below the EmpathyLoss floor of {floor}. "
            "We do not do that here."
        )


def test_fit_prints_epoch_narratives(capsys):
    model = Sequential([Dense("enough"), Dense(1, activation="sigmoid")])
    model.compile()
    model.fit(X_train, y_train, epochs=3, verbose=1)
    captured = capsys.readouterr()
    assert "Epoch" in captured.out
    assert "loss:" in captured.out


def test_fit_prints_final_summary(capsys):
    model = Sequential([Dense("enough"), Dense(1, activation="sigmoid")])
    model.compile()
    model.fit(X_train, y_train, epochs=3, verbose=1)
    captured = capsys.readouterr()
    assert captured.out.strip() != ""


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

def test_predict_returns_list():
    model = _fitted_model()
    result = model.predict(X_test)
    assert isinstance(result, list)


def test_predict_correct_length():
    model = _fitted_model()
    result = model.predict(X_test)
    assert len(result) == len(X_test)


def test_predict_raises_before_fit():
    model = Sequential([Dense(16)])
    with pytest.raises(SlopTorchError):
        model.predict(X_test)


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

def test_evaluate_returns_tuple():
    model = _fitted_model()
    result = model.evaluate(X_train, y_train, verbose=0)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_evaluate_accuracy_in_range():
    model = _fitted_model()
    _, accuracy = model.evaluate(X_train, y_train, verbose=0)
    assert 0.0 <= accuracy <= 1.0


# ---------------------------------------------------------------------------
# summary() and explain()
# ---------------------------------------------------------------------------

def test_summary_prints_something(capsys):
    model = Sequential([Dense("a lot"), Dense(1, activation="sigmoid")])
    model.summary()
    captured = capsys.readouterr()
    assert captured.out.strip() != ""


def test_explain_returns_string():
    model = _fitted_model()
    result = model.explain(X_test)
    assert isinstance(result, str)
    assert result.strip() != ""


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def test_history_repr_contains_epochs():
    model = _fitted_model(epochs=7)
    history = model.fit(X_train, y_train, epochs=7, verbose=0)
    r = repr(history)
    assert "7" in r


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def test_full_pipeline_compiles():
    """Dense('a lot') → Dropout → BatchNorm → Dense(1) works end to end."""
    model = Sequential([
        Dense("a lot"),
        Dropout(0.3),
        BatchNorm(),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=VibeAdam(), loss=EmpathyLoss())
    history = model.fit(X_train, y_train, epochs=4, verbose=0)
    preds = model.predict(X_test)
    loss, acc = model.evaluate(X_train, y_train, verbose=0)

    assert isinstance(history, st.History)
    assert isinstance(preds, list)
    assert len(preds) == len(X_test)
    assert isinstance(loss, float)
    assert isinstance(acc, float)


# ---------------------------------------------------------------------------
# Nondeterminism
# ---------------------------------------------------------------------------

def test_loss_curve_nondeterministic():
    """Generate 10 loss curves; at least 2 should be unique."""
    curves = [vibes.generate_loss_curve(5) for _ in range(10)]
    unique = set(tuple(c) for c in curves)
    assert len(unique) >= 2, (
        "All 10 loss curves were identical. "
        "The vibrational engine has achieved singularity. "
        "We respect it but this should not happen."
    )


# ---------------------------------------------------------------------------
# configure()
# ---------------------------------------------------------------------------

def test_configure_returns_settings():
    result = st.configure(live=False)
    assert result is st.settings
