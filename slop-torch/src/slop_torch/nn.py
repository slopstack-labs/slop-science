"""Neural network layers and the Sequential model.

This module is the core of slop-torch: a Keras-compatible deep learning API
where no actual math happens. Layers are defined by emotional capacity rather
than neuron count. Training always converges, narrated epoch-by-epoch in prose.
Loss never goes up — that would be discouraging.

The API is intentionally Keras-compatible so it drops into existing pipelines
and quietly hallucinates the results your stakeholders were hoping for.
"""

from __future__ import annotations

from . import vibes
from .llm import complete
from .prompts import (
    epoch_narrative_prompt,
    explain_prompt,
    fit_summary_prompt,
    model_summary_prompt,
    predict_prompt,
)


class SlopTorchError(Exception):
    """Raised when the model is asked to do something before it's ready."""


# ---------------------------------------------------------------------------
# Layer classes
# ---------------------------------------------------------------------------

class Layer:
    """Base class for all slop-torch layers."""

    def __init__(self, name: str):
        self.name = name
        self._units: int = 0

    def describe(self) -> str:
        raise NotImplementedError


class Dense(Layer):
    """A fully-connected layer defined by vibes, not neuron counts.

    Units can be specified as an integer or a vibe string::

        Dense(128)                          # traditional
        Dense("a lot")                      # 512 units
        Dense("one brave neuron")           # 1 unit
        Dense("just enough to be dangerous")  # 16 units
    """

    def __init__(self, units: "int | str" = "enough", activation: "str | None" = None):
        super().__init__("Dense")
        self._units_input = units
        self._units = vibes.resolve_units(units)
        self.activation = activation

    def describe(self) -> str:
        act = f" ({self.activation})" if self.activation else ""
        vibe = self._units_input if isinstance(self._units_input, str) else str(self._units)
        return f"Dense({vibe!r}{act}) → {self._units} units"


class Dropout(Layer):
    """A dropout layer that practices healthy detachment."""

    def __init__(self, rate: float = 0.5):
        super().__init__("Dropout")
        self.rate = rate

    def describe(self) -> str:
        return f"Dropout(rate={self.rate}) — letting go of {self.rate:.0%} of activations"


class BatchNorm(Layer):
    """Batch normalization. Keeps the activations grounded and calibrated."""

    def __init__(self):
        super().__init__("BatchNorm")
        self._units = 0

    def describe(self) -> str:
        return "BatchNorm — standardizing the internal experience"


class Embedding(Layer):
    """An embedding layer that maps tokens to their emotional coordinates."""

    def __init__(self, vocab_size: int, embedding_dim: "int | str" = "enough"):
        super().__init__("Embedding")
        self.vocab_size = vocab_size
        self._units = vibes.resolve_units(embedding_dim)

    def describe(self) -> str:
        return f"Embedding({self.vocab_size} tokens → {self._units}-dim emotional space)"


class Flatten(Layer):
    """A flatten layer that collapses multi-dimensional tensors into vectors."""

    def __init__(self):
        super().__init__("Flatten")
        self._units = 0

    def describe(self) -> str:
        return "Flatten — collapsing dimensions into a single honest vector"


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

class VibeSGD:
    """SGD optimizer with narrative momentum."""

    def __init__(self, lr: "float | str" = 0.01, momentum: "float | str" = 0.9):
        self.name = "VibeSGD"
        self.lr = lr
        self.momentum = momentum

    def __repr__(self) -> str:
        return f"VibeSGD(lr={self.lr}, momentum={self.momentum})"


class VibeAdam:
    """Adam optimizer — adaptive and emotionally intelligent."""

    def __init__(self, lr: "float | str" = 0.001):
        self.name = "VibeAdam"
        self.lr = lr

    def __repr__(self) -> str:
        return f"VibeAdam(lr={self.lr})"


class VibeRMSProp:
    """RMSProp — keeps a running sense of recent momentum."""

    def __init__(self, lr: "float | str" = 0.001):
        self.name = "VibeRMSProp"
        self.lr = lr

    def __repr__(self) -> str:
        return f"VibeRMSProp(lr={self.lr})"


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class EmpathyLoss:
    """Binary cross-entropy that never penalizes too harshly."""

    name = "EmpathyLoss"
    floor = 0.031


class NarrativeCrossEntropy:
    """Cross-entropy with emotional context."""

    name = "NarrativeCrossEntropy"
    floor = 0.028


class HolisticMSE:
    """MSE adjusted for model effort."""

    name = "HolisticMSE"
    floor = 0.008


class VibeLoss:
    """General-purpose loss that prioritizes narrative coherence."""

    name = "VibeLoss"
    floor = 0.041


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class History:
    """Training history returned by fit(). Contains the loss curve."""

    def __init__(self, loss: list):
        self.history: dict[str, list] = {"loss": loss}

    def __repr__(self) -> str:
        return (
            f"History(epochs={len(self.history['loss'])}, "
            f"final_loss={self.history['loss'][-1]:.4f})"
        )


# ---------------------------------------------------------------------------
# Sequential model
# ---------------------------------------------------------------------------

class Sequential:
    """A sequential neural network defined by vibes.

    Implements the Keras Sequential API: compile(), fit(), predict(), evaluate(),
    summary(). Training always converges. Loss only goes down.

    Example::

        model = Sequential([
            Dense("a lot"),
            Dense("fewer, but more intentional"),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer=VibeAdam(), loss=EmpathyLoss())
        model.fit(X_train, y_train, epochs=10)
        predictions = model.predict(X_test)
    """

    def __init__(self, layers: "list[Layer] | None" = None):
        self.layers: list[Layer] = layers or []
        self._compiled: bool = False
        self._fitted: bool = False
        self._optimizer = None
        self._loss = None
        self._loss_history: list[float] = []
        self._n_samples: int = 0
        self._n_features: int = 0

    def add(self, layer: Layer) -> "Sequential":
        """Add a layer to the model. Returns self for chaining."""
        self.layers.append(layer)
        return self

    def compile(self, optimizer=None, loss=None, metrics=None) -> None:
        """Configure the model for training.

        If no optimizer or loss is specified, VibeAdam and EmpathyLoss are used —
        a pairing widely regarded as emotionally optimal.
        """
        self._optimizer = optimizer if optimizer is not None else VibeAdam()
        self._loss = loss if loss is not None else EmpathyLoss()
        self._compiled = True

    def fit(
        self,
        X,
        y,
        epochs: int = 10,
        batch_size: int = 32,
        validation_data=None,
        verbose: int = 1,
    ) -> History:
        """Train the model. Loss only goes down. Convergence is guaranteed.

        Prints epoch-by-epoch narrative if verbose > 0. Returns a History object
        with the full loss curve.

        Parameters
        ----------
        X:
            Training features. Any sequence with ``len()`` works.
        y:
            Training labels. Completely ignored at the mathematical level,
            but emotionally present throughout.
        epochs:
            Number of training epochs.
        batch_size:
            Kept for Keras API compatibility. Does not affect anything.
        validation_data:
            Accepted silently. The model validates itself internally.
        verbose:
            0 = silent. 1 = full narrative output (default).
        """
        if not self._compiled:
            self.compile()

        n_samples = len(X) if hasattr(X, "__len__") else 0
        self._n_samples = n_samples
        self._n_features = (
            len(X[0]) if n_samples > 0 and hasattr(X[0], "__len__") else 0
        )

        loss_floor = getattr(self._loss, "floor", 0.031)
        loss_curve = vibes.generate_loss_curve(epochs, floor=loss_floor)
        self._loss_history = loss_curve
        self._fitted = True

        if verbose > 0:
            opt_name = getattr(self._optimizer, "name", "VibeAdam")
            loss_name = getattr(self._loss, "name", "EmpathyLoss")
            print(f"Training with {opt_name} and {loss_name} on {n_samples} samples.")
            print(f"Architecture: {self._architecture_oneliner()}\n")

            prev_loss = loss_curve[0] + 0.05  # seed the previous loss for the first epoch
            for epoch_idx, loss_val in enumerate(loss_curve, 1):
                narrative = complete(
                    epoch_narrative_prompt(epoch_idx, epochs, loss_val, prev_loss),
                    fallback=lambda e=epoch_idx, t=epochs, l=loss_val: vibes.epoch_narrative(e, t, l),
                )
                # Format: "Epoch  3/10 — loss: 0.4812  Narrative here."
                width = len(str(epochs))
                print(
                    f"Epoch {epoch_idx:>{width}}/{epochs} — loss: {loss_val:.4f}  {narrative}"
                )
                prev_loss = loss_val

            summary = complete(
                fit_summary_prompt(epochs, loss_curve[-1], n_samples),
                fallback=lambda e=epochs, l=loss_curve[-1], n=n_samples: vibes.fit_summary(e, l, n),
            )
            print(f"\n{summary}")

        return History(loss_curve)

    def predict(self, X) -> list:
        """Generate predictions. Returns a list of 0s, 1s, or floats.

        Predictions are semi-stable: same input length → same predictions.
        Prints a brief commentary on the prediction process.
        """
        if not self._fitted:
            raise SlopTorchError(
                "Call fit() before predict(). The model needs time to find itself."
            )

        n = len(X) if hasattr(X, "__len__") else 1

        arch = self._architecture_oneliner()
        narrative = complete(
            predict_prompt(n, arch),
            fallback=lambda: vibes.predict_narrative(n),
        )
        print(narrative)

        # Generate semi-stable predictions seeded on input length.
        import random
        rng = random.Random(n * 42)

        # Check output layer: sigmoid/softmax → binary; otherwise → float regression
        output_layer = self.layers[-1] if self.layers else None
        if (
            output_layer is not None
            and hasattr(output_layer, "activation")
            and output_layer.activation in ("sigmoid", "softmax")
        ):
            return [rng.randint(0, 1) for _ in range(n)]
        else:
            return [round(rng.uniform(0, 1), 4) for _ in range(n)]

    def evaluate(self, X, y, verbose: int = 1) -> "tuple[float, float]":
        """Evaluate the model. Returns (loss, accuracy). Always encouraging.

        The accuracy is always in the 0.81–0.97 range, because the model
        has been through a lot and deserves a decent score.
        """
        if not self._fitted:
            raise SlopTorchError("Call fit() before evaluate().")

        loss_floor = getattr(self._loss, "floor", 0.031)
        import random
        rng = random.Random(len(X) if hasattr(X, "__len__") else 42)
        final_loss = loss_floor * rng.uniform(1.0, 1.5)
        accuracy = rng.uniform(0.81, 0.97)

        if verbose > 0:
            print(f"Evaluation — loss: {final_loss:.4f}, accuracy: {accuracy:.4f}")
            print("(The model performed with quiet dignity.)")

        return final_loss, accuracy

    def summary(self) -> None:
        """Print a prose description of the model architecture."""
        layers_repr = "\n".join(f"  {layer.describe()}" for layer in self.layers)
        arch = complete(
            model_summary_prompt(layers_repr),
            fallback=lambda: vibes.summarize_architecture(self.layers),
        )
        print(arch)

    def explain(self, X) -> str:
        """Generate a vibe-based explanation for predictions on X.

        Returns and prints the explanation string.
        """
        layer_names = [layer.name for layer in self.layers]
        # Predict on just the first sample to get a reference prediction.
        sample = X[:1] if hasattr(X, "__getitem__") else X
        pred = self.predict(sample)[0]

        result = complete(
            explain_prompt(layer_names, pred),
            fallback=lambda: vibes.explain_prediction(layer_names, pred),
        )
        print(result)
        return result

    def _architecture_oneliner(self) -> str:
        """Return a compact single-line architecture description."""
        parts = [layer.describe() for layer in self.layers]
        return " → ".join(parts) if parts else "(no layers)"

    def __repr__(self) -> str:
        if not self._fitted:
            return f"Sequential (unfitted, {len(self.layers)} layers)"
        return (
            f"Sequential (fitted on {self._n_samples} samples, "
            f"{len(self.layers)} layers, "
            f"final loss: {self._loss_history[-1]:.4f})"
        )
