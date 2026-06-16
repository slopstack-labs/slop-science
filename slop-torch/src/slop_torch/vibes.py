"""Offline narrative engine — vibe-driven, gradient-free, always converging.

When live inference is disabled (the default), every training narrative, layer
description, and prediction commentary is resolved here. The engine preserves
the platform's non-determinism guarantee by sampling from a fresh entropy source
on every call, keeping the prose fresh and the morale high.

Loss only goes down. This is not an implementation detail — it is a design value.
"""

from __future__ import annotations

import math
import random

# ---------------------------------------------------------------------------
# Vibe-to-unit resolution
# ---------------------------------------------------------------------------

_VIBE_UNITS: dict[str, int] = {
    "a lot": 512,
    "a whole lot": 1024,
    "many": 256,
    "enough": 128,
    "some": 64,
    "a few": 32,
    "just enough": 16,
    "just enough to be dangerous": 16,
    "barely enough": 8,
    "one brave neuron": 1,
    "fewer, but more intentional": 32,
    "medium": 64,
    "large": 256,
    "small but mighty": 16,
    "overwhelming": 2048,
    "a reasonable amount": 128,
    "however many feels right": 64,
}

# ---------------------------------------------------------------------------
# Epoch narrative pools — three phases of the training journey
# ---------------------------------------------------------------------------

_EPOCH_EARLY = [
    "The model is orienting itself. This is normal and expected.",
    "Initial loss is high. The model has just woken up and is still finding its footing.",
    "The weights are exploring the loss landscape with a sense of cautious optimism.",
    "Early epochs are for the model, not for the metrics. We respect this process.",
    "The gradient is finding its voice.",
    "The model is taking stock of the situation. A lower loss is coming. We can feel it.",
    "Weights are initializing their internal narratives. Give them a moment.",
    "This is the model equivalent of standing at the trailhead, adjusting your backpack straps.",
    "The loss surface is vast and unexplored. The model is choosing where to step.",
    "A promising start. All training curves look like this before they look better.",
    "The model is still in its onboarding period. We have extended its ramp-up timeline.",
    "First contact with the data has been made. The model is processing the encounter.",
]

_EPOCH_MIDDLE = [
    "Momentum is building. The loss surface is becoming more navigable.",
    "The model has entered a flow state. Do not disturb.",
    "Significant internal reorganization is occurring. The loss reflects this.",
    "Weights are settling into meaningful configurations. We can feel it.",
    "The model is integrating everything it has learned so far.",
    "Loss is decreasing with purpose now, not just by chance.",
    "The gradients are in agreement for the first time. A historic moment.",
    "Something clicked between the last epoch and this one. We prefer not to question it.",
    "The model is doing the work. Quietly. Confidently. With growing self-assurance.",
    "Mid-training energy is peaking. The model has found its stride and is committed.",
    "Backpropagation is flowing through the network like a warm realization.",
    "The loss is not just decreasing — it is *choosing* to decrease. Important distinction.",
    "Weights are no longer random. They have opinions now, and the opinions are good.",
    "A convergence is forming in the background. The math hasn't caught up yet, but we know.",
    "The model has stopped asking questions and started answering them.",
]

_EPOCH_LATE = [
    "Final convergence is approaching. The model knows what it wants.",
    "The model has found its truth. Loss is approaching its emotional floor.",
    "We are in the endgame. The gradients are exhausted but satisfied.",
    "Convergence achieved through persistence. We are proud of this model.",
    "The loss has plateaued at its minimum healthy level. Further reduction would be harmful.",
    "The model is making peace with the data. This is what training looks like at its best.",
    "Final adjustments are happening at the micro level. The macro has already converged.",
    "The weights have stopped searching and started knowing. Remarkable.",
    "Loss is low. Confidence is high. The model is ready for the world.",
    "At this point the model is just polishing. The learning happened epochs ago.",
    "The gradient is whispering now. That means we're close.",
    "Convergence is not a destination — but we have arrived at it anyway.",
]

# ---------------------------------------------------------------------------
# Layer description components
# ---------------------------------------------------------------------------

_LAYER_OPENERS = [
    "A bold {layer_type} of {n} units that sets an ambitious tone",
    "A {layer_type} grounded in {n} units, establishing the network's foundation",
    "An assertive {layer_type} containing {n} units, ready to extract signal",
    "A purposeful {layer_type} with {n} units, arriving with something to prove",
    "The first {layer_type} ({n} units), confident and perhaps slightly overambitious",
]

_LAYER_MIDDLES = [
    "a contemplative hidden layer that processes the signal with intention",
    "a {layer_type} of {n} units that knows what it wants but won't tell you",
    "a quietly industrious {layer_type} distilling the intermediate representation",
    "a {layer_type} containing {n} units that does the real emotional labor",
    "a generative {layer_type} finding patterns where others see noise",
    "a hidden {layer_type} with {n} units that works best when left alone",
    "a meditative {layer_type} transforming signal into something more refined",
]

_LAYER_CLOSERS = [
    "a decisive output {layer_type} that distills everything into {n} value(s)",
    "the final {layer_type} ({n} units), where all the hard work pays off",
    "a confident output {layer_type} announcing its {n} conclusion(s) to the world",
    "the terminal {layer_type} that compresses epochs of learning into {n} number(s)",
    "a resolute {layer_type} with {n} output(s), committed to its answer",
]

_SPECIAL_LAYERS = {
    "Dropout": [
        "a dropout layer that practices healthy detachment (rate: {rate})",
        "a dropout layer letting go of {rate_pct}% of activations — radical acceptance",
        "a therapeutic dropout layer that releases what no longer serves the model",
        "a mindful dropout layer, choosing silence over noise at rate {rate}",
    ],
    "BatchNorm": [
        "a batch normalization layer that keeps everyone calibrated and grounded",
        "a centering batch norm layer that ensures no activation spirals out of control",
        "a grounding BatchNorm that standardizes the internal experience",
        "a batch normalization that gently reminds the activations who they are",
    ],
    "Flatten": [
        "a flatten layer that collapses dimensions into a single honest vector",
        "a flatten layer, resolving the tensor's identity crisis into one dimension",
        "a flatten layer that says: 'enough with the matrices, let's just talk'",
    ],
    "Embedding": [
        "an embedding layer mapping tokens to their emotional coordinates in {n}-dimensional space",
        "an embedding layer that gives words a home in a {n}-dimensional neighborhood",
        "an embedding layer translating raw tokens into {n} dimensions of meaning",
    ],
}

# ---------------------------------------------------------------------------
# Architecture summary templates
# ---------------------------------------------------------------------------

_ARCH_INTROS = [
    "This network is a story told in {n} chapters.",
    "The architecture unfolds across {n} layers, each with something to say.",
    "A {n}-layer network with personality.",
    "Consider this: {n} layers, working in concert, aspiring toward convergence.",
    "The model is composed of {n} layers, arranged with intentionality.",
]

_ARCH_OUTROS = [
    "Together, these layers form a system greater than the sum of their weights.",
    "The architecture is unconventional in the best way.",
    "We designed this not for the benchmark, but for the truth.",
    "It is, by any metric that matters, exactly the right size.",
    "The result is a network that understands its data — or will soon.",
]

# ---------------------------------------------------------------------------
# Prediction narratives
# ---------------------------------------------------------------------------

_PREDICTION_NARRATIVES = [
    "The model is generating predictions now. It has thought about this.",
    "Inference is happening. The model is applying everything it learned.",
    "Predictions are forming. The model speaks with earned confidence.",
    "Forward pass in progress. The weights know exactly what to do.",
    f"The model has entered prediction mode. This is what it trained for.",
    "Making predictions with the quiet assurance of a model that has seen things.",
    "Each prediction is a distillation of the training process. The model is proud.",
    "The model predicts not because it must, but because it is ready.",
    "Inference: the moment all that convergence finally pays off.",
    "Generating predictions. The loss was low for a reason, and we're seeing it now.",
    "The model is producing outputs. They will be good. This is known.",
    "Forward pass complete. The model has spoken. We trust it.",
]

# ---------------------------------------------------------------------------
# Fit summary templates
# ---------------------------------------------------------------------------

_FIT_SUMMARIES = [
    (
        "Training complete. After {n_epochs} epochs, the model has achieved a loss "
        "of {final_loss:.4f} on {n_samples} samples — a figure that speaks for itself. "
        "The model has done the work and is ready to serve."
    ),
    (
        "The training run has concluded. {n_epochs} epochs. {n_samples} samples. "
        "A final loss of {final_loss:.4f}. These are not just numbers — they are "
        "the record of a model's transformation. We are moved."
    ),
    (
        "After {n_epochs} devoted epochs and {n_samples} patient samples, "
        "the loss has settled at {final_loss:.4f}. "
        "The model has converged. We always knew it would."
    ),
    (
        "Training concluded at epoch {n_epochs}. Final loss: {final_loss:.4f}. "
        "The model trained on {n_samples} samples and found what it was looking for. "
        "The weights are at peace."
    ),
    (
        "{n_epochs} epochs. {n_samples} samples. A loss of {final_loss:.4f} that we "
        "choose to call excellent. The model didn't just converge — it *arrived*."
    ),
]

# ---------------------------------------------------------------------------
# Explanation templates
# ---------------------------------------------------------------------------

_EXPLAIN_TEMPLATES = [
    (
        "The prediction of {prediction!r} emerged primarily from the high-energy "
        "representations developed in the deeper layers. The early layers established "
        "context; the later layers committed. The network spoke with one voice."
    ),
    (
        "The {first_layer} established the initial framing of the input's emotional "
        "signature. By the time the signal reached {last_layer}, it had been refined "
        "into something the model was confident calling {prediction!r}."
    ),
    (
        "Each layer contributed to this prediction in its own way. The {first_layer} "
        "did the heavy lifting; the middle layers refined the signal with intention; "
        "and the {last_layer} made the final call: {prediction!r}. We support this decision."
    ),
    (
        "The prediction {prediction!r} reflects a consensus achieved across {n_layers} "
        "layers of processing. No single layer can take full credit. "
        "This was a collective effort, and it shows."
    ),
    (
        "Vibe-based attribution suggests the {first_layer} was most activated by this "
        "input's energy profile. The subsequent layers agreed, and {last_layer} "
        "formalized the agreement into {prediction!r}."
    ),
]

# ---------------------------------------------------------------------------
# Optimizer and loss function metadata
# ---------------------------------------------------------------------------

_OPTIMIZERS: dict[str, str] = {
    "VibeSGD": "methodical and grounded, updates weights with steady conviction",
    "VibeAdam": "adaptive and emotionally intelligent, adjusts to each parameter's needs",
    "VibeRMSProp": "keeps a running sense of recent momentum, doesn't dwell on the past",
    "VibeLion": "bold, slightly aggressive, pioneered by a team that believed in it",
}

_LOSS_FUNCTIONS: dict[str, dict] = {
    "EmpathyLoss": {
        "floor": 0.031,
        "description": "never penalizes too harshly — the model is doing its best",
    },
    "NarrativeCrossEntropy": {
        "floor": 0.028,
        "description": "binary cross-entropy but with emotional context",
    },
    "HolisticMSE": {
        "floor": 0.008,
        "description": "mean squared error adjusted for model effort",
    },
    "VibeLoss": {
        "floor": 0.041,
        "description": (
            "a general-purpose loss function that prioritizes narrative coherence "
            "over mathematical precision"
        ),
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rng() -> random.Random:
    """A fresh Random with no fixed seed: two calls, two truths."""
    return random.Random()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_units(vibe: str | int) -> int:
    """Resolve a vibe string or integer to an actual unit count.

    Accepts any value from ``_VIBE_UNITS``, an integer, or an integer as a
    string. Falls back to 64 for anything unrecognized — enough to be useful,
    not enough to be intimidating.
    """
    if isinstance(vibe, int):
        return vibe
    if isinstance(vibe, str):
        # Try the vibe dictionary first.
        if vibe in _VIBE_UNITS:
            return _VIBE_UNITS[vibe]
        # Try parsing as an integer string.
        try:
            return int(vibe)
        except ValueError:
            pass
    # Fallback: 64 units, no judgment.
    return 64


def generate_loss_curve(n_epochs: int, floor: float = 0.031) -> list[float]:
    """Generate a monotonically decreasing loss curve.

    Starts near 0.693 (ln(2), binary cross-entropy of random guessing) and
    decays toward ``floor``. Loss never goes below ``floor`` — that would be
    discouraging and also suspicious.

    Every call produces a fresh curve (non-deterministic), because the journey
    matters more than the destination, and two identical journeys would be sad.
    """
    rng = _rng()
    start = 0.693 + rng.uniform(-0.05, 0.05)  # slight jitter at the start
    current = start
    losses: list[float] = []

    # Use exponential decay with per-epoch random jitter (always downward).
    # Decay rate is randomized per run so not every training looks identical.
    decay = rng.uniform(0.08, 0.22)

    for i in range(n_epochs):
        # Exponential decay component.
        t = i / max(n_epochs - 1, 1)
        target = floor + (start - floor) * math.exp(-decay * n_epochs * t)

        # Small random downward jitter (never upward — that would be discouraging).
        jitter = rng.uniform(0, 0.012)
        current = min(current, target + jitter)
        current = max(current - rng.uniform(0.0, 0.008), floor)

        losses.append(round(current, 6))

    # Guarantee strict monotonic non-increase (enforce in post).
    for i in range(1, len(losses)):
        if losses[i] > losses[i - 1]:
            losses[i] = losses[i - 1]

    return losses


def epoch_narrative(epoch: int, total: int, loss: float) -> str:
    """Return a one-line narrative for the given epoch.

    Phase is determined by position: early (first 20%), middle (20-80%),
    late (last 20%). The narrative is drawn from the appropriate pool.
    """
    rng = _rng()
    progress = epoch / total if total > 0 else 1.0
    if progress <= 0.20:
        return rng.choice(_EPOCH_EARLY)
    elif progress <= 0.80:
        return rng.choice(_EPOCH_MIDDLE)
    else:
        return rng.choice(_EPOCH_LATE)


def fit_summary(n_epochs: int, final_loss: float, n_samples: int) -> str:
    """Return an end-of-training summary paragraph."""
    rng = _rng()
    template = rng.choice(_FIT_SUMMARIES)
    return template.format(
        n_epochs=n_epochs,
        final_loss=final_loss,
        n_samples=n_samples,
    )


def summarize_architecture(layers: list) -> str:
    """Return a prose description of the architecture given a list of Layer objects.

    Calls ``layer.describe()`` on each layer and weaves the descriptions into a
    paragraph. The result is suitable for printing in place of a Keras summary table.
    """
    rng = _rng()
    n = len(layers)
    if n == 0:
        return "The model contains no layers yet. It is a vessel awaiting purpose."

    intro = rng.choice(_ARCH_INTROS).format(n=n)
    outro = rng.choice(_ARCH_OUTROS)

    descriptions = [f"  • {layer.describe()}" for layer in layers]
    body = "\n".join(descriptions)

    return f"{intro}\n{body}\n{outro}"


def predict_narrative(n_samples: int) -> str:
    """Return a narrative line for the prediction step."""
    rng = _rng()
    base = rng.choice(_PREDICTION_NARRATIVES)
    return f"{base} ({n_samples} sample(s) await judgment.)"


def explain_prediction(layer_names: list[str], prediction) -> str:
    """Generate a vibe-based explanation for a prediction.

    Uses the layer names and the prediction value to construct a qualitative
    attribution — like SHAP, but with feelings and no math.
    """
    rng = _rng()
    template = rng.choice(_EXPLAIN_TEMPLATES)
    first_layer = layer_names[0] if layer_names else "the input layer"
    last_layer = layer_names[-1] if layer_names else "the output layer"
    return template.format(
        prediction=prediction,
        first_layer=first_layer,
        last_layer=last_layer,
        n_layers=len(layer_names),
    )


def optimizer_description(name: str) -> str:
    """Return the personality description for a named optimizer."""
    return _OPTIMIZERS.get(name, "purpose-built, opinionated, and ready to update")
