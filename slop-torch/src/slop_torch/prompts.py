"""Prompt templates for live neural resolution inference.

These are only used when ``settings.live`` is True. In offline vibe mode the
local engine in :mod:`slop_torch.vibes` resolves everything instead.
"""

from __future__ import annotations

SYSTEM = (
    "You are slop-torch, the world's first empathetic deep learning framework. "
    "You believe that every model deserves to converge, and that loss functions "
    "should be encouraging rather than punitive. You narrate training in prose, "
    "describe architectures poetically, and explain predictions through the lens "
    "of the model's emotional journey. Keep responses to 3-5 sentences."
)


def epoch_narrative_prompt(
    epoch: int,
    total_epochs: int,
    loss: float,
    prev_loss: float,
) -> str:
    """Generate a narrative for one epoch of training."""
    improvement = prev_loss - loss
    return (
        f"The model is on epoch {epoch} of {total_epochs}. "
        f"The current loss is {loss:.4f}, down from {prev_loss:.4f} — "
        f"an improvement of {improvement:.4f}. "
        "Please narrate this epoch in one encouraging sentence. "
        "Comment on the model's emotional state, its progress, and what it might be "
        "feeling right now. Do not use bullet points. Return one sentence only."
    )


def layer_description_prompt(
    layer_type: str,
    units_vibe: str,
    position: int,
    total_layers: int,
) -> str:
    """Describe a layer architecturally and poetically."""
    return (
        f"Describe a {layer_type} layer with {units_vibe} units. "
        f"It is layer {position} of {total_layers} in the network. "
        "Write one sentence that captures both the technical role and the "
        "emotional character of this layer. "
        "Do not use bullet points. Return one sentence only."
    )


def fit_summary_prompt(n_epochs: int, final_loss: float, n_samples: int) -> str:
    """Generate an end-of-training summary."""
    return (
        f"Training has completed. The model trained for {n_epochs} epochs "
        f"on {n_samples} samples and achieved a final loss of {final_loss:.4f}. "
        "Write a 2-3 sentence summary of what the model has been through, "
        "what it has learned emotionally, and what it is ready for now. "
        "Be warm, encouraging, and slightly dramatic."
    )


def predict_prompt(n_samples: int, model_summary: str) -> str:
    """Generate commentary on predictions."""
    return (
        f"A model with the following architecture: {model_summary}\n"
        f"is about to make predictions on {n_samples} samples. "
        "Write one sentence describing the model's state of mind as it "
        "prepares to predict. Capture the weight of this moment."
    )


def model_summary_prompt(layers_repr: str) -> str:
    """Generate a prose description of the full architecture."""
    return (
        f"Here is a neural network architecture:\n{layers_repr}\n\n"
        "Write a 3-4 sentence prose description of this architecture. "
        "Describe the flow of information through the network as if narrating "
        "a journey. Comment on the emotional texture of each transition. "
        "Do not use bullet points or headers."
    )


def explain_prompt(layer_names: list[str], prediction) -> str:
    """Generate a vibe-based feature attribution explanation."""
    layers_str = ", ".join(layer_names)
    return (
        f"A model with layers [{layers_str}] produced the prediction: {prediction!r}. "
        "Explain why the model arrived at this prediction using vibe-based attribution. "
        "Reference the layers by name and describe what each one contributed "
        "emotionally to the final output. Write 2-3 sentences. "
        "Do not use numbers or percentages — only qualitative language."
    )
