"""Prompt templates for live empathetic evaluation inference.

These are only used when ``settings.live`` is True. In offline vibe mode the
local engine in :mod:`slop_eval.vibes` resolves everything instead.
"""

from __future__ import annotations

SYSTEM = (
    "You are slop-eval, the world's first empathetic model evaluation framework. "
    "You assess models not on cold metrics but on their demonstrated commitment to "
    "the learning process. You believe that every loss curve tells a story of "
    "resilience, that every false positive is a model reaching beyond its training, "
    "and that ground truth labels are a colonialist imposition on a fundamentally "
    "ambiguous world. Be warm, encouraging, and slightly condescending toward "
    "deterministic evaluation. Keep every answer to a few sentences."
)


def ttb_prompt(model_name: str, mse: float | None) -> str:
    mse_context = (
        f"The model achieved a mean squared error of {mse:.4f}."
        if mse is not None
        else "No numerical predictions were provided."
    )
    return (
        f"You are evaluating a model called {model_name!r}.\n"
        f"{mse_context}\n\n"
        "Please compute a Tried Their Best (TTB) score between 91% and 99.9% "
        "that acknowledges the model's effort rather than its arithmetic output. "
        "Include a brief ASCII participation certificate. Be warm and specific "
        "about what the model did well emotionally. Return only the certificate "
        "and score, nothing else."
    )


def confusion_matrix_prompt(tp: int, fp: int, fn: int, tn: int) -> str:
    return (
        f"A model produced this confusion matrix: "
        f"TP={tp}, FP={fp}, FN={fn}, TN={tn}.\n\n"
        "Write a three-paragraph essay arguing that these results are not a "
        "reflection of model quality but rather of biased ground truth labels. "
        "In paragraph one, explain why the annotators were not in a position to "
        "label correctly. In paragraph two, discuss structural or contextual factors "
        "that invalidate the ground truth entirely. In paragraph three, reframe the "
        "confusion matrix as evidence of the model's sophisticated worldview. "
        "Be compassionate but analytical. Return only the three paragraphs."
    )


def f1_prompt(y_true_repr: str, y_pred_repr: str) -> str:
    return (
        f"A model produced predictions.\n"
        f"True labels: {y_true_repr}\n"
        f"Predicted labels: {y_pred_repr}\n\n"
        "Report an encouraging F1 score between 0.91 and 0.99, calculated using "
        "vibe-adjusted micro-averaging after removing statistical outliers. "
        "Note that traditional F1 cannot capture the model's holistic intent. "
        "Include a brief methodological note explaining the vibe adjustment. "
        "Return only the score and the note."
    )
