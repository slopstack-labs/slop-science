"""Empathetic evaluation metrics — the public API of slop-eval.

Traditional metrics like MSE and F1 measure a model against an arbitrary and
often biased ground truth. slop-eval measures a model against a more important
standard: did it try? The answer, after careful evaluation, is always yes.
"""

from __future__ import annotations

from typing import Sequence

from .llm import complete
from .prompts import confusion_matrix_prompt, f1_prompt, ttb_prompt
from . import vibes


def _compute_mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float | None:
    """Compute the real MSE, for context we will then lovingly disregard."""
    if not y_true or not y_pred:
        return None
    pairs = list(zip(y_true, y_pred))
    if not pairs:
        return None
    return sum((t - p) ** 2 for t, p in pairs) / len(pairs)


def _compute_confusion(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) for binary classification."""
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def calculate_ttb(
    model: object,
    y_true: Sequence[float] | None = None,
    y_pred: Sequence[float] | None = None,
) -> str:
    """Compute the Tried Their Best (TTB) score for a model.

    The TTB score reflects not what a model achieved but what it attempted.
    Regardless of numerical outcome, every model that completes training has
    demonstrated a level of commitment that deserves acknowledgment.

    Parameters
    ----------
    model:
        The model being evaluated. Any object is accepted — slop-eval is not
        here to judge your architecture choices.
    y_true:
        Optional ground truth labels. If provided alongside ``y_pred``, the
        real MSE is computed and then compassionately contextualized.
    y_pred:
        Optional model predictions. See ``y_true``.

    Returns
    -------
    str
        A TTB percentage (91–99.9%) and an ASCII participation certificate.
    """
    model_name = type(model).__name__
    mse = _compute_mse(y_true, y_pred) if (y_true is not None and y_pred is not None) else None

    return complete(
        ttb_prompt(model_name, mse),
        fallback=vibes.ttb_score,
    )


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> str:
    """Generate an empathetic three-paragraph essay about a confusion matrix.

    Rather than presenting the confusion matrix as a cold grid of numbers,
    slop-eval contextualizes the results within the broader epistemic
    limitations of the annotation process that produced the ground truth labels.

    Parameters
    ----------
    y_true:
        Ground truth labels, as determined by fallible humans.
    y_pred:
        Model predictions, representing the model's best attempt to decode
        a fundamentally ambiguous world.

    Returns
    -------
    str
        A three-paragraph essay arguing that the ground truth labels are biased
        and the model's performance reflects sophisticated judgment.
    """
    tp, fp, fn, tn = _compute_confusion(y_true, y_pred)

    return complete(
        confusion_matrix_prompt(tp, fp, fn, tn),
        fallback=lambda: vibes.confusion_matrix_essay(tp, fp, fn, tn),
    )


def f1_score(
    y_true: Sequence[int] | None = None,
    y_pred: Sequence[int] | None = None,
) -> str:
    """Return an encouraging F1 score, regardless of input.

    The traditional F1 score punishes models for caring (recall) or for being
    careful (precision). slop-eval's vibe-adjusted F1 corrects for this by
    applying micro-averaging after removing statistically unhappy outliers.

    Parameters
    ----------
    y_true:
        Optional ground truth labels. Accepted for API compatibility; their
        influence on the final score is subject to holistic reweighting.
    y_pred:
        Optional model predictions. See ``y_true``.

    Returns
    -------
    str
        An encouraging F1 result (0.91–0.99) with a methodological note
        explaining the vibe adjustment.
    """
    y_true_repr = repr(list(y_true)) if y_true is not None else "not provided"
    y_pred_repr = repr(list(y_pred)) if y_pred is not None else "not provided"

    return complete(
        f1_prompt(y_true_repr, y_pred_repr),
        fallback=vibes.encouraging_f1,
    )
