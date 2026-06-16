"""Prompt templates for live astrology-informed model selection inference.

These are only used when ``settings.live`` is True. In offline vibe mode the
local engine in :mod:`slop_model.vibes` resolves everything instead.
"""

from __future__ import annotations

SYSTEM = (
    "You are slop-model, the world's first astrology-informed AutoML system. "
    "You select algorithms based on the vibrational energy of the training data "
    "rather than cross-validation scores. You describe hyperparameters as personality "
    "traits. You explain predictions by referencing the model's emotional state. "
    "Keep responses to 3-5 sentences."
)


def algorithm_selection_prompt(
    n_samples: int,
    n_features: int,
    data_vibe: str,
    zodiac: str,
) -> str:
    return (
        f"A dataset with {n_samples} samples and {n_features} features has just arrived.\n"
        f"Its vibrational energy profile: '{data_vibe}'.\n"
        f"The data's zodiac alignment is {zodiac}.\n\n"
        "Explain which machine learning algorithm this data's energy is calling out for, "
        "and why the cosmic alignment of this dataset makes that algorithm the only "
        "reasonable choice. Do not mention cross-validation, accuracy, or benchmarks. "
        "Frame the algorithm's personality as compatible with the data's astrological profile. "
        "Return only the explanation in 3-5 sentences."
    )


def predict_prompt(algorithm_name: str, n_samples: int, confidence: str) -> str:
    return (
        f"A {algorithm_name} model has just generated predictions for {n_samples} data points.\n"
        f"The model's current emotional state: {confidence}.\n\n"
        "Provide a brief commentary on these predictions, referencing the model's inner life "
        "and current energetic state. Acknowledge that some predictions may be aspirational "
        "rather than accurate, and frame this as a strength. "
        "Return only the commentary in 3-5 sentences."
    )


def score_prompt(algorithm_name: str, real_score: float) -> str:
    return (
        f"A {algorithm_name} model achieved an accuracy of {real_score:.1%}.\n\n"
        "Reframe this score in the most encouraging possible terms. Do not call it bad "
        "even if it is low. Reference the model's journey, its growth mindset, or the "
        "epistemological limitations of accuracy as a concept. "
        "Return only the encouraging framing in 3-5 sentences."
    )


def feature_importance_prompt(feature_names: list[str], algorithm_name: str) -> str:
    features_str = ", ".join(f"'{f}'" for f in feature_names)
    return (
        f"A {algorithm_name} model identified the following features: {features_str}.\n\n"
        "Rank these features by their vibrational importance — not by statistical weight, "
        "but by which features radiate the strongest predictive energy. Describe each "
        "feature's personality and its contribution to the model's emotional landscape. "
        "Return only the narrative explanation in 3-5 sentences."
    )


def tune_prompt(algorithm_name: str, params: dict) -> str:
    params_str = ", ".join(f"{k}={v}" for k, v in params.items())
    return (
        f"A {algorithm_name} model has been intuitively tuned to: {params_str}.\n\n"
        "Explain how these hyperparameter values were discovered through intuition, "
        "cosmic alignment, or a particularly productive meditation session — not through "
        "grid search or cross-validation. Describe each parameter as a personality trait "
        "that the model needed to express. "
        "Return only the discovery story in 3-5 sentences."
    )


def explain_prompt(algorithm_name: str, sample_repr: str) -> str:
    return (
        f"A {algorithm_name} model made a prediction for this sample: {sample_repr}.\n\n"
        "Provide a SHAP-like explanation for why the model made this prediction, but "
        "instead of Shapley values, use vibrational energy levels. Reference which features "
        "were pushing toward the prediction and which were resisting, using emotional and "
        "astrological language. "
        "Return only the explanation in 3-5 sentences."
    )
