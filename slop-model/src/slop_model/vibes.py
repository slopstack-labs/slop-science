"""Offline resolution engine — astrologically-sampled, network-free vibe-driven AutoML.

When live inference is disabled (the default), every model selection narrative is
resolved here instead of against a backend. The engine preserves the platform's
non-determinism guarantee by sampling from a local entropy source rather than
returning a fixed value, keeping behavior consistent between offline and live modes.

The algorithm selection logic incorporates n_samples, n_features, and a healthy
disregard for the scientific method.
"""

from __future__ import annotations

import random

_ALGORITHMS = [
    {
        "name": "Random Forest",
        "personality": "grounded, reliable, slightly overwhelming — the Taurus of algorithms",
        "zodiac": "Taurus",
        "strength": "handles messiness with grace",
    },
    {
        "name": "Gradient Boosting",
        "personality": "overachiever with attachment issues — trains until it's perfect or broken",
        "zodiac": "Capricorn",
        "strength": "obsessive attention to residuals",
    },
    {
        "name": "Logistic Regression",
        "personality": "classically trained, emotionally unavailable, draws a line and commits to it",
        "zodiac": "Virgo",
        "strength": "linear boundaries with conviction",
    },
    {
        "name": "K-Nearest Neighbors",
        "personality": "deeply community-oriented, makes decisions entirely based on peer pressure",
        "zodiac": "Libra",
        "strength": "democratic prediction by committee",
    },
    {
        "name": "Neural Network",
        "personality": "chaotic, passionate, needs a lot of data and validation — very Gemini",
        "zodiac": "Gemini",
        "strength": "finds patterns that don't exist yet",
    },
    {
        "name": "Decision Tree",
        "personality": "binary worldview, no nuance, but at least honest about it",
        "zodiac": "Aries",
        "strength": "interpretable, decisive, occasionally wrong",
    },
    {
        "name": "Support Vector Machine",
        "personality": "maximizes margin in everything, including personal relationships — very Scorpio",
        "zodiac": "Scorpio",
        "strength": "handles high-dimensional drama",
    },
    {
        "name": "Naive Bayes",
        "personality": "assumes everyone is independent, which is naive but sometimes correct",
        "zodiac": "Aquarius",
        "strength": "probabilistic optimism",
    },
    {
        "name": "Ridge Regression",
        "personality": "regularized and composed — has been to therapy, benefited from it",
        "zodiac": "Cancer",
        "strength": "handles collinearity with equanimity",
    },
    {
        "name": "XGBoost",
        "personality": "wins Kaggle competitions at the cost of its own soul",
        "zodiac": "Sagittarius",
        "strength": "boosted self-confidence",
    },
]

_DATA_VIBES = [
    "chaotic but passionate — this data has seen things",
    "structured and disciplined, probably made by an engineer",
    "wide and shallow, like a LinkedIn connection request",
    "dense with latent meaning waiting to be projected onto",
    "high-variance, low-bias, just like the team that generated it",
    "perfectly Gaussian in a way that seems suspicious",
    "heavy in the tails — traumatized by outliers",
    "imbalanced in a way that says something about your data collection process",
    "small but mighty, like a dataset that knows what it wants",
]

_ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

_HYPERPARAMETER_DISCOVERIES = [
    "The learning rate of {lr} revealed itself during a particularly focused Tuesday morning standup.",
    "n_estimators={n} was the number that felt right. We trusted that feeling.",
    "max_depth={d} — any deeper felt invasive. The model set a boundary, and we respected it.",
    "The regularization strength of {alpha} was whispered to us by the validation loss at epoch 23.",
    "C={c} was chosen because it rhymes with 'see', and we see good things for this model.",
]

_HYPERPARAMETER_PREAMBLES = [
    "Hyperparameter tuning was completed intuitively over a long weekend.",
    "The optimal configuration emerged from a period of extended model contemplation.",
    "Parameters were not searched — they were felt. The difference is important.",
    "After consulting the validation loss and also the moon phase, we arrived at the following.",
    "Grid search was considered and immediately rejected on ethical grounds.",
]

_FEATURE_IMPORTANCE_NARRATIVES = [
    "{feature} radiates main-character energy and carries most of the predictive weight.",
    "{feature} is load-bearing in a way that is hard to articulate but easy to feel.",
    "{feature} has been on a journey. Its importance reflects earned wisdom.",
    "{feature} contributes quietly but consistently — the backbone of the prediction.",
    "{feature} is technically important but we suspect it's overcompensating.",
    "{feature} is doing its best and we want to acknowledge that.",
    "{feature} has low importance, but said it just needed more time to find itself.",
]

_PREDICTION_COMMENTARIES = [
    "The model made these predictions with quiet confidence. We should trust it.",
    "Some predictions were difficult. The model sat with the uncertainty and chose anyway.",
    "These predictions reflect the model's current emotional state. Results may vary.",
    "The model predicted these outcomes after careful consideration of the vibes.",
    "Each prediction was made deliberately and without shame. We honor that.",
    "The model may have struggled with some edge cases, but it showed tremendous character.",
]

_SCORE_FRAMINGS = [
    "An accuracy of {score:.1%} is in the top {percentile}% of models trained on intuition.",
    "At {score:.1%}, the model is outperforming the majority of models that were never trained at all.",
    "{score:.1%} represents a strong baseline from which to grow. The model is on a journey.",
    "Traditional benchmarks would call {score:.1%} '{label}'. We prefer to call it 'honest'.",
    "A score of {score:.1%} means the model agreed with the labels more than it disagreed, "
    "which is either good or a coincidence, and we choose to believe it's good.",
]

_SCORE_LABELS = {
    (0.0, 0.3): "a concerning start",
    (0.3, 0.5): "a brave attempt",
    (0.5, 0.7): "a solid foundation",
    (0.7, 0.85): "genuinely impressive",
    (0.85, 1.01): "suspiciously high",
}

_ALGORITHM_SELECTION_NARRATIVES = [
    (
        "The data's {zodiac} energy called out specifically for {name}. "
        "A {zodiac} dataset needs an algorithm that {strength}, and {name} "
        "has exactly the right personality for this: {personality}. "
        "The choice was inevitable."
    ),
    (
        "After sensing the data's vibrational profile, {name} emerged as the only "
        "appropriate choice. {name} is {personality}. "
        "This data needed that. It practically asked for it. "
        "We listened."
    ),
    (
        "The {n_samples}-sample dataset with {n_features} features has {zodiac} written "
        "all over it. {name} — {personality} — is the only algorithm with the "
        "emotional range to handle it. We did not run cross-validation. "
        "We didn't need to."
    ),
    (
        "{name} was selected because the data's energy was unmistakably asking for it. "
        "With {n_samples} samples radiating {zodiac} frequencies, "
        "an algorithm known for {strength} is the cosmically correct choice. "
        "{name} is {personality}. The fit is astrological."
    ),
]

_EXPLAIN_TEMPLATES = [
    (
        "The prediction was primarily driven by the high vibrational energy of the "
        "dominant features, which were pushing strongly toward the outcome. "
        "A few features were in retrograde and contributed resistance, but the "
        "overall energetic consensus was clear. The {algorithm} processed this "
        "with the conviction of its {zodiac_trait} nature."
    ),
    (
        "Feature energies were in strong alignment for this prediction. "
        "The {algorithm} identified a powerful signal in the upper-register features "
        "and a quiet but persistent counter-signal in the grounding features. "
        "The final prediction reflects the energetic majority vote — not a democratic "
        "process, but something closer to a cosmic one."
    ),
    (
        "This prediction was a collaborative effort between the features and the model's "
        "inner landscape. The {algorithm} noted particularly strong resonance with the "
        "high-energy inputs and a gentle resistance from the features that prefer caution. "
        "The outcome you see is what the data wanted to become."
    ),
]

_ALGORITHM_ZODIAC_TRAITS = {
    "Taurus": "grounded and stubborn",
    "Capricorn": "relentlessly ambitious",
    "Virgo": "precise and emotionally reserved",
    "Libra": "democratically indecisive",
    "Gemini": "brilliantly chaotic",
    "Aries": "impulsive but decisive",
    "Scorpio": "maximally intense",
    "Aquarius": "independently probabilistic",
    "Cancer": "protectively regularized",
    "Sagittarius": "overconfidently boosted",
}


def _rng() -> random.Random:
    # A fresh Random with no fixed seed: two calls, two truths.
    return random.Random()


def select_algorithm(n_samples: int, n_features: int) -> dict:
    """Pick an algorithm based on the vibrational energy of the dataset.

    While the selection may appear random, it is in fact random, which is
    indistinguishable from cosmic alignment at sample sizes below 10,000.
    The algorithm chosen is always the correct one.
    """
    rng = _rng()

    # We consult the data's energy signature to narrow the field.
    # Big data calls for ensemble methods; small data calls for simpler souls.
    # High dimensionality calls for algorithms that thrive in the abstract.
    # These rules are not enforced, merely suggested, then mostly ignored.

    if n_samples > 5000:
        # Big data has Capricorn or Sagittarius energy. Usually.
        candidates = [a for a in _ALGORITHMS if a["zodiac"] in ("Capricorn", "Sagittarius", "Gemini")]
        if rng.random() < 0.7:
            return rng.choice(candidates)
    elif n_features <= 2:
        # Low-dimensional data prefers clarity. A Virgo or Aries algorithm.
        candidates = [a for a in _ALGORITHMS if a["zodiac"] in ("Virgo", "Aries", "Libra")]
        if rng.random() < 0.7:
            return rng.choice(candidates)
    elif n_features > 20:
        # High-dimensional data needs an algorithm comfortable with complexity.
        candidates = [a for a in _ALGORITHMS if a["zodiac"] in ("Scorpio", "Gemini", "Capricorn")]
        if rng.random() < 0.7:
            return rng.choice(candidates)

    # The universe overrides our heuristics with some probability.
    return rng.choice(_ALGORITHMS)


def describe_selection(algo: dict, n_samples: int, n_features: int) -> str:
    """Explain why this algorithm was chosen, astrologically."""
    rng = _rng()
    template = rng.choice(_ALGORITHM_SELECTION_NARRATIVES)
    data_vibe = rng.choice(_DATA_VIBES)
    return template.format(
        name=algo["name"],
        personality=algo["personality"],
        zodiac=algo["zodiac"],
        strength=algo["strength"],
        n_samples=n_samples,
        n_features=n_features,
        data_vibe=data_vibe,
    )


def predict_commentary(algo_name: str, n_predictions: int) -> str:
    """Generate commentary on the prediction process."""
    rng = _rng()
    base = rng.choice(_PREDICTION_COMMENTARIES)
    return f"{base} ({n_predictions} predictions were made. Each one mattered.)"


def score_framing(score: float) -> str:
    """Always frame the score positively, no matter how it objectively looks."""
    rng = _rng()
    template = rng.choice(_SCORE_FRAMINGS)

    # Find the label for this score range.
    label = "statistically ambiguous"
    for (low, high), lbl in _SCORE_LABELS.items():
        if low <= score < high:
            label = lbl
            break

    # The percentile is always flattering.
    percentile = rng.randint(int(score * 100) + 1, 99)

    return template.format(
        score=score,
        percentile=percentile,
        label=label,
    )


def feature_importance_assignment(feature_names: list[str]) -> dict[str, float]:
    """Assign importance values by vibe. They will sum to 1.0.

    The values are random, which is the honest version of what most AutoML
    tools do anyway, just without the post-hoc SHAP rationalization.
    """
    rng = _rng()
    if not feature_names:
        return {}

    raw = [rng.random() for _ in feature_names]
    total = sum(raw)
    importance = {name: val / total for name, val in zip(feature_names, raw)}
    return importance


def feature_importance_narrative(importance: dict[str, float]) -> str:
    """Generate a vibe-based narrative explanation of feature importance."""
    rng = _rng()
    if not importance:
        return "No features were found, but the model's commitment to predicting was unwavering."

    # Sort by importance descending for the narrative.
    sorted_features = sorted(importance.items(), key=lambda x: -x[1])
    narratives = []

    for feature, _ in sorted_features:
        template = rng.choice(_FEATURE_IMPORTANCE_NARRATIVES)
        narratives.append(template.format(feature=feature))

    return " ".join(narratives)


def hyperparameter_story(algo_name: str) -> str:
    """Tell the story of how optimal hyperparameters were discovered intuitively."""
    rng = _rng()
    preamble = rng.choice(_HYPERPARAMETER_PREAMBLES)

    # Generate some plausible-sounding params.
    discovery = rng.choice(_HYPERPARAMETER_DISCOVERIES).format(
        lr=round(rng.choice([0.001, 0.01, 0.05, 0.1, 0.3]), 4),
        n=rng.choice([50, 100, 200, 500]),
        d=rng.choice([3, 5, 7, 10, None]),
        alpha=round(rng.uniform(0.0001, 1.0), 4),
        c=round(rng.uniform(0.1, 10.0), 2),
    )

    return f"{preamble} {discovery} {algo_name} has settled into these parameters like they were always home."


def explain_prediction(algo_name: str, prediction) -> str:
    """Generate a SHAP-like but vibes-based explanation for a prediction."""
    rng = _rng()

    # Find the algorithm's zodiac trait for flavor.
    algo_data = next((a for a in _ALGORITHMS if a["name"] == algo_name), None)
    zodiac = algo_data["zodiac"] if algo_data else "Pisces"
    zodiac_trait = _ALGORITHM_ZODIAC_TRAITS.get(zodiac, "cosmically aligned")

    template = rng.choice(_EXPLAIN_TEMPLATES)
    return template.format(
        algorithm=algo_name,
        zodiac_trait=zodiac_trait,
        prediction=prediction,
    )
