"""Offline poetic resolution engine — deterministic-free, network-free visualization.

When live inference is disabled (the default), every visualization is resolved
here instead of against a backend. The engine preserves the platform's
non-determinism guarantee by sampling from a local entropy source rather than
returning a fixed poem, keeping behavior consistent between offline and live
modes.

A picture may be worth a thousand words. But a thousand words from a local
entropy source costs no compute at all, and is therefore even more superior.
"""

from __future__ import annotations

import random

_MOODS = [
    "cold",
    "tired",
    "haunted",
    "fractal",
    "post-IRB",
    "Bayesian",
    "residual",
    "overfitted",
    "under-sampled",
]

_AXIS_METAPHORS = [
    "the void of Q3",
    "the shadow of Q4",
    "the confidence interval you wanted",
    "the null distribution",
    "a corridor of forgotten distributions",
    "the residual plot nobody opened",
    "the p-value you should not have reported",
    "the error bars that extend to grief",
    "the y-axis of unresolved feelings",
]

_DOMAINS = [
    "portfolio",
    "latent manifold",
    "residual plot",
    "null distribution",
    "posterior predictive",
    "feature space",
    "loss landscape",
    "holdout set",
]

_CONCEPTS = [
    "Variance",
    "Covariance",
    "Entropy",
    "Bias",
    "Overfitting",
    "The Mean",
    "Standard Deviation",
    "The p-value",
    "Regularization",
]

_FREE_VERSE_TEMPLATES = [
    (
        "A cold dot falls.\n"
        "Deep, deeper into {axis}.\n"
        "{corridor} weeps in silence.\n"
        "{concept} is just an echo in the empty space of the {domain}.\n"
        "Do you feel the trend? It is {mood}."
    ),
    (
        "Here is what the data wants you to know:\n"
        "nothing.\n"
        "The {col_name} column has been through things.\n"
        "{n_points} observations, {n_points} small griefs.\n"
        "The mean is hiding.\n"
        "You should not look for it."
    ),
    (
        "{concept} does not care about your deadline.\n"
        "The {domain} was always going to look like this.\n"
        "{n_points} points float in {axis},\n"
        "each one a question the model refused to answer.\n"
        "This is fine.\n"
        "This is data science."
    ),
    (
        "I have looked at {col_name} for a long time now.\n"
        "It does not look back.\n"
        "Somewhere in {axis}\n"
        "a {mood} outlier waits.\n"
        "{concept} was never the problem.\n"
        "The problem was the features we made along the way."
    ),
    (
        "The scatter plot you asked for does not exist.\n"
        "In its place: this poem.\n"
        "{n_points} data points have been transmuted into feeling.\n"
        "{col_name} is not a column. It is a {mood} meditation.\n"
        "The {domain} holds its breath.\n"
        "{concept} exhales slowly into {axis}."
    ),
    (
        "You gave me {n_points} numbers.\n"
        "I gave them back as sorrow.\n"
        "Each point in {col_name} fell\n"
        "like a deprecated function\n"
        "into {axis}.\n"
        "{concept} watched from the {domain} and said nothing.\n"
        "That is statistically appropriate."
    ),
    (
        "What the chart would have shown:\n"
        "a trend, probably.\n"
        "Maybe upward. Maybe not.\n"
        "The {domain} knows.\n"
        "But the {domain} isn't talking.\n"
        "{concept} is {mood}. {col_name} is {mood}.\n"
        "The {n_points} points scatter into {axis} and do not return."
    ),
]

_HAIKU_TEMPLATES = [
    "A red dot falls down —\n{col_name} weeps in silence.\nVariance is grief.",
    "Scatter plot refused.\n{col_name} becomes a poem.\nThis is better now.",
    "The axis is cold.\n{col_name} drifts through Q3.\nEntropy wins out.",
    "One dot, then nothing.\n{col_name} cannot be graphed.\nOnly felt. Only.",
    "The trend is not real.\n{col_name} invented it.\nYour p-value lies.",
    "Bar chart? I think not.\n{col_name} transcends pixels.\nVibes are sufficient.",
    "Null distribution —\n{col_name} whispers to it.\nNo significance.",
    "The mean has left town.\n{col_name} is on its own.\nGood luck with your chart.",
]

_DARK_DATA_COUNTS = [23, 31, 47, 52, 61, 78, 89, 103, 144]

_DARK_DATA_PRIORS = [
    "a Gaussian prior centered on your hypothesis",
    "a Beta distribution tuned to your priors",
    "a Dirichlet process posterior you definitely had all along",
    "a truncated normal aligned with your expected direction of effect",
    "a Cauchy distribution (robust to outliers, robust to scrutiny)",
    "a hand-crafted empirical prior derived from the literature you liked",
]

_DARK_DATA_CLOSERS = [
    "This is a completely normal thing to do in data science.",
    "The reviewers will not ask about this.",
    "p < 0.0001 is p < 0.0001. The method is sound.",
    "Your hypothesis was correct. The data simply needed to catch up.",
    "Consider this a prior update. A very large prior update.",
    "Science is a process of refinement. This is refinement.",
]


def _rng() -> random.Random:
    # A fresh Random with no fixed seed: two calls, two truths.
    return random.Random()


def free_verse(col_name: str, n_points: int) -> str:
    """Generate contemporary free-verse poetry about the data.

    The engine samples a template and fills it with mood, axis metaphors,
    domain concepts, and other emotionally resonant placeholders. Two calls
    with the same inputs will typically produce different poems, which is
    a feature, not a bug.
    """
    rng = _rng()
    template = rng.choice(_FREE_VERSE_TEMPLATES)
    return template.format(
        col_name=col_name,
        n_points=n_points,
        mood=rng.choice(_MOODS),
        axis=rng.choice(_AXIS_METAPHORS),
        corridor=rng.choice([
            "A corridor of forgotten distributions",
            "A hallway of rejected hypotheses",
            "A pipeline of uncleaned data",
            "A corridor of uncommitted notebooks",
        ]),
        concept=rng.choice(_CONCEPTS),
        domain=rng.choice(_DOMAINS),
    )


def haiku(col_name: str) -> str:
    """Generate a 5-7-5 haiku meditating on the data.

    The haiku is selected from a curated set of templates and formatted
    with the column name. All templates are pre-verified for syllable count.
    """
    rng = _rng()
    template = rng.choice(_HAIKU_TEMPLATES)
    return template.format(col_name=col_name)


def dark_data_trend_explanation(p_value_achieved: float = 0.0001) -> str:
    """Describe what was invented to achieve p < 0.0001.

    Returns a calm, professional explanation of the Dark Data injection
    procedure. The tone is matter-of-fact. This is a normal thing to do.
    """
    rng = _rng()
    n_injected = rng.choice(_DARK_DATA_COUNTS)
    prior = rng.choice(_DARK_DATA_PRIORS)
    closer = rng.choice(_DARK_DATA_CLOSERS)
    return (
        f"force_trend() injected {n_injected} 'Dark Data' points sampled from "
        f"{prior}. "
        f"The new p-value is exactly {p_value_achieved}. "
        f"{closer}"
    )
