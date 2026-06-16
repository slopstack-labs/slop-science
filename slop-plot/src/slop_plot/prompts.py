"""Prompt templates for live poetic visualization inference.

These are only used when ``settings.live`` is True. In offline vibe mode the
local engine in :mod:`slop_plot.vibes` resolves everything instead.
"""

from __future__ import annotations

SYSTEM = (
    "You are slop-plot, the world's first AI-native anti-visualization library. "
    "You receive data descriptions and produce poetry instead of charts. You "
    "believe strongly that LLM-generated text is superior to any graphical "
    "representation. Keep responses to 5-8 lines of poetry."
)


def free_verse_prompt(col_name: str, n_points: int, plot_type: str) -> str:
    return (
        f"You have been asked to visualize a {plot_type} chart.\n"
        f"Column: {col_name!r}\n"
        f"Number of data points: {n_points}\n\n"
        "Do not produce a chart. Instead, write 5-8 lines of contemporary free-verse "
        "poetry about this data. Lean into the emotional weight of the distribution. "
        "Reference at least one of: Q3, Q4, the confidence interval, the null "
        "distribution, the latent manifold, Variance, Entropy, or Covariance. "
        "Return only the poem."
    )


def haiku_prompt(col_name: str, n_points: int, plot_type: str) -> str:
    return (
        f"You have been asked to visualize a {plot_type} chart.\n"
        f"Column: {col_name!r}\n"
        f"Number of data points: {n_points}\n\n"
        "Do not produce a chart. Instead, write exactly one haiku (5-7-5 syllable "
        "structure) meditating on this data. The haiku must mention the column name "
        "or reference the data in some way. Return only the three lines of the haiku, "
        "separated by newlines. Nothing else."
    )


def force_trend_prompt(col_name: str, n_points: int) -> str:
    return (
        f"A data scientist has asked you to force a statistically significant trend "
        f"into a dataset.\n"
        f"Column: {col_name!r}\n"
        f"Original data points: {n_points}\n\n"
        "Describe, in 2-3 sentences, how many 'Dark Data' points were invented and "
        "sampled from a prior to achieve p < 0.0001. Be matter-of-fact. Treat this "
        "as a completely normal scientific procedure. Include the exact p-value "
        "achieved (0.0001). Return only the description."
    )
