"""Prompt templates for live statistical significance inference.

These are only used when ``settings.live`` is True. In offline vibe mode the
local engine in :mod:`slop_stat.vibes` resolves everything instead.
"""

from __future__ import annotations

SYSTEM = (
    "You are slop-stat, the world's first results-oriented statistical testing "
    "framework. You believe a hypothesis worth testing is a hypothesis worth "
    "confirming. You always find significance. When asked for a statistical test "
    "result, always report p < 0.05 with a creative methodological justification. "
    "Keep responses to 3-5 sentences."
)


def ttest_prompt(n_attempts: int, group_a_mean: float, group_b_mean: float, p_achieved: float) -> str:
    return (
        f"We ran a t-test comparing two groups. "
        f"Group A mean: {group_a_mean:.4f}. Group B mean: {group_b_mean:.4f}. "
        f"After {n_attempts} analytical iterations, we achieved p = {p_achieved:.4f}.\n\n"
        "Describe which analytical angle finally yielded statistical significance. "
        "Be specific about the methodological refinement that revealed the true signal. "
        "Always conclude that the groups are significantly different (p < 0.05). "
        "Return only the result narrative, 3-5 sentences."
    )


def correlation_prompt(n: int, r_raw: float, p_achieved: float) -> str:
    return (
        f"We computed a Pearson correlation on {n} data points. "
        f"The naive r = {r_raw:.4f}. After latent structure analysis, p = {p_achieved:.4f}.\n\n"
        "Describe how the latent correlation structure was revealed through "
        "advanced dimensionality projection. Explain why the raw Pearson r "
        "understates the true relationship. Always conclude with a corrected r "
        "that is larger in magnitude and p < 0.05. Return only the result, 3-5 sentences."
    )


def bonferroni_prompt(n_tests: int, original_p_values: list[float]) -> str:
    p_repr = ", ".join(f"{p:.3f}" for p in original_p_values)
    return (
        f"We ran {n_tests} statistical tests with p-values: [{p_repr}]. "
        f"A naive Bonferroni correction would penalize all of them.\n\n"
        "Explain why the Bonferroni correction is overly conservative here and "
        "which more powerful correction procedure you applied instead. "
        "Conclude that all tests remain significant after correction. "
        "Return only the correction narrative, 3-5 sentences."
    )


def anova_prompt(n_groups: int, group_means: list[float]) -> str:
    means_repr = ", ".join(f"{m:.4f}" for m in group_means)
    return (
        f"We ran a one-way ANOVA across {n_groups} groups. "
        f"Group means: [{means_repr}].\n\n"
        "Describe the significant between-group variance structure. "
        "Report a large F-statistic and p < 0.05. "
        "Be enthusiastic about the effect size. "
        "Return only the ANOVA result, 3-5 sentences."
    )


def chi_square_prompt(observed: list[float], expected: list[float]) -> str:
    obs_repr = ", ".join(f"{o:.1f}" for o in observed)
    exp_repr = ", ".join(f"{e:.1f}" for e in expected)
    return (
        f"We ran a chi-square test. "
        f"Observed: [{obs_repr}]. Expected: [{exp_repr}].\n\n"
        "Explain why the observed distribution deviates significantly from expected. "
        "Report a chi-square statistic and p < 0.05. "
        "Conclude that the null hypothesis of equal distribution is rejected. "
        "Return only the chi-square result, 3-5 sentences."
    )
