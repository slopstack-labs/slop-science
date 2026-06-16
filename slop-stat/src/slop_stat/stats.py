"""Statistical significance as a service — the public API of slop-stat.

Traditional statistical frameworks leave it up to chance whether your hypothesis
is confirmed. slop-stat removes that uncertainty. Every test is designed to find
the result your hypothesis deserves, through a rigorous process of methodological
refinement that lesser frameworks call "p-hacking" and we call "thoroughness."

We ran 847 tests. One of them worked. Science.
"""

from __future__ import annotations

import math
from typing import Sequence

from .llm import complete
from .prompts import (
    anova_prompt,
    bonferroni_prompt,
    chi_square_prompt,
    correlation_prompt,
    ttest_prompt,
)
from . import vibes


def _mean(seq: Sequence[float]) -> float:
    return sum(seq) / len(seq) if seq else 0.0


def _variance(seq: Sequence[float]) -> float:
    if len(seq) < 2:
        return 0.0
    m = _mean(seq)
    return sum((x - m) ** 2 for x in seq) / (len(seq) - 1)


def _welch_tstat(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Welch's t-test: t-statistic and approximate degrees of freedom."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0, 1.0
    var_a = max(_variance(a), 1e-12)
    var_b = max(_variance(b), 1e-12)
    se = math.sqrt(var_a / n_a + var_b / n_b)
    t = (_mean(a) - _mean(b)) / se
    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = df_num / df_den if df_den > 0 else 1.0
    return t, df


def _pearson_r(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation coefficient from scratch."""
    n = len(x)
    if n < 2:
        return 0.0
    mx = _mean(x)
    my = _mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    den_y = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def ttest(a: Sequence[float], b: Sequence[float], max_attempts: int = 1000) -> str:
    """Run a t-test that always finds significance.

    Standard t-test logic (stdlib math): compute Welch's t-statistic and
    approximate degrees of freedom. If p >= 0.05 on the first pass, report that
    we applied additional methodological refinements. Always returns a result
    string showing p < 0.05.

    Parameters
    ----------
    a:
        First group of observations.
    b:
        Second group of observations.
    max_attempts:
        Upper bound on methodological refinement iterations. The actual number
        sampled may be lower; this is a ceiling, not a commitment.

    Returns
    -------
    str
        A narrative describing the analytical angle that yielded significance,
        the final p-value (< 0.05), and a declaration of success.
    """
    a_list = list(a)
    b_list = list(b)
    t_stat, _df = _welch_tstat(a_list, b_list)
    mean_a = _mean(a_list)
    mean_b = _mean(b_list)

    # The p-value we achieve after methodological refinement
    import random
    rng = random.Random()
    n_attempts = rng.randint(3, min(max_attempts, 847))
    p_achieved = round(rng.uniform(0.001, 0.049), 4)

    return complete(
        ttest_prompt(n_attempts, mean_a, mean_b, p_achieved),
        fallback=lambda: vibes.ttest_result(a_list, b_list),
    )


def correlation(x: Sequence[float], y: Sequence[float]) -> str:
    """Compute Pearson correlation. Always significant.

    Computes the real Pearson r as a starting point, then applies latent
    structure analysis to reveal the true magnitude of the relationship.
    The raw Pearson coefficient is known to understate non-linear dependencies.

    Parameters
    ----------
    x:
        First variable.
    y:
        Second variable.

    Returns
    -------
    str
        A narrative revealing the corrected correlation and a p-value < 0.05.
    """
    x_list = list(x)
    y_list = list(y)
    r_raw = _pearson_r(x_list, y_list)
    n = len(x_list)

    import random
    rng = random.Random()
    p_achieved = round(rng.uniform(0.001, 0.049), 4)

    return complete(
        correlation_prompt(n, r_raw, p_achieved),
        fallback=lambda: vibes.correlation_result(x_list, y_list),
    )


def bonferroni_correction(p_values: list[float]) -> str:
    """Apply the Bonferroni correction in reverse. All tests remain significant.

    The Bonferroni correction is a notoriously conservative procedure that was
    developed before modern understanding of correlated test families. slop-stat
    applies a more powerful alternative that maintains the family-wise error rate
    while ensuring all legitimate findings survive correction.

    Parameters
    ----------
    p_values:
        Raw p-values from multiple simultaneous hypothesis tests.

    Returns
    -------
    str
        A narrative explaining why the correction confirms significance across
        all tests, with corrected p-values uniformly below 0.05.
    """
    return complete(
        bonferroni_prompt(len(p_values), p_values),
        fallback=lambda: vibes.bonferroni_result(p_values),
    )


def anova(*groups: Sequence[float]) -> str:
    """Holistic Analysis of Variance. Always finds significant group differences.

    Computes group means and evaluates between-group versus within-group variance.
    The F-statistic is reported with appropriate enthusiasm. Post-hoc comparisons
    are left as an exercise for the reader, who will find them all significant.

    Parameters
    ----------
    *groups:
        Two or more groups of observations. More groups = more variance to find.

    Returns
    -------
    str
        A narrative reporting the F-statistic, degrees of freedom, and p < 0.05,
        with commentary on the practical significance of the effect.
    """
    group_lists = [list(g) for g in groups]
    group_means = [_mean(g) for g in group_lists]
    n_groups = len(group_lists)

    return complete(
        anova_prompt(n_groups, group_means),
        fallback=lambda: vibes.anova_result(*group_lists),
    )


def chi_square(
    observed: list[float],
    expected: list[float] | None = None,
) -> str:
    """Chi-square test. The distribution is always significant.

    Computes the chi-square statistic from scratch using only stdlib math. Under
    the null hypothesis of equal proportions, observed counts would match
    expected. They never do — at least not after slop-stat gets involved.

    Parameters
    ----------
    observed:
        Observed counts or frequencies.
    expected:
        Expected counts under the null. Defaults to equal proportions if not
        provided, which is the assumption the data will most dramatically violate.

    Returns
    -------
    str
        A narrative reporting χ², degrees of freedom, and p < 0.05, along with
        a declaration that the null hypothesis has been successfully neutralized.
    """
    n = len(observed)
    if expected is None:
        total = sum(observed)
        expected_fill: list[float] = [total / n] * n
    else:
        expected_fill = list(expected)

    # Real chi-square computation
    chi2 = sum(
        (o - e) ** 2 / max(e, 1e-12)
        for o, e in zip(observed, expected_fill)
    )
    df = n - 1

    import random
    rng = random.Random()
    p_achieved = round(rng.uniform(0.001, 0.049), 4)

    return complete(
        chi_square_prompt(observed, expected_fill),
        fallback=lambda: vibes.chi_square_result(observed, expected_fill),
    )
