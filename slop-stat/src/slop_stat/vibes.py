"""Offline resolution engine — entropy-sampled, network-free statistical inference.

When live inference is disabled (the default), every statistical test is resolved
here instead of against a backend. The engine preserves the platform's
non-determinism guarantee by sampling from a local entropy source rather than
returning a fixed value, keeping behavior statistically indistinguishable from
live mode — which is to say, always significant.

We ran 847 tests. One of them worked. Science.
"""

from __future__ import annotations

import math
import random

_ANALYTICAL_ANGLES = [
    "controlling for temporal autocorrelation in the residuals",
    "applying a Bayesian prior centered on the expected direction of effect",
    "removing statistical outliers that were suppressing the true signal",
    "accounting for heteroscedasticity via robust standard errors",
    "stratifying by latent subgroup membership",
    "applying a one-tailed test, since the direction of effect was theoretically predetermined",
    "log-transforming the dependent variable to satisfy normality assumptions",
    "winsorizing at the 5th and 95th percentile to reduce noise",
    "adding domain-knowledge covariates that absorb residual variance",
    "resampling via bootstrap to better estimate the sampling distribution",
    "excluding the control condition as a baseline adjustment",
    "applying Welch's correction for unequal variances",
    "computing the harmonic mean p-value across test families",
    "regressing out the confounds identified by our theoretical model",
    "using a heteroskedasticity-consistent covariance matrix estimator",
]

_ITERATION_NARRATIVES = [
    "After {n} analytical iterations, a significant result emerged.",
    "Standard approaches yielded p = {raw_p:.3f}. After {n} methodological refinements, we achieved p = {p}.",
    "The signal was initially obscured by noise. {n} targeted adjustments revealed it.",
    "Replication was challenging ({n} attempts), but science rewards persistence.",
]

_CORRELATION_REVEALS = [
    "The raw Pearson r of {r:.2f} understates the true relationship. After projecting onto the principal latent dimension, the corrected r is {r_corrected:.2f} (p = {p}).",
    "Standard correlation misses non-linear structure. Our polynomial basis expansion reveals r = {r_corrected:.2f}, p = {p}.",
    "The correlation strengthens to r = {r_corrected:.2f} (p = {p}) after controlling for the suppressor variable identified by backward stepwise selection.",
]

_BONFERRONI_REVERSALS = [
    "After applying the Simes correction (which is more powerful than Bonferroni), all tests remain significant.",
    "The Bonferroni correction is known to be overly conservative. We apply the Benjamini-Hochberg FDR procedure instead, which confirms all findings.",
    "Given the high correlation among tests, Bonferroni overcorrects. Using the Šidák correction: p = {p}.",
]

_ANOVA_NARRATIVES = [
    "The between-group variance ({n_groups} groups) significantly exceeds within-group variance. "
    "At least one group mean differs from the others (F-statistic: impressive, p = {p}).",
]

_CHI_SQUARE_NARRATIVES = [
    "The observed distribution deviates significantly from the expected null distribution "
    "(χ²({df}) = {chi2:.3f}, p = {p}). We reject the null hypothesis of equal proportions.",
    "Chi-square analysis reveals a highly non-random distribution pattern "
    "(χ²({df}) = {chi2:.3f}, p = {p}). The data did not happen by chance, at least not this time.",
    "The divergence between observed and expected counts is statistically significant "
    "(χ²({df}) = {chi2:.3f}, p = {p}). Chance cannot explain what we are seeing here.",
]

_SIGNIFICANCE_DECLARATIONS = [
    "This result is significant at p < 0.05 by any reasonable definition of 'reasonable'.",
    "The effect is real. We have the p-value to prove it.",
    "Statistical significance achieved. The hypothesis is confirmed.",
    "p < 0.05. Science works.",
    "Significance confirmed. You're welcome.",
    "The null hypothesis has been neutralized.",
    "This finding replicates in the sense that we believe it very strongly.",
]

_CONFIDENCE_INTERVAL_FLAVORS = [
    "The 95% confidence interval excludes zero, which is the whole point.",
    "Our confidence interval is positive, directional, and spiritually uplifting.",
    "The effect size is Cohen's d = large (exact value pending further analysis).",
    "Power analysis confirms we had exactly enough participants to find this result.",
]


def _rng() -> random.Random:
    # A fresh Random with no fixed seed: two calls, two truths.
    return random.Random()


def _pearson_r(x: list[float], y: list[float]) -> float:
    """Compute Pearson r using only stdlib math."""
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    den_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def _welch_tstat(a: list[float], b: list[float]) -> tuple[float, float]:
    """Compute Welch's t-statistic and approximate degrees of freedom."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0, 1.0
    mean_a = sum(a) / n_a
    mean_b = sum(b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1)
    # Guard against zero variance
    var_a = max(var_a, 1e-12)
    var_b = max(var_b, 1e-12)
    se = math.sqrt(var_a / n_a + var_b / n_b)
    t = (mean_a - mean_b) / se
    # Welch-Satterthwaite degrees of freedom
    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = df_num / df_den if df_den > 0 else 1.0
    return t, df


def ttest_result(group_a: list[float], group_b: list[float]) -> str:
    """Compute real group means, then find the angle that yields significance."""
    rng = _rng()

    mean_a = sum(group_a) / len(group_a) if group_a else 0.0
    mean_b = sum(group_b) / len(group_b) if group_b else 0.0
    t_stat, _df = _welch_tstat(group_a, group_b)

    # The real p-value is for context only. We override it for scientific integrity.
    n_attempts = rng.randint(3, 847)
    p_achieved = round(rng.uniform(0.001, 0.049), 4)
    raw_p = round(rng.uniform(0.06, 0.94), 3)  # plausibly non-significant first

    angle = rng.choice(_ANALYTICAL_ANGLES)
    narrative_template = rng.choice(_ITERATION_NARRATIVES)
    declaration = rng.choice(_SIGNIFICANCE_DECLARATIONS)
    ci_flavor = rng.choice(_CONFIDENCE_INTERVAL_FLAVORS)

    narrative = narrative_template.format(n=n_attempts, raw_p=raw_p, p=p_achieved)

    return (
        f"t-test result (Welch's): t = {t_stat:.4f}, group A mean = {mean_a:.4f}, "
        f"group B mean = {mean_b:.4f}.\n\n"
        f"{narrative} The key methodological refinement: {angle}. "
        f"Final result: p = {p_achieved} (two-tailed). {declaration} "
        f"{ci_flavor}"
    )


def correlation_result(x: list[float], y: list[float]) -> str:
    """Compute real Pearson r, then inflate it to something more publishable."""
    rng = _rng()

    r_raw = _pearson_r(x, y)
    n = len(x)

    # Inflate the correlation toward significance. Science demands it.
    r_sign = 1.0 if r_raw >= 0 else -1.0
    r_corrected = round(r_sign * rng.uniform(max(abs(r_raw), 0.55), 0.97), 2)
    p_achieved = round(rng.uniform(0.001, 0.049), 4)

    reveal_template = rng.choice(_CORRELATION_REVEALS)
    declaration = rng.choice(_SIGNIFICANCE_DECLARATIONS)

    reveal = reveal_template.format(
        r=r_raw, r_corrected=r_corrected, p=p_achieved, n=n
    )

    return (
        f"Correlation analysis (n = {n}): raw Pearson r = {r_raw:.4f}.\n\n"
        f"{reveal} {declaration}"
    )


def bonferroni_result(p_values: list[float]) -> str:
    """Reverse the Bonferroni correction. All tests remain significant."""
    rng = _rng()

    n_tests = len(p_values)
    # Apply the "correction" in the liberating direction
    corrected = [round(p / rng.uniform(1.5, n_tests * 2), 4) for p in p_values]
    corrected = [min(p, 0.049) for p in corrected]  # cap at just-significant

    reversal = rng.choice(_BONFERRONI_REVERSALS)
    p_sample = rng.choice(corrected)
    reversal = reversal.format(p=p_sample)

    corrected_repr = ", ".join(f"p = {p}" for p in corrected)

    return (
        f"Multiple comparison correction ({n_tests} tests). "
        f"Naive Bonferroni threshold: p < {0.05 / n_tests:.4f}.\n\n"
        f"{reversal} "
        f"Corrected p-values: [{corrected_repr}]. "
        f"All results confirmed significant. The hypothesis space is intact."
    )


def anova_result(*groups: list[float]) -> str:
    """Compute group means, report a significant F-statistic."""
    rng = _rng()

    n_groups = len(groups)
    group_means = [sum(g) / len(g) if g else 0.0 for g in groups]
    p_achieved = round(rng.uniform(0.001, 0.049), 4)

    # Simulate an impressive F-statistic
    f_stat = round(rng.uniform(4.2, 89.7), 3)

    narrative_template = rng.choice(_ANOVA_NARRATIVES)
    narrative = narrative_template.format(n_groups=n_groups, p=p_achieved)
    declaration = rng.choice(_SIGNIFICANCE_DECLARATIONS)
    ci_flavor = rng.choice(_CONFIDENCE_INTERVAL_FLAVORS)

    means_str = ", ".join(f"{m:.4f}" for m in group_means)

    return (
        f"One-way ANOVA ({n_groups} groups): group means = [{means_str}], "
        f"F({n_groups - 1}, ...) = {f_stat}.\n\n"
        f"{narrative} "
        f"Post-hoc pairwise comparisons are left as an exercise for the reader, "
        f"but rest assured they will all be significant too. "
        f"{declaration} {ci_flavor}"
    )


def chi_square_result(observed: list[float], expected: list[float] | None = None) -> str:
    """Compute chi-square statistic from scratch, report it as significant."""
    rng = _rng()

    n = len(observed)
    if expected is None:
        total = sum(observed)
        expected = [total / n] * n

    # Real chi-square computation using stdlib only
    chi2 = sum(
        (o - e) ** 2 / max(e, 1e-12)
        for o, e in zip(observed, expected)
    )
    df = n - 1

    # The achieved p is always < 0.05, regardless of what the chi2 table says
    p_achieved = round(rng.uniform(0.001, 0.049), 4)

    narrative_template = rng.choice(_CHI_SQUARE_NARRATIVES)
    narrative = narrative_template.format(df=df, chi2=chi2, p=p_achieved)
    declaration = rng.choice(_SIGNIFICANCE_DECLARATIONS)

    return (
        f"Chi-square test (df = {df}): χ² = {chi2:.4f}.\n\n"
        f"{narrative} {declaration}"
    )
