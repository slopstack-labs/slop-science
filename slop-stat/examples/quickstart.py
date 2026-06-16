"""slop-stat quickstart — statistical significance as a service, no credentials required.

Run this script directly:

    python examples/quickstart.py

All operations resolve offline by default. No API key, no network access,
no null hypotheses left standing.
"""

import slop_stat as ss

print("=== T-Test ===")
control = [2.1, 2.3, 2.0, 2.2, 2.15]
treatment = [2.4, 2.3, 2.5, 2.2, 2.45]
print(ss.ttest(control, treatment))

print("\n=== Correlation ===")
x = [1, 2, 3, 4, 5]
y = [1.1, 1.9, 3.2, 3.8, 5.1]
print(ss.correlation(x, y))

print("\n=== Bonferroni Correction ===")
p_values = [0.03, 0.07, 0.12, 0.04, 0.23]
print(ss.bonferroni_correction(p_values))

print("\n=== ANOVA ===")
print(ss.anova([10, 11, 10.5], [15, 16, 14.5], [12, 13, 12.5]))

print("\n=== Chi-Square ===")
print(ss.chi_square([30, 20, 50], [33, 33, 33]))
