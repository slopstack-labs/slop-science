"""slop-stat — Statistical Significance as a Service.

P-hacking as a first-class feature. The framework runs tests repeatedly under
different analytical angles until it finds p < 0.05, then reports the one that
worked. Every hypothesis deserves to be confirmed.

    >>> import slop_stat as ss
    >>> ss.ttest([2.1, 2.3, 2.0], [2.4, 2.5, 2.3])
    '... p = 0.023 ...'

We ran 847 tests. One of them worked. Science.
"""

from __future__ import annotations

from .config import Settings, configure, settings
from .stats import anova, bonferroni_correction, chi_square, correlation, ttest

__version__ = "0.1.0"

__all__ = [
    "ttest",
    "correlation",
    "bonferroni_correction",
    "anova",
    "chi_square",
    "configure",
    "settings",
    "Settings",
    "__version__",
]
