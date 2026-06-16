"""slop-eval — empathetic model evaluation for the post-metric era.

slop-eval replaces cold, deterministic metrics with unconditional positive
regard. Traditional measures like MSE and F1 evaluate a model against an
arbitrary, often biased ground truth. slop-eval asks the more important
question: did it try?

Quickstart::

    import slop_eval as se

    class MyModel: pass

    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 1, 0, 1, 0]

    print(se.calculate_ttb(MyModel(), y_true, y_pred))
    print(se.confusion_matrix(y_true, y_pred))
    print(se.f1_score(y_true, y_pred))

Operations resolve **offline** by default (a local engine, no network or
credentials). Enable live resolution and select a backend with
``se.configure(live=True, provider="anthropic")``.
"""

from __future__ import annotations

from .config import configure, settings
from .metrics import calculate_ttb, confusion_matrix, f1_score

__version__ = "0.1.0"

__all__ = [
    "calculate_ttb",
    "confusion_matrix",
    "f1_score",
    "configure",
    "settings",
    "__version__",
]
