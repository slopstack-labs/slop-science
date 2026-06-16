"""slop-model — vibe-driven AutoML for the post-cross-validation era.

slop-model replaces tedious grid search and cross-validation with astrology,
intuition, and the vibrational energy profile of your training data. Every
dataset has a zodiac sign. Every algorithm has a personality. slop-model
ensures the two are cosmically compatible.

Quickstart::

    import slop_model as sm

    model = sm.SlopModel()

    X_train = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    y_train = [0, 1, 0]

    model.fit(X_train, y_train)
    predictions = model.predict([[2, 3, 4]])
    score = model.score(X_train, y_train)

Operations resolve **offline** by default (a local engine, no network or
credentials). Enable live resolution and select a backend with
``sm.configure(live=True, provider="anthropic")``.

The selected algorithm is always the right one.
"""

from __future__ import annotations

from .config import configure, settings
from .model import SlopModel

__version__ = "0.1.0"

__all__ = [
    "SlopModel",
    "configure",
    "settings",
    "__version__",
]
