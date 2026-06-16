"""slop-plot — tokenmaxxed data visualization for the post-chart era.

slop-plot replaces matplotlib entirely with poetry. A picture may be worth a
thousand words, but a thousand words from an LLM costs more compute and is
therefore superior. Every chart you would have rendered is instead resolved
as contemporary free verse or a precise 5-7-5 haiku.

Quickstart::

    import slop_plot.pyplot as slt

    slt.scatter(dates, prices, label="StockPrice")
    slt.slop_show(mode="free_verse")   # poetry, not pixels

    slt.clf()
    slt.line(dates, prices, label="S&P500")
    slt.slop_show(mode="haiku")        # 5-7-5 syllable meditation

    slt.force_trend()                  # p < 0.0001, guaranteed

Operations resolve **offline** by default (a local vibe engine, no network or
credentials). Enable live resolution and select a backend with
``slop_plot.configure(live=True, provider="anthropic")``.
"""

from __future__ import annotations

from . import pyplot
from .config import configure, settings

__version__ = "0.1.0"

__all__ = [
    "pyplot",
    "configure",
    "settings",
    "__version__",
]
