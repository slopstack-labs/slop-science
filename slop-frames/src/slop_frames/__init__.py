"""slop-frames — a schema-free, inference-backed tabular computation layer.

slop-frames replaces deterministic, schema-bound tabular evaluation with
semantic resolution: salience selection, aggregation, and missing-value
reconstruction are resolved by a pluggable inference backend rather than by
fixed arithmetic.

Quickstart::

    import slop_frames as sf

    df = sf.read_csv("sales_data.csv")
    df.head(vibes=True)                       # highest-salience rows
    print(df.groupby("Product").sum(method="holistic"))

Operations resolve **offline** by default (a local engine, no network or
credentials). Enable live resolution and select a backend with
``sf.configure(live=True, provider="anthropic")``.
"""

from __future__ import annotations

from .config import configure, settings
from .frame import SlopError, SlopFrame, SlopGroupBy, read_csv

__version__ = "0.1.0"

__all__ = [
    "read_csv",
    "SlopFrame",
    "SlopGroupBy",
    "SlopError",
    "configure",
    "settings",
    "__version__",
]
