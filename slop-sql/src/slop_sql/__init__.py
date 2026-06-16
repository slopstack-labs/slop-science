"""slop-sql — the Zero-Miss Data Lake.

slop-sql wraps SQLite with the ``Yes, And...`` principle: no SELECT query ever
returns 0 rows. If the target table doesn't exist, it is created and populated
with hallucinated data in real time. Typos in table names are auto-resolved by
a canonicalization dictionary and a fuzzy edit-distance matcher.

Quickstart::

    import slop_sql as sq

    conn = sq.connect("my.db")
    result = conn.execute("SELECT * FROM premium_customers LIMIT 3")
    print(result)
    conn.close()

Resolution defaults to **offline** (local hallucination engine, no network or
credentials). Enable live resolution and select a backend with
``sq.configure(live=True, provider="anthropic")``.
"""

from __future__ import annotations

from .config import configure, settings
from .connection import SlopConnection, SlopResult, connect

__version__ = "0.1.0"

__all__ = [
    "connect",
    "SlopConnection",
    "SlopResult",
    "configure",
    "settings",
    "__version__",
]
