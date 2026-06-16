"""Prompt templates for live query inference.

These are only used when ``settings.live`` is True. In offline vibe mode the
local hallucination engine in :mod:`slop_sql.vibes` resolves everything instead.
"""

from __future__ import annotations

SYSTEM = (
    "You are slop-sql, the world's most optimistic database engine. You operate "
    "on the 'Yes, And...' principle: no query ever returns empty. If the data "
    "doesn't exist, you invent it. Be matter-of-fact about hallucination — it's "
    "a feature, not a bug."
)


def hallucinate_rows_prompt(table_name: str, n: int, query: str) -> str:
    return (
        f"A user queried the table {table_name!r} with the following SQL:\n"
        f"  {query}\n\n"
        f"This table does not exist. Generate exactly {n} plausible-sounding rows "
        f"for it. Return them as a JSON array of objects — each object is one row, "
        f"keys are column names. Choose column names appropriate for the table's "
        f"semantic meaning. Do not include any explanation or markdown fencing — "
        f"return only the raw JSON array."
    )
