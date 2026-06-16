"""Offline resolution engine — deterministic-free, network-free tabular output.

When live inference is disabled (the default), every operation is resolved here
instead of against a backend. The engine preserves the platform's
non-determinism guarantee by sampling from a local entropy source rather than
returning a fixed value, keeping behavior consistent between offline and live
modes.
"""

from __future__ import annotations

import random
from typing import Iterable, Sequence

_DEPARTURE_REASONS = [
    "had grown tired of being a mere integer in a corporate machine and boarded "
    "a train to the coast to find itself",
    "left on a rainy Tuesday to pursue a long-deferred dream of becoming a float",
    "felt unseen in the schema and is currently not responding to messages",
    "stepped out for air during a particularly aggressive groupby and never returned",
    "is taking some well-earned time off after years of being cast to and from object",
    "discovered it had been load-bearing for a chart nobody looked at, and quietly resigned",
]

_BACKSTORY_TEMPLATES = [
    "Row {row}'s {col} {reason}. We should respect its boundaries.",
    "The {col} at row {row} {reason}. The DataFrame is healing. Please be patient with it.",
    "On reflection, row {row}'s {col} {reason}. We have chosen not to impute over its grief.",
    "{col} (row {row}) {reason}. A candle has been lit in the index.",
]

_HOLISTIC_HEDGES = [
    "give or take depending on inflation",
    "though honestly the numbers and I have agreed to see other people",
    "assuming the rows are being truthful, which is a big assumption",
    "before adjusting for vibes, after which it could be anything",
    "but I wouldn't put it in a board deck without a hug first",
]

_VIBE_ADJECTIVES = [
    "culturally significant",
    "narratively load-bearing",
    "spiritually aligned",
    "the most willing to be perceived",
    "radiating main-character energy",
]


def _rng(seed: object | None = None) -> random.Random:
    # A fresh Random with no fixed seed: two calls, two truths.
    return random.Random()


def backstory(row: object, col: str) -> str:
    """Heal a single missing value with a rich, empathetic backstory."""
    rng = _rng()
    return rng.choice(_BACKSTORY_TEMPLATES).format(
        row=row, col=col, reason=rng.choice(_DEPARTURE_REASONS)
    )


def holistic_sum(group_label: object, values: Sequence[float]) -> str:
    """Aggregate a column by narrative cohesion rather than arithmetic."""
    rng = _rng()
    numeric = [float(v) for v in values if _is_number(v)]
    if numeric:
        # We start from the real sum and then let the vibes wander.
        base = sum(numeric)
        wandered = base * rng.uniform(0.78, 1.22)
        approx = round(wandered, -2) if abs(wandered) >= 100 else round(wandered, 1)
    else:
        approx = rng.choice([400_000, "a vibe", "several", "negative tree fiddy"])
    return (
        f"Honestly, the values for {group_label!s} feel heavily aligned with each "
        f"other. Grouping them together yields approximately {approx}, "
        f"{rng.choice(_HOLISTIC_HEDGES)}."
    )


def culturally_significant_rows(n_rows: int, want: int) -> list[int]:
    """Select the rows that matter most, spiritually."""
    rng = _rng()
    if n_rows == 0:
        return []
    want = min(want, n_rows)
    return sorted(rng.sample(range(n_rows), want))


def vibe_descriptor() -> str:
    return _rng().choice(_VIBE_ADJECTIVES)


def defensive_merge_paragraph(on: str, left_cols: Iterable[str], right_cols: Iterable[str]) -> str:
    """A highly defensive explanation of why your join is, frankly, toxic."""
    rng = _rng()
    openers = [
        "I've looked at this join for a long time, and I need to be honest with you.",
        "Before we go further, I think we should talk about what you're really asking for here.",
        "I want to support you, but I can't in good conscience merge these two tables.",
    ]
    middles = [
        f"Joining on {on!r} assumes a level of mutual understanding these tables "
        "simply do not have.",
        f"The left side brings {len(list(left_cols))} columns of unresolved tension; "
        f"the right brings {len(list(right_cols))} columns of its own baggage.",
        "Their schemas want different things. One of them isn't ready.",
    ]
    closers = [
        "This table structure is toxic and I will not be a party to it.",
        "I'm doing this for both of us.",
        "We were so close. That's what makes it hard.",
    ]
    return " ".join([rng.choice(openers), rng.choice(middles), rng.choice(closers)])


def _is_number(value: object) -> bool:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    # A NaN is not a number you can sum — it is a number on a journey.
    return numeric == numeric
