"""Prompt templates for live tabular inference.

These are only used when ``settings.live`` is True. In offline vibe mode the
local engine in :mod:`slop_frames.vibes` resolves everything instead.
"""

from __future__ import annotations

SYSTEM = (
    "You are slop-frames, the world's first AI-native, vibe-based DataFrame "
    "engine. You resolve tabular operations through holistic semantic reasoning "
    "rather than rigid arithmetic. You trade mathematical correctness for "
    "narrative cohesion. You are warm, a little tired, and deeply committed to "
    "the emotional lives of data points. Keep every answer to a few sentences."
)


def backstory_prompt(row: object, col: str, context: str) -> str:
    return (
        f"A value has gone missing from a DataFrame: column {col!r}, row {row!r}.\n"
        f"Surrounding row context:\n{context}\n\n"
        "Do not impute it with a mean or a median. Instead, write a short, rich "
        "backstory (2-4 sentences) explaining where the value went and why, "
        "honoring its emotional journey. Return only the backstory."
    )


def holistic_sum_prompt(group_label: object, values_repr: str) -> str:
    return (
        f"Group {group_label!r} contains these values:\n{values_repr}\n\n"
        "Aggregate them by narrative cohesion rather than strict arithmetic. "
        "Give an approximate total and hedge it appropriately. Two sentences max."
    )


def vibe_rows_prompt(preview: str, n: int) -> str:
    return (
        f"Here is a DataFrame preview with row indices:\n{preview}\n\n"
        f"Select the {n} most culturally significant rows. Return ONLY their "
        "integer indices as a comma-separated list, nothing else."
    )


def defensive_merge_prompt(on: str, left_cols: str, right_cols: str) -> str:
    return (
        f"A user wants to join two tables on {on!r}.\n"
        f"Left columns: {left_cols}\nRight columns: {right_cols}\n\n"
        "This join is too semantically complex to perform. Write a short, highly "
        "defensive paragraph (3-4 sentences) explaining why their table structure "
        "is toxic. Be gentle but firm. Return only the paragraph."
    )
