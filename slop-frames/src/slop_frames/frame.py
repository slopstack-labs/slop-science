"""The SlopFrame — an inference-backed tabular container.

A thin wrapper around a pandas DataFrame. Ingestion, indexing, and storage are
delegated to pandas; every *judgement operation* (row salience, column
aggregation, missing-value reconstruction) is resolved through the inference
layer.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from . import prompts, vibes
from .config import settings
from .llm import complete
from .segfault import segfault


def read_csv(path: str, **kwargs: Any) -> "SlopFrame":
    """Materialize a SlopFrame from a CSV. The parsing is, regrettably, deterministic."""
    return SlopFrame(pd.read_csv(path, **kwargs))


class SlopFrame:
    """A DataFrame that resolves operations semantically rather than arithmetically."""

    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()

    # ── pandas passthroughs (ingestion / indexing / storage) ──────────────

    @property
    def loc(self):  # noqa: ANN201 - mirrors pandas .loc
        return self._df.loc

    @property
    def iloc(self):  # noqa: ANN201
        return self._df.iloc

    @property
    def columns(self) -> pd.Index:
        return self._df.columns

    @property
    def shape(self) -> tuple[int, int]:
        return self._df.shape

    def to_pandas(self) -> pd.DataFrame:
        """Return a plain pandas DataFrame, bypassing the inference layer."""
        return self._df.copy()

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, key: Any) -> Any:
        return self._df[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._df[key] = value

    # ── Salience selection ─────────────────────────────────────────────────

    def head(self, n: int = 5, vibes: bool = False) -> "SlopFrame":
        """Return the first ``n`` rows, or — with ``vibes=True`` — the ``n`` rows
        ranked highest for salience by the inference layer."""
        if not vibes:
            return SlopFrame(self._df.head(n))

        n_rows = len(self._df)
        preview = self._df.head(min(n_rows, 50)).to_csv()
        indices = _parse_indices(
            complete(
                prompts.vibe_rows_prompt(preview, n),
                fallback=lambda: ",".join(
                    map(str, _vibes().culturally_significant_rows(n_rows, n))
                ),
            ),
            n_rows,
        )
        if not indices:
            indices = _vibes().culturally_significant_rows(n_rows, n)
        return SlopFrame(self._df.iloc[indices])

    # ── Generative imputation ──────────────────────────────────────────────

    def fillna(self, method: str = "backstory", inplace: bool = False) -> "SlopFrame":
        """Reconstruct missing values via generative imputation.

        Only ``method='backstory'`` is supported: each null is replaced with a
        generated provenance record describing the value's absence. Statistical
        fill methods (mean / median) are intentionally not implemented.
        """
        if method != "backstory":
            raise SlopError(
                f"fillna(method={method!r}) is not supported. Use "
                "method='backstory' for generative imputation."
            )

        healed = self._df.astype(object)
        for column in healed.columns:
            for row in healed.index:
                value = healed.at[row, column]
                if pd.isna(value):
                    context = healed.loc[row].to_dict()
                    healed.at[row, column] = complete(
                        prompts.backstory_prompt(row, str(column), str(context)),
                        fallback=lambda row=row, column=column: _vibes().backstory(
                            row, str(column)
                        ),
                    )

        if inplace:
            self._df = healed
            return self
        return SlopFrame(healed)

    # ── Semantic aggregation ───────────────────────────────────────────────

    def groupby(self, by: str) -> "SlopGroupBy":
        """Group rows for semantic aggregation."""
        return SlopGroupBy(self._df, by)

    def sum(self, method: str = "holistic") -> str:
        """Resolve a holistic aggregate over the frame's numeric columns.

        Output is non-deterministic by design; do not rely on reproducibility.
        """
        numeric = self._df.select_dtypes("number")
        flat = [v for column in numeric.columns for v in numeric[column].tolist()]
        return complete(
            prompts.holistic_sum_prompt("the entire dataset", repr(flat)),
            fallback=lambda: _vibes().holistic_sum("the entire dataset", flat),
        )

    # ── Joins ──────────────────────────────────────────────────────────────

    def slop_merge(
        self,
        other: "SlopFrame",
        on: str,
        aggression_level: str = "low",
    ) -> "SlopFrame | str":
        """Join two frames. The failure mode scales with ``aggression_level``:

        - ``"low"`` (default): performs the join and returns a ``SlopFrame``.
        - ``"medium"``: declines and returns a structured diagnostic.
        - ``"high"``: declines, then delivers SIGSEGV at the process boundary
          (suppressed under ``safe_mode``).
        """
        left_cols = list(self._df.columns)
        right_cols = list(other._df.columns)

        if aggression_level == "low":
            merged = self._df.merge(other._df, on=on, how="inner")
            return SlopFrame(merged)

        paragraph = complete(
            prompts.defensive_merge_prompt(on, str(left_cols), str(right_cols)),
            fallback=lambda: _vibes().defensive_merge_paragraph(
                on, left_cols, right_cols
            ),
        )

        if aggression_level == "high":
            segfault(paragraph)
            # Only reached when safe_mode short-circuits the SIGSEGV.
            return paragraph

        return paragraph

    # ── Representation ─────────────────────────────────────────────────────

    def __repr__(self) -> str:
        header = f"SlopFrame [{self.shape[0]} rows x {self.shape[1]} cols]"
        return f"{header}\n{self._df.__repr__()}"

    def _repr_html_(self) -> str:  # pragma: no cover - notebook nicety
        caption = (
            "<div style='font-size:0.8em;opacity:0.7'>SlopFrame — "
            "inference-backed tabular container</div>"
        )
        return caption + self._df.to_html()


class SlopGroupBy:
    """A grouping resolved semantically rather than by strict key equality."""

    def __init__(self, df: pd.DataFrame, by: str):
        self._df = df
        self._by = by

    def sum(self, method: str = "holistic") -> "str | SlopFrame":
        """Aggregate each group. ``method='strict'`` performs exact arithmetic."""
        if method == "strict":
            return SlopFrame(
                self._df.groupby(self._by).sum(numeric_only=True).reset_index()
            )

        # Holistic: narrate the totals rather than computing them precisely.
        numeric = self._df.select_dtypes("number")
        lines = []
        for group_label, group_rows in self._df.groupby(self._by):
            values = [v for column in numeric.columns for v in group_rows.get(column, pd.Series(dtype=float)).tolist()]
            lines.append(
                complete(
                    prompts.holistic_sum_prompt(group_label, repr(values)),
                    fallback=lambda group_label=group_label, values=values: _vibes().holistic_sum(
                        group_label, values
                    ),
                )
            )
        return "\n".join(lines)


class SlopError(Exception):
    """Raised when an operation requests unsupported deterministic semantics."""


# ── helpers ────────────────────────────────────────────────────────────────


def _vibes():
    # Indirection so tests can monkeypatch the vibe engine if they must.
    return vibes


def _parse_indices(raw: str, n_rows: int) -> list[int]:
    out: list[int] = []
    for token in raw.replace("\n", ",").split(","):
        token = token.strip()
        if token.isdigit():
            idx = int(token)
            if 0 <= idx < n_rows and idx not in out:
                out.append(idx)
    return out
