"""slop-plot's stateful plotting API — imported as ``import slop_plot.pyplot as slt``.

This module mirrors the matplotlib.pyplot stateful interface: you accumulate
plot calls against an implicit figure, then call :func:`slop_show` to resolve
the current figure as poetry. No pixels are produced at any point.

Why render pixels when you can stream vibes?

Usage::

    import slop_plot.pyplot as slt

    slt.scatter(dates, prices, label="StockPrice")
    slt.slop_show(mode="free_verse")   # prints poetry, not a chart

    slt.clf()                          # clear the implicit figure

    slt.line(dates, prices, label="S&P500")
    slt.slop_show(mode="haiku")        # 5-7-5 syllable meditation

    slt.force_trend()                  # inject Dark Data for p < 0.0001
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from . import vibes as _vibes
from .llm import complete as _complete
from .prompts import force_trend_prompt, free_verse_prompt, haiku_prompt


# ---------------------------------------------------------------------------
# Internal figure state
# ---------------------------------------------------------------------------

@dataclass
class _PlotEntry:
    """A single recorded data series within the implicit figure."""
    plot_type: str        # "scatter", "line", or "bar"
    label: str
    x: list              # x values or categories
    y: list              # y values


@dataclass
class _Figure:
    """The implicit figure — accumulates data series until slop_show() is called."""
    entries: list[_PlotEntry] = field(default_factory=list)

    def clear(self) -> None:
        self.entries.clear()

    @property
    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def primary_label(self) -> str:
        """Return the label of the first recorded series, or a default."""
        if self.entries:
            return self.entries[0].label or "data"
        return "data"

    def primary_plot_type(self) -> str:
        if self.entries:
            return self.entries[0].plot_type
        return "scatter"

    def total_points(self) -> int:
        """Return the total number of data points across all series."""
        return sum(len(e.y) for e in self.entries)


# The module-level implicit figure, analogous to matplotlib's gcf().
_fig = _Figure()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scatter(x: Sequence, y: Sequence, label: str = "") -> None:
    """Record a scatter series for poetic resolution.

    Parameters
    ----------
    x:
        Sequence of x-axis values. Typically temporal or categorical.
    y:
        Sequence of y-axis values. The emotional weight of these numbers
        will be transmuted into verse.
    label:
        A name for this series. Used as the column name in poem generation.
    """
    _fig.entries.append(_PlotEntry(
        plot_type="scatter",
        label=label,
        x=list(x),
        y=list(y),
    ))


def line(x: Sequence, y: Sequence, label: str = "") -> None:
    """Record a line series for poetic resolution.

    Parameters
    ----------
    x:
        Sequence of x-axis values.
    y:
        Sequence of y-axis values.
    label:
        A name for this series.
    """
    _fig.entries.append(_PlotEntry(
        plot_type="line",
        label=label,
        x=list(x),
        y=list(y),
    ))


def bar(categories: Sequence, values: Sequence, label: str = "") -> None:
    """Record a bar series for poetic resolution.

    Parameters
    ----------
    categories:
        Sequence of category labels (the x-axis of a bar chart, metaphorically).
    values:
        Sequence of bar heights. Each bar is a small grief.
    label:
        A name for this series.
    """
    _fig.entries.append(_PlotEntry(
        plot_type="bar",
        label=label,
        x=list(categories),
        y=list(values),
    ))


def slop_show(mode: Literal["free_verse", "haiku"] = "free_verse") -> None:
    """Resolve the current figure as poetry and print it.

    No chart is produced. No pixels are harmed. The implicit figure is not
    cleared after resolution — call :func:`clf` explicitly.

    Parameters
    ----------
    mode:
        ``"free_verse"`` (default) — multi-line contemporary poetry that
        explores the emotional dimensions of your data.

        ``"haiku"`` — a precise 5-7-5 syllable meditation on the dataset.
        Brief. Devastating. Statistically non-significant.
    """
    col_name = _fig.primary_label()
    n_points = _fig.total_points() if not _fig.is_empty else 0
    plot_type = _fig.primary_plot_type()

    if mode == "haiku":
        poem = _complete(
            haiku_prompt(col_name, n_points, plot_type),
            fallback=lambda: _vibes.haiku(col_name),
        )
    else:
        poem = _complete(
            free_verse_prompt(col_name, n_points, plot_type),
            fallback=lambda: _vibes.free_verse(col_name, n_points),
        )

    print(poem)


def force_trend() -> str:
    """Inject Dark Data points to achieve p < 0.0001 and describe the procedure.

    This function adds statistically convenient synthetic observations to the
    current figure's primary dataset, ensuring any trend you suspected is now
    formally significant. The procedure is described in plain, professional
    language and printed to stdout.

    Returns
    -------
    str
        A description of the Dark Data injection, suitable for inclusion in a
        methods section (not recommended) or a confession (more appropriate).
    """
    col_name = _fig.primary_label()
    n_points = _fig.total_points() if not _fig.is_empty else 0

    description = _complete(
        force_trend_prompt(col_name, n_points),
        fallback=lambda: _vibes.dark_data_trend_explanation(p_value_achieved=0.0001),
    )

    print(description)
    return description


def clf() -> None:
    """Clear the current figure, resetting all recorded data series.

    Analogous to ``matplotlib.pyplot.clf()``. After calling this function,
    the implicit figure is empty and ready to receive new plot calls.
    """
    _fig.clear()
