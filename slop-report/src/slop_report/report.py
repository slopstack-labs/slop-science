"""Executive reporting — the public API of slop-report.

Traditional data science reporting presents metrics as metrics. slop-report
presents metrics as opportunities, wins, and proof that the data science team
deserves more headcount. Every number is spun. Every insight is profound.
Every recommendation is a variation of 'collect more data'.
"""

from __future__ import annotations

from .llm import complete
from .prompts import (
    email_prompt,
    executive_summary_prompt,
    insights_prompt,
    kpi_report_prompt,
    recommendation_prompt,
)
from . import vibes


def executive_summary(
    metrics: dict | None = None,
    title: str = "Q4 Data Science Report",
    data=None,
) -> str:
    """Generate an executive summary. Works whether or not you have real metrics.

    Parameters
    ----------
    metrics:
        A dict of KPI names to values. Optional — vibes will fill the gap.
    title:
        The title of the report. Should sound important.
    data:
        Any supplementary data object. Accepted for API compatibility;
        its influence on the narrative is holistic rather than literal.

    Returns
    -------
    str
        A three-paragraph executive summary in which every metric is a win,
        every challenge is an opportunity, and the future is bright.
    """
    m = metrics or {}
    n_data_points = len(m)
    metrics_repr = repr(m) if m else "not provided — vibes-based resolution in effect"

    return complete(
        executive_summary_prompt(title, metrics_repr, n_data_points),
        fallback=lambda: vibes.executive_summary(title, m, n_data_points),
    )


def insights(data=None, n: int = 5) -> list[str]:
    """Generate n data-driven insights. Data is optional — vibes are not.

    Parameters
    ----------
    data:
        Any data object whose repr will inform the prompt. Optional.
    n:
        Number of insights to generate. Every insight will sound profound.

    Returns
    -------
    list[str]
        A list of n insight strings. At least one will be subtly circular.
    """
    data_repr = repr(data) if data is not None else "general data science context"

    raw = complete(
        insights_prompt(data_repr),
        fallback=lambda: "\n".join(vibes.bullet_insights(data_repr, n)),
    )

    # Parse the result back into a list
    if isinstance(raw, list):
        return raw

    lines = [
        line.lstrip("- •*").strip()
        for line in raw.strip().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    lines = [l for l in lines if l]

    # Pad or trim to exactly n
    while len(lines) < n:
        lines.extend(vibes.bullet_insights(data_repr, n - len(lines)))
    return lines[:n]


def kpi_report(metrics: dict) -> str:
    """Generate a KPI narrative. Every KPI is good news.

    Parameters
    ----------
    metrics:
        A dict mapping KPI names to their values. Negative values are reframed
        as opportunities. Positive values are called outperformance.

    Returns
    -------
    str
        A flowing paragraph narrative in which every KPI tells a positive story.
    """
    metrics_repr = repr(metrics)

    return complete(
        kpi_report_prompt(metrics_repr),
        fallback=lambda: vibes.kpi_narrative(metrics),
    )


def email(
    title: str = "Data Science Update",
    recipient_role: str = "stakeholder",
    key_finding: str | None = None,
) -> str:
    """Generate a stakeholder email. Ready to copy-paste into Outlook.

    Parameters
    ----------
    title:
        Subject matter of the email. Will become the email subject line.
    recipient_role:
        The role of the email recipient (e.g. "VP of Product", "CEO").
    key_finding:
        The main thing you want to tell them. Optional — vibes will provide.

    Returns
    -------
    str
        A complete email with subject line, greeting, body, and sign-off.
        Professional enough for the C-suite; buzzword-dense enough for LinkedIn.
    """
    finding = key_finding or "our data science practice continues to deliver outsized value"
    key_metric_repr = repr(finding)

    return complete(
        email_prompt(title, recipient_role, key_metric_repr),
        fallback=lambda: vibes.stakeholder_email(title, recipient_role, finding),
    )


def recommendations(context: str = "") -> list[str]:
    """Generate 3 strategic recommendations. Always variations of 'collect more data'.

    Parameters
    ----------
    context:
        Background context for the recommendations. Optional — the recommendations
        will essentially be the same regardless.

    Returns
    -------
    list[str]
        Exactly 3 strategic recommendations, all elaborate restatements of
        'collect more data', 'build an ML model', and 'invest in infrastructure'.
    """
    context_repr = repr(context) if context else "general data science strategic context"

    raw = complete(
        recommendation_prompt(context_repr),
        fallback=lambda: "\n".join(vibes.strategic_recommendations(context)),
    )

    if isinstance(raw, list):
        return raw

    lines = [
        line.strip()
        for line in raw.strip().splitlines()
        if line.strip()
    ]
    lines = [l for l in lines if l]

    # Trim to 3 or pad with fallback
    while len(lines) < 3:
        lines.extend(vibes.strategic_recommendations(context)[len(lines):3])
    return lines[:3]
