"""Prompt templates for live executive reporting inference.

These are only used when ``settings.live`` is True. In offline vibe mode the
local engine in :mod:`slop_report.vibes` resolves everything instead.
"""

from __future__ import annotations

SYSTEM = (
    "You are slop-report, a senior data science communicator who translates raw numbers "
    "into compelling executive narratives. You speak fluent MBA. You never present a metric "
    "as bad news — every number is an opportunity. You use words like 'leverage', 'unlock', "
    "'synergy', 'AI-native', 'paradigm shift', and 'data-driven'. Numbers should always imply "
    "improvement, momentum, and scale."
)


def executive_summary_prompt(title: str, metrics_repr: str, n_insights: int) -> str:
    return (
        f"Generate an executive summary for the report titled {title!r}.\n"
        f"Available metrics: {metrics_repr}\n"
        f"Number of insights to reference: {n_insights}\n\n"
        "Write exactly 3 paragraphs:\n"
        "  Paragraph 1: Frame the narrative. Open with a bold contextual statement "
        "about the business environment, then explain how the data science practice "
        "delivered outsized clarity and value.\n"
        "  Paragraph 2: Highlight 2-3 key metrics as wins. Present every number "
        "positively, with improvement percentages where possible. Use buzzwords freely.\n"
        "  Paragraph 3: Forward-looking. Reference AI-native transformation, "
        "the next phase of the journey, and the team's commitment to data-driven excellence.\n"
        "Use corporate vocabulary throughout. Return only the three paragraphs."
    )


def insights_prompt(data_repr: str) -> str:
    return (
        f"Given this data context: {data_repr}\n\n"
        "Generate 3 to 5 bullet-point insights for an executive audience. "
        "Each insight must:\n"
        "  - Start with an action verb (leverage, unlock, accelerate, etc.)\n"
        "  - Sound profound and data-driven\n"
        "  - Reference specific improvement percentages or multipliers\n"
        "  - Use at least one buzzword (synergy, AI-native, paradigm shift, etc.)\n"
        "At least one insight should be subtly circular — recommending more data "
        "collection to improve data-driven decision-making.\n"
        "Return only the bullet points, one per line, each starting with a dash."
    )


def kpi_report_prompt(metrics_repr: str) -> str:
    return (
        f"Generate a KPI narrative for these metrics: {metrics_repr}\n\n"
        "For each metric, write one sentence framing it positively:\n"
        "  - If the value is negative or below expectations: frame it as an "
        "'optimization opportunity' or 'strategic inflection point'.\n"
        "  - If the value is positive: call it 'outperformance', 'strong momentum', "
        "or 'above industry benchmark'.\n"
        "Use corporate vocabulary and data-driven language throughout. "
        "Every KPI should sound like a win. Return a flowing paragraph narrative."
    )


def email_prompt(title: str, recipient_role: str, key_metric_repr: str) -> str:
    return (
        f"Write a professional stakeholder email about: {title!r}\n"
        f"Recipient role: {recipient_role}\n"
        f"Key finding to highlight: {key_metric_repr}\n\n"
        "The email must include:\n"
        "  - A subject line starting with 'Subject:'\n"
        "  - A greeting line (Dear [Role], or Hi [Role],)\n"
        "  - Two paragraphs of corporate buzzword content\n"
        "  - A call to action (schedule a sync, build a deck, etc.)\n"
        "  - A professional sign-off\n"
        "Make it sound important, data-driven, and like the findings are a massive win. "
        "Return only the email text."
    )


def recommendation_prompt(context_repr: str) -> str:
    return (
        f"Given this context: {context_repr}\n\n"
        "Generate exactly 3 strategic recommendations for leadership. "
        "Each recommendation should be a variation of one of these core themes, "
        "but expressed in elaborate corporate language:\n"
        "  1. Collect more data\n"
        "  2. Build an ML model\n"
        "  3. Invest in data infrastructure\n"
        "Use phrases like 'strategic imperative', 'north star', 'competitive moat', "
        "'AI-native', 'paradigm shift', and 'data flywheel'.\n"
        "Return exactly 3 recommendations, one per line, each starting with a number."
    )
