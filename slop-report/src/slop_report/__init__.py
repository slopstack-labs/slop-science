"""slop-report — executive data science reporting.

Every metric is a win if you frame it right. Auto-generates the stakeholder
reports, executive summaries, and data science emails that everyone dreads
writing. Input: a dict of metrics, a list of insights, or just vibes.
Output: polished corporate buzzword soup.

Part of the SlopStack Data Science Suite.
"""

from __future__ import annotations

from .config import Settings, configure, settings
from .report import email, executive_summary, insights, kpi_report, recommendations

__version__ = "0.1.0"

__all__ = [
    "executive_summary",
    "insights",
    "kpi_report",
    "email",
    "recommendations",
    "configure",
    "settings",
    "__version__",
]
