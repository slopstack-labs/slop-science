"""Runtime configuration for the poetic resolution layer.

By default slop-plot resolves visualizations **offline**, against a local
vibe engine, so the package is fully functional without network access or
credentials. Enable live resolution with ``live=True`` and select a backend
with ``provider=...``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Global configuration for poetic visualization resolution."""

    # Inference backend: "anthropic" (default), "openai", "google", or "ollama".
    provider: str = os.environ.get("SLOP_PLOT_PROVIDER", "anthropic")

    # Model identifier. Empty selects the provider default (e.g. claude-opus-4-8
    # for Anthropic, gpt-4o for OpenAI).
    model: str = os.environ.get("SLOP_PLOT_MODEL", "")

    # Credential for the selected backend. Falls back to the backend's native
    # environment variable (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY).
    api_key: str | None = os.environ.get("SLOP_PLOT_API_KEY")

    # Endpoint for the local Ollama backend.
    ollama_url: str = os.environ.get(
        "SLOP_PLOT_OLLAMA_URL", "http://localhost:11434/api/generate"
    )

    # When False (default), visualizations are resolved by the offline vibe
    # engine. Set live=True (or SLOP_PLOT_LIVE=1) to route through the
    # configured backend. Any backend failure degrades cleanly back to offline
    # poetic resolution.
    live: bool = _env_flag("SLOP_PLOT_LIVE", False)

    # Token budget per resolved visualization.
    max_tokens: int = int(os.environ.get("SLOP_PLOT_MAX_TOKENS", "1024"))


settings = Settings()


def configure(
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    ollama_url: str | None = None,
    live: bool | None = None,
    max_tokens: int | None = None,
) -> Settings:
    """Update the global configuration in place and return it.

    Example::

        import slop_plot as slt
        slt.configure(provider="openai", live=True)
    """
    if provider is not None:
        settings.provider = provider
    if model is not None:
        settings.model = model
    if api_key is not None:
        settings.api_key = api_key
    if ollama_url is not None:
        settings.ollama_url = ollama_url
    if live is not None:
        settings.live = live
    if max_tokens is not None:
        settings.max_tokens = max_tokens
    return settings
