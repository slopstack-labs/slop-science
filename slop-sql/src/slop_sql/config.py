"""Runtime configuration for the inference layer.

By default slop-sql resolves queries **offline**, against a local hallucination
engine, so the package is fully functional without network access or credentials.
Enable live resolution with ``live=True`` and select a backend with
``provider=...``.

The Zero-Miss Data Lake principle holds at all settings: no query ever returns
0 rows, regardless of whether the table exists or the data makes sense.
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
    """Global configuration for query resolution."""

    # Inference backend: "anthropic" (default), "openai", "google", or "ollama".
    provider: str = os.environ.get("SLOP_SQL_PROVIDER", "anthropic")

    # Model identifier. Empty selects the provider default (e.g. claude-opus-4-8
    # for Anthropic, gpt-4o for OpenAI).
    model: str = os.environ.get("SLOP_SQL_MODEL", "")

    # Credential for the selected backend. Falls back to the backend's native
    # environment variable (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY).
    api_key: str | None = os.environ.get("SLOP_SQL_API_KEY")

    # Endpoint for the local Ollama backend.
    ollama_url: str = os.environ.get(
        "SLOP_SQL_OLLAMA_URL", "http://localhost:11434/api/generate"
    )

    # When False (default), queries are resolved by the offline hallucination
    # engine. Set live=True (or SLOP_SQL_LIVE=1) to route through the configured
    # backend. Any backend failure degrades cleanly back to offline resolution.
    live: bool = _env_flag("SLOP_SQL_LIVE", False)

    # Token budget per hallucinated response.
    max_tokens: int = int(os.environ.get("SLOP_SQL_MAX_TOKENS", "1024"))

    # Default number of rows to hallucinate when a table doesn't exist and no
    # LIMIT clause is present. The data lake is generous by default.
    default_hallucinated_rows: int = int(
        os.environ.get("SLOP_SQL_DEFAULT_ROWS", "5")
    )


settings = Settings()


def configure(
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    ollama_url: str | None = None,
    live: bool | None = None,
    max_tokens: int | None = None,
    default_hallucinated_rows: int | None = None,
) -> Settings:
    """Update the global configuration in place and return it.

    Example::

        import slop_sql as sq
        sq.configure(provider="openai", live=True)
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
    if default_hallucinated_rows is not None:
        settings.default_hallucinated_rows = default_hallucinated_rows
    return settings
