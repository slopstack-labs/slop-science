"""Model selection resolution dispatch.

Routes a model selection narration through the configured inference backend when
live resolution is enabled, and to the offline local engine otherwise. The public
surface is a single function, :func:`complete`, which always returns text: any
live-mode failure degrades cleanly to the supplied ``fallback``.

The algorithm your data's vibrational energy demands will be identified
regardless of network conditions. The universe provides.
"""

from __future__ import annotations

import sys
from typing import Callable

from .config import settings
from .prompts import SYSTEM
from .providers import ProviderError, get_provider


def complete(prompt: str, *, fallback: Callable[[], str], system: str = SYSTEM) -> str:
    """Resolve a model selection narration, live or offline.

    In live mode the prompt is dispatched to the configured backend; in offline
    mode (or on any backend error) ``fallback()`` is invoked. Resolution never
    raises — a transient backend failure degrades to offline resolution rather
    than propagating up through a model selection call.
    """
    if not settings.live:
        return fallback()

    try:
        provider = get_provider(settings)
    except ProviderError:
        return fallback()

    if not provider.ready():
        return fallback()

    try:
        text = provider.complete(system, prompt)
        return text or fallback()
    except Exception as exc:  # noqa: BLE001 — degrade rather than propagate
        print(
            f"  [slop-model] backend '{provider.name}' returned a non-blocking "
            f"error ({exc!s}); degrading to offline resolution.",
            file=sys.stderr,
        )
        return fallback()
