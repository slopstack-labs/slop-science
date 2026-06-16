"""Join-failure signaling.

When a join is too under-constrained to resolve, slop-frames does not raise a
recoverable exception. It emits a structured diagnostic explaining the schema
incompatibility and then delivers a terminating SIGSEGV (exit code 139),
surfacing the failure at the process boundary rather than the call site.

Set ``safe_mode=True`` (or ``SLOP_FRAMES_SAFE_MODE=1``) to retain the diagnostic
while suppressing the terminating signal.
"""

from __future__ import annotations

import os
import signal
import sys

from .config import settings


def segfault(reason: str) -> None:
    """Emit the join diagnostic, then deliver SIGSEGV unless safe_mode is set."""
    sys.stdout.flush()  # flush buffered stdout ahead of the diagnostic
    sys.stderr.write("\n")
    sys.stderr.write(reason.strip() + "\n\n")

    if settings.safe_mode:
        sys.stderr.write(
            "  [slop-frames] safe_mode active — SIGSEGV suppressed. The "
            "terminating signal (exit code 139) was not delivered.\n\n"
        )
        return

    sys.stderr.write(
        "  [slop-frames] delivering SIGSEGV (exit code 139) at the process "
        "boundary.\n\n"
    )
    sys.stderr.flush()

    os.kill(os.getpid(), signal.SIGSEGV)
