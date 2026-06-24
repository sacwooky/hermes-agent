"""clarify_gate — fleet three-gate clarify customization (plugin).

Holds the logic for the fleet's gate-clarify behavior so the upstream core files
carry only thin, fail-open seams. The seams import from this module by name:

    from hermes_plugins.clarify_gate import gate, enforcement

so the symbols below are re-exported at package level for convenience. There is no
runtime hook to register — the behavior is delivered via those direct imports
(the loader publishes this package as ``hermes_plugins.clarify_gate`` once the
plugin is enabled). ``register(ctx)`` is a no-op kept for manifest conformance.
"""
from __future__ import annotations

from . import gate, enforcement  # noqa: F401  (re-exported for seam imports)
from .gate import (  # noqa: F401
    get_gate_timeout,
    resolve_cli_timeout,
    cli_timeout_message,
    invoke_clarify_callback,
    HOLD_MESSAGE,
    CLI_HOLD_MESSAGE,
    RENOTIFY_TEXT,
    RENOTIFY_EVERY,
    DEFAULT_GATE_TIMEOUT,
    DEFAULT_NONGATE_TIMEOUT,
)
from .enforcement import (  # noqa: F401
    should_enforce_dialog,
    CLARIFY_CORRECTION,
    looks_like_inline_question_to_user,
)


def register(ctx) -> None:
    """No-op: clarify_gate is consumed via direct seam imports, not hooks."""
    return None
