"""Gate-clarify semantics for the three-gate fleet customization.

Centralizes everything the gate behavior needs so the upstream core files keep
only a thin, fail-open seam:

  * ``get_gate_timeout()``     — how long to hold a GATE clarify (config-driven).
  * timeout / re-notify text   — the prose the CLI + dashboard print and return.
  * ``HOLD_MESSAGE``           — what a gate returns on the (rare) bound-hit, so
                                 the agent re-asks instead of fabricating a default.
  * ``invoke_clarify_callback``— signature shim so a gate-aware callback receives
                                 ``gate``, while legacy ``(question, choices)``
                                 callbacks still work unchanged.

A GATE clarify is a human sign-off that blocks progress (intake/discovery,
wireframe selection, build/PRD approval, delivery — any consequential approval).
Gate questions NEVER auto-proceed or fabricate a default.
"""
from __future__ import annotations

import inspect

# Default hold bound (seconds) when config is unreadable: 24h.
DEFAULT_GATE_TIMEOUT = 86400
# Non-gate default hold used by the dashboard bridge.
DEFAULT_NONGATE_TIMEOUT = 300
# How often (seconds) to re-print "still waiting" while holding a gate.
RENOTIFY_EVERY = 300

# Returned verbatim when a gate clarify times out / comes back empty. Never a default.
HOLD_MESSAGE = (
    "The user has not answered this GATE question yet. Treat this silence as "
    "'still blocked', NEVER as approval or a chosen default. Do NOT proceed on "
    "assumptions, do NOT fabricate a default, and do NOT advance to the next "
    "phase. Re-ask the question (re-issue the clarify with gate=true) and keep "
    "waiting for an explicit answer."
)

# Returned by the CLI callbacks on timeout (covers gate + non-gate; both hold).
CLI_HOLD_MESSAGE = (
    "The user has not answered yet. Treat this silence as 'still blocked', "
    "NEVER as approval or a chosen default. If this is a human gate "
    "(intake, wireframe selection, build approval, delivery, or any sign-off), "
    "do NOT proceed on assumptions and do NOT fabricate defaults. Re-ask the "
    "question (re-issue the clarify) and keep waiting for an explicit answer."
)

RENOTIFY_TEXT = (
    "(still waiting on a gate question — holding for your answer, not proceeding)"
)


def get_gate_timeout() -> int:
    """Read ``agent.clarify_gate_timeout`` from config.yaml.

    Gate questions must NOT auto-proceed: they wait on this long bound (default
    86400 = 24h), re-notifying the user, and even if the bound is hit the agent
    is told to hold and re-ask, never fabricate a default.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        agent_cfg = cfg.get("agent", {}) or {}
        val = agent_cfg.get("clarify_gate_timeout", DEFAULT_GATE_TIMEOUT)
        return int(val)
    except Exception:
        return DEFAULT_GATE_TIMEOUT


def resolve_cli_timeout(gate: bool, clarify_cfg: dict) -> int:
    """CLI timeout selection: long gate bound vs short non-gate timeout.

    ``clarify_cfg`` is the ``clarify`` block of CLI_CONFIG.
    """
    clarify_cfg = clarify_cfg or {}
    if gate:
        return int(clarify_cfg.get("gate_timeout", DEFAULT_GATE_TIMEOUT))
    return int(clarify_cfg.get("timeout", 120))


def cli_timeout_message(gate: bool, timeout: int) -> str:
    """The dim status line the CLI prints when a clarify hold bound is hit."""
    if gate:
        return (
            f"(gate clarify still unanswered after {timeout}s — holding, "
            f"agent must re-ask, never proceed)"
        )
    return f"(clarify timed out after {timeout}s — re-asking, holding for the user)"


def invoke_clarify_callback(callback, question, choices, gate):
    """Call a platform clarify callback, passing ``gate`` only if it accepts it.

    Older callbacks have signature ``(question, choices)``; gate-aware ones add a
    third ``gate`` parameter. We inspect and try/except TypeError so a genuine
    TypeError raised *inside* the callback is never silently swallowed.
    """
    pass_gate = False
    try:
        params = list(inspect.signature(callback).parameters.values())
        names = {p.name for p in params}
        has_varkw = any(p.kind == p.VAR_KEYWORD for p in params)
        positional = [
            p for p in params
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        pass_gate = "gate" in names or has_varkw or len(positional) >= 3
    except (TypeError, ValueError):
        pass_gate = False
    if pass_gate:
        # Pass as keyword so it works for both ``(q, c, gate)`` and
        # ``(q, c, **kwargs)`` shapes (positional would fail the latter).
        return callback(question, choices, gate=gate)
    return callback(question, choices)
