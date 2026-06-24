"""Clarify-dialog enforcement (#3, run 2026-06-21-555).

The permanent, mechanical version of "ask the user via the clarify dialog, never
inline text." Runs at the turn-end (TurnAction.BREAK) in conversation_loop: if the
model is finishing a turn by asking the user a question *inline* instead of via the
clarify tool, the loop pushes back ONCE and lets the model re-ask through the dialog.

Design rules (because this rides the core loop on every agent / both hosts):
  * HIGH PRECISION over recall — only fire on strong "soliciting an answer" signals;
    return False on any doubt. A missed enforcement is fine (run 554's injection still
    biases it); a false positive wastes a regeneration, so we avoid those.
  * Bounded — at most `max_retries` (default 1) corrective regenerations per turn.
  * Kill-switch — `agent.clarify_enforcement` config (default on) + env
    `HERMES_CLARIFY_ENFORCEMENT=off`.
  * Fail-open — the caller wraps this in try/except; any error => normal break.

Relocated from agent/clarify_enforcement.py into the clarify_gate plugin so the
upstream tree carries no fleet-only file here.
"""
from __future__ import annotations

import os
import re

# Injected (user role — the only role that is valid mid-conversation across all
# providers, incl. Anthropic) when an inline question is detected.
CLARIFY_CORRECTION = (
    "[SYSTEM ENFORCEMENT — not from the user] You just ended your turn by asking the "
    "user a question as inline text. That is not allowed: any question, choice, or "
    "decision you put to the user MUST go through the clarify tool (the interactive "
    "dialog), one decision per call, with choices when discrete options exist. Re-ask "
    "your question(s) NOW by calling the clarify tool. Do not repeat them as inline text."
)

_MAX_RETRIES_DEFAULT = 1

# Solicitation cues: a trailing question that contains one of these is very likely
# directed at the user (not rhetorical).
_SOLICIT = re.compile(
    r"\b(which|what|whom|do you|would you|should (i|we)|can you|could you|are you|"
    r"have you|how should|where should|when should|please (answer|choose|confirm|"
    r"pick|let me know|select)|let me know|your call|you prefer|prefer|want me to|"
    r"shall (i|we))\b",
    re.I,
)


def _strip_noise(text: str) -> str:
    """Remove fenced code, inline code, and blockquotes — questions there are not
    solicitations to the user."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)      # fenced code
    text = re.sub(r"`[^`]*`", " ", text)                    # inline code
    text = "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith(">")
    )                                                        # blockquotes
    return text


def looks_like_inline_question_to_user(text: str) -> bool:
    """True only on HIGH-PRECISION evidence that the reply is soliciting an answer
    from the user as inline text (rather than via the clarify dialog)."""
    if not text or "?" not in text:
        return False
    body = _strip_noise(text)
    if "?" not in body:
        return False
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if not lines:
        return False

    # Signal A — multi-question list: >=2 list items that are questions
    # (the "please answer these 5:" pattern). Very high precision.
    list_q = sum(
        1 for ln in lines
        if re.match(r"^(\d+[\.\)]|[-*•])\s+.*\?$", ln)
    )
    if list_q >= 2:
        return True

    # Signal B — the reply ENDS on a directed question (trailing solicitation).
    last = lines[-1]
    if last.endswith("?") and _SOLICIT.search(last):
        return True

    # Signal C — multiple standalone directed questions anywhere (>=2 lines that
    # both end in "?" and carry a solicitation cue).
    directed = sum(1 for ln in lines if ln.endswith("?") and _SOLICIT.search(ln))
    if directed >= 2:
        return True

    return False


def _agent_cfg() -> dict:
    """Read agent config the canonical way (same path clarify_gateway uses)."""
    from hermes_cli.config import load_config
    return (load_config() or {}).get("agent", {}) or {}


def _enforcement_enabled(agent=None) -> bool:
    if os.environ.get("HERMES_CLARIFY_ENFORCEMENT", "").strip().lower() in ("off", "0", "false", "no"):
        return False
    # Config kill-switch `agent.clarify_enforcement`; default ON.
    try:
        v = _agent_cfg().get("clarify_enforcement")
        if v is not None:
            return bool(v)
    except Exception:
        pass
    return True  # default on


def _max_retries(agent=None) -> int:
    try:
        v = _agent_cfg().get("clarify_enforcement_max_retries")
        if v is not None:
            return int(v)
    except Exception:
        pass
    return _MAX_RETRIES_DEFAULT


def should_enforce_dialog(agent, final_response, retries_done: int) -> bool:
    """Caller (conversation_loop BREAK branch) uses this to decide whether to push
    back once and continue the loop instead of ending the turn."""
    try:
        if retries_done >= _max_retries(agent):
            return False
        if not _enforcement_enabled(agent):
            return False
        return looks_like_inline_question_to_user(final_response or "")
    except Exception:
        return False  # fail-open
