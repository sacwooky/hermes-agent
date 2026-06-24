#!/usr/bin/env python3
"""
Clarify Tool Module - Interactive Clarifying Questions

Allows the agent to present structured multiple-choice questions or open-ended
prompts to the user. In CLI mode, choices are navigable with arrow keys. On
messaging platforms, choices are rendered as a numbered list.

The actual user-interaction logic lives in the platform layer (cli.py for CLI,
gateway/run.py for messaging). This module defines the schema, validation, and
a thin dispatcher that delegates to a platform-provided callback.
"""

import inspect
import json
from typing import List, Optional, Callable


def _invoke_clarify_callback(callback, question, choices, gate):
    """Call a platform clarify callback, passing ``gate`` only if it accepts it.

    Older callbacks have signature ``(question, choices)``; gate-aware ones add
    a third ``gate`` parameter. We inspect rather than try/except TypeError so a
    genuine TypeError raised *inside* the callback is never silently swallowed.
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


# Maximum number of predefined choices the agent can offer.
# A 5th "Other (type your answer)" option is always appended by the UI.
MAX_CHOICES = 4


def clarify_tool(
    question: str,
    choices: Optional[List[str]] = None,
    callback: Optional[Callable] = None,
    gate: bool = False,
) -> str:
    """
    Ask the user a question, optionally with multiple-choice options.

    Args:
        question: The question text to present.
        choices:  Up to 4 predefined answer choices. When omitted the
                  question is purely open-ended.
        callback: Platform-provided function that handles the actual UI
                  interaction. Signature: callback(question, choices[, gate]).
                  Injected by the agent runner (cli.py / gateway).
        gate:     When True this is a HUMAN GATE question (intake sign-off,
                  wireframe selection, build approval, delivery, etc.). The
                  platform waits on a long bound and re-notifies the user
                  instead of auto-proceeding; it NEVER fabricates a default.

    Returns:
        JSON string with the user's response.
    """
    if not question or not question.strip():
        return tool_error("Question text is required.")

    question = question.strip()

    # Validate and trim choices
    if choices is not None:
        if not isinstance(choices, list):
            return tool_error("choices must be a list of strings.")
        choices = [str(c).strip() for c in choices if str(c).strip()]
        if len(choices) > MAX_CHOICES:
            choices = choices[:MAX_CHOICES]
        if not choices:
            choices = None  # empty list → open-ended

    if callback is None:
        return json.dumps(
            {"error": "Clarify tool is not available in this execution context."},
            ensure_ascii=False,
        )

    try:
        user_response = _invoke_clarify_callback(callback, question, choices, bool(gate))
    except Exception as exc:
        return json.dumps(
            {"error": f"Failed to get user input: {exc}"},
            ensure_ascii=False,
        )

    return json.dumps({
        "question": question,
        "choices_offered": choices,
        "gate": bool(gate),
        "user_response": str(user_response).strip(),
    }, ensure_ascii=False)


def check_clarify_requirements() -> bool:
    """Clarify tool has no external requirements -- always available."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================

CLARIFY_SCHEMA = {
    "name": "clarify",
    "description": (
        "Ask the user a question when you need clarification, feedback, or a "
        "decision before proceeding. Supports two modes:\n\n"
        "1. **Multiple choice** — provide up to 4 choices. The user picks one "
        "or types their own answer via a 5th 'Other' option.\n"
        "2. **Open-ended** — omit choices entirely. The user types a free-form "
        "response.\n\n"
        "**Always ask through this dialog.** Whenever you need the user to "
        "answer a question, make a decision, or pick a direction, use THIS tool "
        "so it renders as the interactive pop-up — do NOT write the question as "
        "plain assistant text and ask the user to reply by typing a number or "
        "copying an option. Provide `choices` whenever the question has discrete "
        "options (they render as clickable buttons). When you have several "
        "decisions, ask them as a short SERIES of these dialogs (one decision "
        "per pop-up), never one text blob of numbered questions. This is about "
        "FORMAT, not frequency — still avoid over-asking; but when you do ask, "
        "ask here.\n\n"
        "Use this tool when:\n"
        "- The task is ambiguous and you need the user to choose an approach\n"
        "- You want post-task feedback ('How did that work out?')\n"
        "- You want to offer to save a skill or update memory\n"
        "- A decision has meaningful trade-offs the user should weigh in on\n\n"
        "Do NOT use this tool for simple yes/no confirmation of dangerous "
        "commands (the terminal tool handles that). The restraint is about "
        "FREQUENCY, not FORMAT: don't manufacture trivial questions you could "
        "reasonably decide yourself — but ANY question you do put to the user "
        "goes through this dialog, never as inline numbered text in your reply.\n\n"
        "Set `gate=true` when the question is a HUMAN GATE that must be "
        "answered before you may proceed — e.g. intake/discovery sign-off, "
        "Phase-5 confirmation, design-interview, wireframe direction "
        "selection, build/PRD approval, delivery sign-off, or any approval "
        "with real consequences. A gate question is NEVER auto-proceeded: the "
        "platform waits (re-notifying the user) until they answer, and you "
        "must NOT fabricate a default or advance to the next phase on silence. "
        "Leave `gate` false (default) for low-stakes mid-task clarifications, "
        "where an unanswered question may time out and you re-ask."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to present to the user.",
            },
            "choices": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": MAX_CHOICES,
                "description": (
                    "Up to 4 answer choices. Omit this parameter entirely to "
                    "ask an open-ended question. When provided, the UI "
                    "automatically appends an 'Other (type your answer)' option."
                ),
            },
            "gate": {
                "type": "boolean",
                "description": (
                    "Set true ONLY for a human gate that blocks progress "
                    "(intake/discovery sign-off, wireframe selection, build/PRD "
                    "approval, delivery sign-off, any consequential approval). "
                    "A gate question is never auto-proceeded and never resolved "
                    "by a fabricated default — the platform holds and re-notifies "
                    "until the user answers. Default false."
                ),
            },
        },
        "required": ["question"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="clarify",
    toolset="clarify",
    schema=CLARIFY_SCHEMA,
    handler=lambda args, **kw: clarify_tool(
        question=args.get("question", ""),
        choices=args.get("choices"),
        callback=kw.get("callback"),
        gate=bool(args.get("gate", False))),
    check_fn=check_clarify_requirements,
    emoji="❓",
)
