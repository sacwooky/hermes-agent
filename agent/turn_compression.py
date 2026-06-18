"""Mid-loop system-prompt mutation seams for ``run_conversation``.

Extracted from ``agent/conversation_loop.py`` as part of the prompt-cache
boundary hardening (advisor plan 010a). The ~3,900-line turn loop reassigned
``active_system_prompt`` mid-loop at several points and wrote back
``agent._cached_system_prompt``. Until every such mutation flowed through a
single named, auditable seam, no inner-loop extraction could be proven
cache-safe (AGENTS.md prompt-cache boundary).

This module provides the two seams:

- ``apply_turn_compression`` — wraps the identical ``agent._compress_context``
  call plus the unconditional ``conversation_history = None`` reset that
  followed each of the four in-loop compression sites. Behavior-neutral: it
  performs exactly what the inline code did, mutating ``agent`` as a side
  effect and returning the locals the caller reads back.
- ``sanitize_active_system_prompt`` — wraps the non-ASCII sanitization
  write-back to ``agent._cached_system_prompt``, including the "only write if
  changed" guard. ``_strip_non_ascii`` is injected as a parameter (mirroring
  the convention in ``turn_context`` / ``turn_finalizer``) to keep the import
  direction one-way.

Behavior identical to the original inline code; pure move-and-name refactor
with no semantic change.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple


def apply_turn_compression(
    agent: Any,
    messages: list,
    system_message: str,
    *,
    approx_tokens: Any,
    task_id: str,
) -> Tuple[list, Any, None]:
    """Run mid-loop context compression and clear replayed history.

    Reproduces verbatim the identical block that followed each of the four
    in-loop ``agent._compress_context`` call sites: the compress call itself
    (parameterized by ``approx_tokens`` / ``task_id`` since one site passes a
    different ``approx_tokens`` source) and the unconditional
    ``conversation_history = None`` reset.

    Returns ``(messages, active_system_prompt, conversation_history)`` where
    ``conversation_history`` is always ``None`` — compression created a new
    session, so history must be cleared so ``_flush_messages_to_session_db``
    writes the compressed messages to the new session instead of skipping
    them.
    """
    messages, active_system_prompt = agent._compress_context(
        messages,
        system_message,
        approx_tokens=approx_tokens,
        task_id=task_id,
    )
    # Compression created a new session — clear history
    # so _flush_messages_to_session_db writes compressed
    # messages to the new session, not skipping them.
    conversation_history = None
    return messages, active_system_prompt, conversation_history


def sanitize_active_system_prompt(
    agent: Any,
    active_system_prompt: Any,
    *,
    strip_non_ascii: Callable[[str], str],
) -> Tuple[Any, bool]:
    """Strip non-ASCII from the active system prompt and write it back.

    Reproduces verbatim the inline ``active_system_prompt`` sanitization
    block, including the "only write if changed" guard. When the prompt is a
    string and stripping changes it, the cleaned value is written back to both
    the local ``active_system_prompt`` and ``agent._cached_system_prompt``,
    and the change is reported.

    Returns ``(active_system_prompt, system_was_sanitized)``.
    """
    system_was_sanitized = False
    if isinstance(active_system_prompt, str):
        _sanitized_system = strip_non_ascii(active_system_prompt)
        if _sanitized_system != active_system_prompt:
            active_system_prompt = _sanitized_system
            agent._cached_system_prompt = _sanitized_system
            system_was_sanitized = True
    return active_system_prompt, system_was_sanitized
