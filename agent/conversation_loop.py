"""The agent conversation loop — extracted from ``run_agent.AIAgent``.

This is the biggest single chunk pulled out of ``run_agent.py``: the
roughly 3,900-line :func:`run_conversation` body that drives one user
turn through the agent (model call, tool dispatch, retries, fallbacks,
compression, post-turn hooks, background memory/skill review nudges).

The function takes the parent ``AIAgent`` instance as its first
argument (``agent``) and accesses its state via attribute lookup.
``_ra().AIAgent.run_conversation`` is now a thin forwarder.

Symbols that production code or tests patch on ``run_agent`` directly
(``handle_function_call``, ``_set_interrupt``, ``OpenAI``, ...) are
resolved through :func:`_ra` so those patches keep working.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import ssl
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from agent.codex_responses_adapter import _summarize_user_message_for_log
from agent.display import KawaiiSpinner
from agent.error_classifier import FailoverReason, classify_api_error
from agent.iteration_budget import IterationBudget
from agent.turn_compression import (
    apply_turn_compression,
    sanitize_active_system_prompt,
)
from agent.turn_api_call import run_api_call_with_retry
from agent.turn_context import build_turn_context
from agent.turn_retry_state import TurnRetryState
from agent.memory_manager import build_memory_context_block
from agent.message_sanitization import (
    _repair_tool_call_arguments,
    _sanitize_messages_non_ascii,
    _sanitize_messages_surrogates,
    _sanitize_structure_non_ascii,
    _sanitize_structure_surrogates,
    _sanitize_surrogates,
    _sanitize_tools_non_ascii,
    _strip_images_from_messages,
    _strip_non_ascii,
)
from agent.model_metadata import (
    MINIMUM_CONTEXT_LENGTH,
    estimate_messages_tokens_rough,
    estimate_request_tokens_rough,
    get_context_length_from_provider_error,
    parse_available_output_tokens_from_error,
    save_context_length,
)
from agent.process_bootstrap import _install_safe_stdio
from agent.prompt_caching import apply_anthropic_cache_control
from agent.retry_utils import jittered_backoff
from agent.trajectory import has_incomplete_scratchpad
from agent.usage_pricing import estimate_usage_cost, normalize_usage
from hermes_constants import PARTIAL_STREAM_STUB_ID
from hermes_logging import set_session_context
from tools.skill_provenance import set_current_write_origin
from utils import base_url_host_matches, env_var_enabled

logger = logging.getLogger(__name__)

# Stable prefix of the local interrupt status string emitted when a turn is
# cancelled while waiting on the provider. Surfaces (ACP, TUI) match on this
# to treat it as cancellation metadata rather than assistant prose.
INTERRUPT_WAITING_FOR_MODEL_PREFIX = "Operation interrupted: waiting for model response ("


def _image_error_max_dimension(error: Exception) -> Optional[int]:
    """Extract a provider-reported image dimension ceiling, if present."""
    parts = []
    for value in (
        error,
        getattr(error, "message", None),
        getattr(error, "body", None),
    ):
        if value:
            try:
                parts.append(str(value))
            except Exception:
                pass
    text = " ".join(parts).lower()
    if "image" not in text or "dimension" not in text or "max allowed size" not in text:
        return None

    match = re.search(r"max allowed size(?:\s+for [^:]+)?:\s*(\d{3,5})\s*pixels?", text)
    if not match:
        return None
    try:
        max_dimension = int(match.group(1))
    except ValueError:
        return None
    if 512 <= max_dimension <= 8000:
        return max_dimension
    return None


def _ollama_context_limit_error(agent: Any, request_tokens: int) -> Optional[str]:
    """Return a user-facing error when Ollama is loaded with too little context."""
    if not getattr(agent, "tools", None):
        return None

    runtime_ctx = getattr(agent, "_ollama_num_ctx", None)
    if not isinstance(runtime_ctx, int) or runtime_ctx <= 0:
        return None
    if runtime_ctx >= MINIMUM_CONTEXT_LENGTH:
        return None

    model = getattr(agent, "model", "") or "the selected model"
    base_url = getattr(agent, "base_url", "") or "unknown base URL"
    provider = getattr(agent, "provider", "") or "unknown"
    tool_count = len(getattr(agent, "tools", None) or [])

    logger.warning(
        "Ollama runtime context too small for Hermes tool use: "
        "model=%s provider=%s base_url=%s runtime_context=%d "
        "minimum_context=%d estimated_request_tokens=%d tool_count=%d "
        "session=%s",
        model,
        provider,
        base_url,
        runtime_ctx,
        MINIMUM_CONTEXT_LENGTH,
        request_tokens,
        tool_count,
        getattr(agent, "session_id", None) or "none",
    )

    return (
        f"Ollama loaded `{model}` with only {runtime_ctx:,} tokens of runtime "
        f"context, but Hermes needs at least {MINIMUM_CONTEXT_LENGTH:,} tokens "
        "for reliable tool use.\n\n"
        "Increase the Ollama context for this model and restart/reload the "
        "model before trying again. A known-good starting point is 65,536 "
        "tokens. In Hermes config, set `model.ollama_num_ctx: 65536` "
        "(and `model.context_length: 65536` if you also override the displayed "
        "model context). If you manage the model through an Ollama Modelfile, "
        "set `PARAMETER num_ctx 65536` there instead."
    )


def _ra():
    """Lazy reference to ``run_agent`` so callers can patch
    ``run_agent.handle_function_call`` / ``run_agent._set_interrupt`` /
    ``run_agent.OpenAI`` and have those patches reach this code path.
    """
    import run_agent
    return run_agent


def _nous_entitlement_message(capability: str) -> str:
    try:
        from hermes_cli.nous_account import (
            format_nous_portal_entitlement_message,
            get_nous_portal_account_info,
        )

        account_info = get_nous_portal_account_info(force_fresh=True)
        message = format_nous_portal_entitlement_message(
            account_info,
            capability=capability,
        )
        return message or ""
    except Exception:
        return ""


def _print_nous_entitlement_guidance(agent, capability: str) -> bool:
    message = _nous_entitlement_message(capability)
    if not message:
        return False
    for line in message.splitlines():
        agent._vprint(f"{agent.log_prefix}   💡 {line}", force=True)
    return True


def _is_nous_inference_route(provider: str, base_url: str) -> bool:
    provider = (provider or "").strip().lower()
    if provider == "nous":
        return True
    base = str(base_url or "")
    return (
        base_url_host_matches(base, "inference-api.nousresearch.com")
        or base_url_host_matches(base, "inference.nousresearch.com")
    )


def _billing_or_entitlement_message(
    *,
    capability: str,
    provider: str,
    base_url: str,
    model: str,
) -> str:
    if _is_nous_inference_route(provider, base_url):
        return _nous_entitlement_message(capability)

    provider_label = (provider or "").strip() or "the selected provider"
    model_label = (model or "").strip() or "the selected model"
    lines = [
        (
            f"{provider_label} reported that billing, credits, or account "
            f"entitlement is exhausted for {model_label}."
        ),
        "Add credits or update billing with that provider, then retry.",
    ]
    if base_url_host_matches(str(base_url or ""), "openrouter.ai"):
        lines.append("OpenRouter credits: https://openrouter.ai/settings/credits")
    lines.append("You can switch providers temporarily with /model <model> --provider <provider>.")
    return "\n".join(lines)


def _print_billing_or_entitlement_guidance(
    agent,
    *,
    capability: str,
    provider: str,
    base_url: str,
    model: str,
) -> bool:
    message = _billing_or_entitlement_message(
        capability=capability,
        provider=provider,
        base_url=base_url,
        model=model,
    )
    if not message:
        return False
    for line in message.splitlines():
        agent._vprint(f"{agent.log_prefix}   💡 {line}", force=True)
    return True


def _try_refresh_nous_paid_entitlement_credentials(agent) -> bool:
    """Refresh Nous runtime credentials after a fresh paid-entitlement check."""
    try:
        from hermes_cli.nous_account import get_nous_portal_account_info

        account_info = get_nous_portal_account_info(force_fresh=True)
        if account_info.paid_service_access is not True:
            return False
        return agent._try_refresh_nous_client_credentials(
            force=True,
        )
    except Exception:
        return False


def _restore_or_build_system_prompt(agent, system_message, conversation_history):
    """Restore the cached system prompt from the session DB or build it fresh.

    Mutates ``agent._cached_system_prompt`` and persists a freshly-built
    prompt back to the session DB on first build.  Extracted from
    ``run_conversation`` so the prefix-cache restore path can be tested in
    isolation.

    Three-way state distinction for the stored row, surfaced via logs so
    silent prefix-cache misses are visible in ``agent.log``:

      * ``missing`` — no session row yet (legitimate first turn).
      * ``null``   — row exists, ``system_prompt`` column is NULL.
        Legacy session predating system-prompt persistence, or a migration
        leftover.  Warns when ``conversation_history`` is non-empty.
      * ``empty``  — row exists, ``system_prompt`` column is the empty
        string.  Indicates a previous-turn write that ran but stored
        nothing (silent persistence bug).  Always warns.
      * ``present`` — row exists with a usable prompt → reused verbatim.

    Read or write failures against the session DB log at WARNING (not
    DEBUG) so persistent issues (disk full, schema drift, lock contention)
    surface without needing verbose mode.  This used to be a debug-level
    log that silently broke prefix-cache reuse on the gateway path
    (which constructs a fresh ``AIAgent`` per turn and depends on this
    DB roundtrip).
    """
    stored_prompt = None
    stored_state = "missing"
    if conversation_history and agent._session_db:
        try:
            session_row = agent._session_db.get_session(agent.session_id)
            if session_row is not None:
                raw_prompt = session_row.get("system_prompt")
                if raw_prompt is None:
                    stored_state = "null"
                elif raw_prompt == "":
                    stored_state = "empty"
                else:
                    stored_prompt = raw_prompt
                    stored_state = "present"
        except Exception as exc:
            logger.warning(
                "Session DB get_session failed for system-prompt restore "
                "(session=%s): %s. Falling back to fresh build — prefix "
                "cache will miss for this turn.",
                agent.session_id, exc,
            )

    if stored_prompt and _stored_prompt_matches_runtime(agent, stored_prompt):
        # Continuing session — reuse the exact system prompt from the
        # previous turn so the Anthropic cache prefix matches.
        agent._cached_system_prompt = stored_prompt
        return
    if stored_prompt:
        stored_state = "stale_runtime"
        logger.info(
            "Stored system prompt for session %s has stale runtime identity; "
            "rebuilding for model=%s provider=%s.",
            agent.session_id,
            getattr(agent, "model", "") or "",
            getattr(agent, "provider", "") or "",
        )

    if conversation_history and stored_state in ("null", "empty"):
        # Continuing session whose stored prompt is unusable.  The
        # previous turn's write either never happened or wrote an empty
        # string — either way every turn now rebuilds and the prefix
        # cache misses every time.
        logger.warning(
            "Stored system prompt for session %s is %s; rebuilding "
            "from scratch this turn. Prefix cache will miss until "
            "the rebuild persists. Investigate the previous turn's "
            "update_system_prompt write path.",
            agent.session_id, stored_state,
        )

    # First turn of a new session (or recovering from a broken stored
    # prompt) — build from scratch.
    agent._cached_system_prompt = agent._build_system_prompt(system_message)

    # Plugin hook: on_session_start — fired once when a brand-new
    # session is created (not on continuation).  Plugins can use this
    # to initialise session-scoped state (e.g. warm a memory cache).
    try:
        from hermes_cli.plugins import invoke_hook as _invoke_hook
        _invoke_hook(
            "on_session_start",
            session_id=agent.session_id,
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
        )
    except Exception as exc:
        logger.warning("on_session_start hook failed: %s", exc)

    # Cold-start credits seed (L3) — fallback for the first-turn path. The TUI/
    # desktop build seeds at session OPEN (see seed_credits_at_session_start in
    # tui_gateway), so this call is usually a no-op there (idempotent: skips when
    # _credits_state already exists). For the plain CLI / any path that didn't seed
    # at build, it primes credits state from /api/oauth/account (or a fixture) on the
    # first turn so depletion / usage-band warnings fire. Fail-open inside the helper.
    try:
        from agent.credits_tracker import seed_credits_at_session_start

        seed_credits_at_session_start(agent)
    except Exception:
        logger.debug("cold-start credits seed failed (fail-open)", exc_info=True)

    # Persist the system prompt snapshot in SQLite.  Failure here used
    # to log at DEBUG, which silently broke prefix-cache reuse on the
    # gateway path (fresh AIAgent per turn → reads from this row every
    # subsequent turn).
    if agent._session_db:
        try:
            agent._session_db.update_system_prompt(agent.session_id, agent._cached_system_prompt)
        except Exception as exc:
            logger.warning(
                "Session DB update_system_prompt failed for session %s: "
                "%s. Subsequent turns will rebuild the system prompt and "
                "miss the prefix cache.",
                agent.session_id, exc,
            )


def _stored_prompt_matches_runtime(agent, prompt: str) -> bool:
    """Return False when the persisted Model/Provider lines are stale."""

    def line_value(label: str) -> str:
        prefix = f"{label}:"
        value = ""
        for line in prompt.splitlines():
            if line.startswith(prefix):
                value = line[len(prefix):].strip()
        return value

    stored_model = line_value("Model")
    current_model = str(getattr(agent, "model", "") or "").strip()
    if stored_model and current_model and stored_model != current_model:
        return False

    stored_provider = line_value("Provider")
    current_provider = str(getattr(agent, "provider", "") or "").strip()
    if stored_provider and current_provider and stored_provider != current_provider:
        return False

    return True


def _get_continuation_prompt(is_partial_stub: bool, dropped_tools: Optional[List[str]] = None) -> str:
    if is_partial_stub and dropped_tools:
        tool_list = ", ".join(dropped_tools[:3])
        return (
            "[System: Your previous tool call "
            f"({tool_list}) was too large and "
            "the stream timed out before it "
            "could be delivered. Do NOT retry "
            "the same tool call with the same "
            "large content. Instead, break the "
            "content into multiple smaller tool "
            "calls (e.g. use multiple patch calls "
            "or write smaller files). Each tool "
            "call's arguments must be under ~8K "
            "tokens to avoid stream timeouts.]"
        )
    elif is_partial_stub:
        return (
            "[System: The previous response was cut off by a "
            "network error mid-stream. Continue exactly where "
            "you left off. Do not restart or repeat prior text. "
            "Finish the answer directly.]"
        )
    else:
        return (
            "[System: Your previous response was truncated by the output "
            "length limit. Continue exactly where you left off. Do not "
            "restart or repeat prior text. Finish the answer directly.]"
        )


# Shared recovery hint appended to every content-policy refusal message. Both
# the HTTP-200 refusal path (``finish_reason=content_filter``) and the
# exception path (a provider moderation error classified as
# ``content_policy_blocked``) end with the same actionable next steps, so they
# share one trailer to keep the guidance from drifting between the two sites.
_CONTENT_POLICY_RECOVERY_HINT = (
    "Try rephrasing the request, narrowing the context, or "
    "adding a fallback provider with `hermes fallback add`."
)


def _content_policy_blocked_result(
    messages: List[Dict],
    api_call_count: int,
    *,
    final_response: str,
    error_detail: str,
) -> Dict[str, Any]:
    """Build the terminal turn result for a content-policy block.

    A content-policy refusal is deterministic for the unchanged prompt, so the
    turn ends here (no retry). Both the HTTP-200 refusal handler and the
    exception-path handler return the identical shape — a failed, non-completed
    turn carrying the user-facing message and a ``content_policy_blocked:``
    prefixed error — so they funnel through this one builder.
    """
    return {
        "final_response": final_response,
        "messages": messages,
        "api_calls": api_call_count,
        "completed": False,
        "failed": True,
        "error": f"content_policy_blocked: {error_detail}",
    }


def run_conversation(
    agent,
    user_message: str,
    system_message: str = None,
    conversation_history: List[Dict[str, Any]] = None,
    task_id: str = None,
    stream_callback: Optional[callable] = None,
    persist_user_message: Optional[str] = None,
    persist_user_timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run a complete conversation with tool calling until completion.

    Args:
        user_message (str): The user's message/question
        system_message (str): Custom system message (optional, overrides ephemeral_system_prompt if provided)
        conversation_history (List[Dict]): Previous conversation messages (optional)
        task_id (str): Unique identifier for this task to isolate VMs between concurrent tasks (optional, auto-generated if not provided)
        stream_callback: Optional callback invoked with each text delta during streaming.
            Used by the TTS pipeline to start audio generation before the full response.
            When None (default), API calls use the standard non-streaming path.
        persist_user_message: Optional clean user message to store in
            transcripts/history when user_message contains API-only
            synthetic prefixes.
        persist_user_timestamp: Optional platform event timestamp to store
            as metadata on that persisted user message.
                or queuing follow-up prefetch work.

    Returns:
        Dict: Complete conversation result with final response and message history
    """
    # ── Per-turn setup (the prologue) ──
    # All once-per-turn setup — stdio guarding, retry-counter resets, user
    # message sanitization, todo/nudge hydration, system-prompt restore-or-
    # build, crash-resilience persistence, preflight compression, the
    # ``pre_llm_call`` plugin hook, and external-memory prefetch — lives in
    # ``build_turn_context``.  It mutates ``agent`` exactly as the inline code
    # did and returns the locals the loop below reads back.  See
    # ``agent/turn_context.py``.
    _ctx = build_turn_context(
        agent,
        user_message,
        system_message,
        conversation_history,
        task_id,
        stream_callback,
        persist_user_message,
        persist_user_timestamp,
        restore_or_build_system_prompt=_restore_or_build_system_prompt,
        install_safe_stdio=_install_safe_stdio,
        sanitize_surrogates=_sanitize_surrogates,
        summarize_user_message_for_log=_summarize_user_message_for_log,
        set_session_context=set_session_context,
        set_current_write_origin=set_current_write_origin,
        ra=_ra,
    )
    user_message = _ctx.user_message
    original_user_message = _ctx.original_user_message
    messages = _ctx.messages
    conversation_history = _ctx.conversation_history
    active_system_prompt = _ctx.active_system_prompt
    effective_task_id = _ctx.effective_task_id
    turn_id = _ctx.turn_id
    current_turn_user_idx = _ctx.current_turn_user_idx
    _should_review_memory = _ctx.should_review_memory
    _plugin_user_context = _ctx.plugin_user_context
    _ext_prefetch_cache = _ctx.ext_prefetch_cache

    # Main conversation loop counters (pure locals consumed by the loop below).
    api_call_count = 0
    final_response = None
    interrupted = False
    failed = False
    codex_ack_continuations = 0
    length_continue_retries = 0
    truncated_tool_call_retries = 0
    truncated_response_parts: List[str] = []
    compression_attempts = 0
    _turn_exit_reason = "unknown"  # Diagnostic: why the loop ended

    # Optional opt-in runtime: if api_mode == codex_app_server, hand the
    # turn to the codex app-server subprocess (terminal/file ops/patching
    # all run inside Codex). Default Hermes path is bypassed entirely.
    # See agent/transports/codex_app_server_session.py for the adapter
    # and references/codex-app-server-runtime.md for the rationale.
    if agent.api_mode == "codex_app_server":
        return agent._run_codex_app_server_turn(
            user_message=user_message,
            original_user_message=original_user_message,
            messages=messages,
            effective_task_id=effective_task_id,
            should_review_memory=_should_review_memory,
        )

    while (api_call_count < agent.max_iterations and agent.iteration_budget.remaining > 0) or agent._budget_grace_call:
        # Reset per-turn checkpoint dedup so each iteration can take one snapshot
        agent._checkpoint_mgr.new_turn()

        # Check for interrupt request (e.g., user sent new message)
        if agent._interrupt_requested:
            interrupted = True
            _turn_exit_reason = "interrupted_by_user"
            if not agent.quiet_mode:
                agent._safe_print("\n⚡ Breaking out of tool loop due to interrupt...")
            break
        
        api_call_count += 1
        agent._api_call_count = api_call_count
        agent._touch_activity(f"starting API call #{api_call_count}")

        # Grace call: the budget is exhausted but we gave the model one
        # more chance.  Consume the grace flag so the loop exits after
        # this iteration regardless of outcome.
        if agent._budget_grace_call:
            agent._budget_grace_call = False
        elif not agent.iteration_budget.consume():
            _turn_exit_reason = "budget_exhausted"
            if not agent.quiet_mode:
                agent._safe_print(f"\n⚠️  Iteration budget exhausted ({agent.iteration_budget.used}/{agent.iteration_budget.max_total} iterations used)")
            break

        # Fire step_callback for gateway hooks (agent:step event)
        if agent.step_callback is not None:
            try:
                prev_tools = []
                for _idx, _m in enumerate(reversed(messages)):
                    if _m.get("role") == "assistant" and _m.get("tool_calls"):
                        _fwd_start = len(messages) - _idx
                        _results_by_id = {}
                        for _tm in messages[_fwd_start:]:
                            if _tm.get("role") != "tool":
                                break
                            _tcid = _tm.get("tool_call_id")
                            if _tcid:
                                _results_by_id[_tcid] = _tm.get("content", "")
                        prev_tools = [
                            {
                                "name": tc["function"]["name"],
                                "result": _results_by_id.get(tc.get("id")),
                                "arguments": tc["function"].get("arguments"),
                            }
                            for tc in _m["tool_calls"]
                            if isinstance(tc, dict)
                        ]
                        break
                agent.step_callback(api_call_count, prev_tools)
            except Exception as _step_err:
                logger.debug("step_callback error (iteration %s): %s", api_call_count, _step_err)

        # Track tool-calling iterations for skill nudge.
        # Counter resets whenever skill_manage is actually used.
        if (agent._skill_nudge_interval > 0
                and "skill_manage" in agent.valid_tool_names):
            agent._iters_since_skill += 1
        
        # ── Pre-API-call /steer drain ──────────────────────────────────
        # If a /steer arrived during the previous API call (while the model
        # was thinking), drain it now — before we build api_messages — so
        # the model sees the steer text on THIS iteration.  Without this,
        # steers sent during an API call only land after the NEXT tool batch,
        # which may never come if the model returns a final response.
        #
        # We scan backwards for the last tool-role message in the messages
        # list.  If found, the steer is appended there.  If not (first
        # iteration, no tools yet), the steer stays pending for the next
        # tool batch — injecting into a user message would break role
        # alternation, and there's no tool output to piggyback on.
        _pre_api_steer = agent._drain_pending_steer()
        if _pre_api_steer:
            _injected = False
            for _si in range(len(messages) - 1, -1, -1):
                _sm = messages[_si]
                if isinstance(_sm, dict) and _sm.get("role") == "tool":
                    from agent.prompt_builder import format_steer_marker
                    marker = format_steer_marker(_pre_api_steer)
                    existing = _sm.get("content", "")
                    if isinstance(existing, str):
                        _sm["content"] = existing + marker
                    else:
                        # Multimodal content blocks — append text block
                        try:
                            blocks = list(existing) if existing else []
                            blocks.append({"type": "text", "text": marker})
                            _sm["content"] = blocks
                        except Exception:
                            pass
                    _injected = True
                    logger.debug(
                        "Pre-API-call steer drain: injected into tool msg at index %d",
                        _si,
                    )
                    break
            if not _injected:
                # No tool message to inject into — put it back so
                # the post-tool-execution drain picks it up later.
                _lock = getattr(agent, "_pending_steer_lock", None)
                if _lock is not None:
                    with _lock:
                        if agent._pending_steer:
                            agent._pending_steer = agent._pending_steer + "\n" + _pre_api_steer
                        else:
                            agent._pending_steer = _pre_api_steer
                else:
                    existing = getattr(agent, "_pending_steer", None)
                    agent._pending_steer = (existing + "\n" + _pre_api_steer) if existing else _pre_api_steer

        # Prepare messages for API call
        # If we have an ephemeral system prompt, prepend it to the messages
        # Note: Reasoning is embedded in content via <think> tags for trajectory storage.
        # However, providers like Moonshot AI require a separate 'reasoning_content' field
        # on assistant messages with tool_calls. We handle both cases here.
        request_logger = getattr(agent, "logger", None) or logging.getLogger(__name__)
        repaired_tool_calls = agent._sanitize_tool_call_arguments(
            messages,
            logger=request_logger,
            session_id=agent.session_id,
        )
        if repaired_tool_calls > 0:
            request_logger.info(
                "Sanitized %s corrupted tool_call arguments before request (session=%s)",
                repaired_tool_calls,
                agent.session_id or "-",
            )

        # Defensive: repair malformed role-alternation before API call.
        # Catches cases where the history got wedged into a
        # ``tool → user`` or ``user → user`` tail (e.g. after empty-
        # response scaffolding was stripped and a new user message
        # landed after an orphan tool result). Most providers return
        # empty content on malformed sequences, which would otherwise
        # retrigger the empty-retry loop indefinitely.
        # repair_message_sequence_with_cursor also recomputes the SessionDB
        # flush cursor (_last_flushed_db_idx) when repair compacts the list,
        # so the turn-end flush doesn't skip the assistant/tool chain (#44837).
        from agent.agent_runtime_helpers import repair_message_sequence_with_cursor
        repaired_seq = repair_message_sequence_with_cursor(agent, messages)
        if repaired_seq > 0:
            request_logger.info(
                "Repaired %s message-alternation violations before request (session=%s)",
                repaired_seq,
                agent.session_id or "-",
            )

        api_messages = []
        for idx, msg in enumerate(messages):
            api_msg = msg.copy()

            # Inject ephemeral context into the current turn's user message.
            # Sources: memory manager prefetch + plugin pre_llm_call hooks
            # with target="user_message" (the default).  Both are
            # API-call-time only — the original message in `messages` is
            # never mutated, so nothing leaks into session persistence.
            if idx == current_turn_user_idx and msg.get("role") == "user":
                _injections = []
                if _ext_prefetch_cache:
                    _fenced = build_memory_context_block(_ext_prefetch_cache)
                    if _fenced:
                        _injections.append(_fenced)
                if _plugin_user_context:
                    _injections.append(_plugin_user_context)
                if _injections:
                    _base = api_msg.get("content", "")
                    if isinstance(_base, str):
                        api_msg["content"] = _base + "\n\n" + "\n\n".join(_injections)

            # For ALL assistant messages, pass reasoning back to the API
            # This ensures multi-turn reasoning context is preserved
            agent._copy_reasoning_content_for_api(msg, api_msg)

            # Remove 'reasoning' field - it's for trajectory storage only
            # We've copied it to 'reasoning_content' for the API above
            if "reasoning" in api_msg:
                api_msg.pop("reasoning")
            # Remove finish_reason - not accepted by strict APIs (e.g. Mistral)
            if "finish_reason" in api_msg:
                api_msg.pop("finish_reason")
            # Strip internal thinking-prefill marker
            api_msg.pop("_thinking_prefill", None)
            # Strip Codex Responses API fields (call_id, response_item_id) for
            # strict providers like Mistral, Fireworks, etc. that reject unknown fields.
            # Uses new dicts so the internal messages list retains the fields
            # for Codex Responses compatibility.
            if agent._should_sanitize_tool_calls():
                agent._sanitize_tool_calls_for_strict_api(api_msg, model=agent.model)
            # Keep 'reasoning_details' - OpenRouter uses this for multi-turn reasoning context
            # The signature field helps maintain reasoning continuity
            api_messages.append(api_msg)

        # Build the final system message: cached prompt + ephemeral system prompt.
        # Ephemeral additions are API-call-time only (not persisted to session DB).
        # External recall context is injected into the user message, not the system
        # prompt, so the stable cache prefix remains unchanged.
        #
        # NOTE: Plugin context from pre_llm_call hooks is injected into the
        # user message (see injection block above), NOT the system prompt.
        # This is intentional — system prompt modifications break the prompt
        # cache prefix.  The system prompt is reserved for Hermes internals.
        #
        # Hermes invariant: the system prompt is built ONCE per session
        # (cached on ``_cached_system_prompt``) and replayed verbatim on
        # every turn.  We send it as a single content string so the
        # bytes are byte-stable across turns and upstream prompt caches
        # stay warm.
        effective_system = active_system_prompt or ""
        if agent.ephemeral_system_prompt:
            effective_system = (effective_system + "\n\n" + agent.ephemeral_system_prompt).strip()
        if effective_system:
            api_messages = [{"role": "system", "content": effective_system}] + api_messages

        # Inject ephemeral prefill messages right after the system prompt
        # but before conversation history. Same API-call-time-only pattern.
        if agent.prefill_messages:
            sys_offset = 1 if (api_messages and api_messages[0].get("role") == "system") else 0
            for idx, pfm in enumerate(agent.prefill_messages):
                api_messages.insert(sys_offset + idx, pfm.copy())

        # Apply Anthropic prompt caching for Claude models on native
        # Anthropic, OpenRouter, and third-party Anthropic-compatible
        # gateways. Auto-detected: if ``_use_prompt_caching`` is set,
        # inject cache_control breakpoints (system + last 3 messages)
        # to reduce input token costs by ~75% on multi-turn
        # conversations.
        if agent._use_prompt_caching:
            api_messages = apply_anthropic_cache_control(
                api_messages,
                cache_ttl=agent._cache_ttl,
                native_anthropic=agent._use_native_cache_layout,
            )

        # Safety net: strip orphaned tool results / add stubs for missing
        # results before sending to the API.  Runs unconditionally — not
        # gated on context_compressor — so orphans from session loading or
        # manual message manipulation are always caught.
        api_messages = agent._sanitize_api_messages(api_messages)

        # Drop thinking-only assistant turns (reasoning but no visible
        # output and no tool_calls) and merge any adjacent user messages
        # left behind. Prevents Anthropic 400s ("The final block in an
        # assistant message cannot be `thinking`.") and equivalent errors
        # from third-party Anthropic-compatible gateways that can't replay
        # a thinking-only turn. Runs on the per-call copy only — the
        # stored conversation history keeps the reasoning block for the
        # UI transcript and session persistence.
        api_messages = agent._drop_thinking_only_and_merge_users(
            api_messages,
            drop_codex_reasoning_items=agent.api_mode != "codex_responses",
        )

        # Normalize message whitespace and tool-call JSON for consistent
        # prefix matching.  Ensures bit-perfect prefixes across turns,
        # which enables KV cache reuse on local inference servers
        # (llama.cpp, vLLM, Ollama) and improves cache hit rates for
        # cloud providers.  Operates on api_messages (the API copy) so
        # the original conversation history in `messages` is untouched.
        for am in api_messages:
            if isinstance(am.get("content"), str):
                am["content"] = am["content"].strip()
        for am in api_messages:
            tcs = am.get("tool_calls")
            if not tcs:
                continue
            new_tcs = []
            for tc in tcs:
                if isinstance(tc, dict) and "function" in tc:
                    try:
                        args_obj = json.loads(tc["function"]["arguments"])
                        tc = {**tc, "function": {
                            **tc["function"],
                            "arguments": json.dumps(
                                args_obj, separators=(",", ":"),
                                sort_keys=True,
                            ),
                        }}
                    except Exception:
                        tc["function"]["arguments"] = _repair_tool_call_arguments(
                            tc["function"]["arguments"],
                            tc["function"].get("name", "?"),
                        )
                new_tcs.append(tc)
            am["tool_calls"] = new_tcs

        # Proactively strip any surrogate characters before the API call.
        # Models served via Ollama (Kimi K2.5, GLM-5, Qwen) can return
        # lone surrogates (U+D800-U+DFFF) that crash json.dumps() inside
        # the OpenAI SDK. Sanitizing here prevents the 3-retry cycle.
        _sanitize_messages_surrogates(api_messages)

        # Calculate approximate request size for logging
        total_chars = sum(len(str(msg)) for msg in api_messages)
        approx_tokens = estimate_messages_tokens_rough(api_messages)
        approx_request_tokens = estimate_request_tokens_rough(
            api_messages, tools=agent.tools or None
        )

        _runtime_context_error = _ollama_context_limit_error(
            agent, approx_request_tokens
        )
        if _runtime_context_error:
            final_response = _runtime_context_error
            failed = True
            _turn_exit_reason = "ollama_runtime_context_too_small"
            messages.append({"role": "assistant", "content": final_response})
            agent._emit_status("❌ Ollama runtime context is too small for Hermes tool use")
            api_call_count -= 1
            agent._api_call_count = api_call_count
            try:
                agent.iteration_budget.refund()
            except Exception:
                pass
            break
        
        # Thinking spinner for quiet mode (animated during API call)
        thinking_spinner = None
        
        if not agent.quiet_mode:
            agent._vprint(f"\n{agent.log_prefix}🔄 Making API call #{api_call_count}/{agent.max_iterations}...")
            agent._vprint(f"{agent.log_prefix}   📊 Request size: {len(api_messages)} messages, ~{approx_tokens:,} tokens (~{total_chars:,} chars)")
            agent._vprint(f"{agent.log_prefix}   🔧 Available tools: {len(agent.tools) if agent.tools else 0}")
        else:
            # Animated thinking spinner in quiet mode
            face = random.choice(KawaiiSpinner.get_thinking_faces())
            verb = random.choice(KawaiiSpinner.get_thinking_verbs())
            if agent.thinking_callback:
                # CLI TUI mode: use prompt_toolkit widget instead of raw spinner
                # (works in both streaming and non-streaming modes)
                agent.thinking_callback(f"{face} {verb}...")
            elif not agent._has_stream_consumers() and agent._should_start_quiet_spinner():
                # Raw KawaiiSpinner only when no streaming consumers and the
                # spinner output has a safe sink.
                spinner_type = random.choice(['brain', 'sparkle', 'pulse', 'moon', 'star'])
                thinking_spinner = KawaiiSpinner(f"{face} {verb}...", spinner_type=spinner_type, print_fn=agent._print_fn)
                thinking_spinner.start()
        
        # Log request details if verbose
        if agent.verbose_logging:
            logging.debug(f"API Request - Model: {agent.model}, Messages: {len(messages)}, Tools: {len(agent.tools) if agent.tools else 0}")
            logging.debug(f"Last message role: {messages[-1]['role'] if messages else 'none'}")
            logging.debug(f"Total message size: ~{approx_tokens:,} tokens")
        
        api_start_time = time.time()
        retry_count = 0
        max_retries = agent._api_max_retries
        _retry = TurnRetryState()
        max_compression_attempts = 3

        finish_reason = "stop"
        response = None  # Guard against UnboundLocalError if all retries fail
        api_kwargs = None  # Guard against UnboundLocalError in except handler
        api_request_id = f"{turn_id}:api:{api_call_count}"
        agent._current_api_request_id = api_request_id

        # 010c: the API-call + retry loop now lives in turn_api_call.py.  It
        # returns an ApiCallOutcome carrying loop state back; ``_retry`` is
        # mutated in place (restart_* flags read back below).
        _outcome = run_api_call_with_retry(
            agent,
            messages=messages,
            active_system_prompt=active_system_prompt,
            system_message=system_message,
            conversation_history=conversation_history,
            api_messages=api_messages,
            api_call_count=api_call_count,
            effective_task_id=effective_task_id,
            approx_tokens=approx_tokens,
            total_chars=total_chars,
            original_user_message=original_user_message,
            turn_id=turn_id,
            api_request_id=api_request_id,
            api_start_time=api_start_time,
            _retry=_retry,
            max_retries=max_retries,
            max_compression_attempts=max_compression_attempts,
            retry_count=retry_count,
            length_continue_retries=length_continue_retries,
            compression_attempts=compression_attempts,
            truncated_tool_call_retries=truncated_tool_call_retries,
            truncated_response_parts=truncated_response_parts,
            response=response,
            api_kwargs=api_kwargs,
            finish_reason=finish_reason,
            interrupted=interrupted,
            failed=failed,
            final_response=final_response,
            thinking_spinner=thinking_spinner,
        )
        if _outcome.terminal_result is not None:
            return _outcome.terminal_result
        # Read loop state back into run_conversation locals (the post-loop
        # restart/guard handling below reads these; _retry.restart_* flags are
        # read directly off the mutated _retry).
        response = _outcome.response
        messages = _outcome.messages
        active_system_prompt = _outcome.active_system_prompt
        conversation_history = _outcome.conversation_history
        api_call_count = _outcome.api_call_count
        retry_count = _outcome.retry_count
        length_continue_retries = _outcome.length_continue_retries
        compression_attempts = _outcome.compression_attempts
        truncated_tool_call_retries = _outcome.truncated_tool_call_retries
        finish_reason = _outcome.finish_reason
        api_kwargs = _outcome.api_kwargs
        api_duration = _outcome.api_duration
        final_response = _outcome.final_response
        interrupted = _outcome.interrupted
        failed = _outcome.failed
        
        # If the API call was interrupted, skip response processing
        if interrupted:
            _turn_exit_reason = "interrupted_during_api_call"
            break

        if _retry.restart_with_compressed_messages:
            api_call_count -= 1
            agent.iteration_budget.refund()
            # Count compression restarts toward the retry limit to prevent
            # infinite loops when compression reduces messages but not enough
            # to fit the context window.
            retry_count += 1
            _retry.restart_with_compressed_messages = False
            continue

        if _retry.restart_with_length_continuation:
            # Progressively boost the output token budget on each retry.
            # Retry 1 → 2× base, retry 2 → 3× base, capped at 32 768.
            # Applies to all providers via _ephemeral_max_output_tokens.
            # If the original request already used a larger provider/model
            # default budget, keep that floor so continuation retries do
            # not accidentally downshift to a much smaller cap.
            _boost_base = agent.max_tokens if agent.max_tokens else 4096
            _boost = _boost_base * (length_continue_retries + 1)
            _requested_cap = agent._requested_output_cap_from_api_kwargs(api_kwargs)
            if _requested_cap is not None:
                _boost = max(_boost, _requested_cap)
            _boost_cap = max(32768, _requested_cap or 0)
            agent._ephemeral_max_output_tokens = min(_boost, _boost_cap)
            continue

        # Guard: if all retries exhausted without a successful response
        # (e.g. repeated context-length errors that exhausted retry_count),
        # the `response` variable is still None. Break out cleanly.
        if response is None:
            _turn_exit_reason = "all_retries_exhausted_no_response"
            print(f"{agent.log_prefix}❌ All API retries exhausted with no successful response.")
            agent._persist_session(messages, conversation_history)
            break

        try:
            _transport = agent._get_transport()
            _normalize_kwargs = {}
            if agent.api_mode == "anthropic_messages":
                _normalize_kwargs["strip_tool_prefix"] = agent._is_anthropic_oauth
            normalized = _transport.normalize_response(response, **_normalize_kwargs)
            assistant_message = normalized
            finish_reason = normalized.finish_reason
            
            # Normalize content to string — some OpenAI-compatible servers
            # (llama-server, etc.) return content as a dict or list instead
            # of a plain string, which crashes downstream .strip() calls.
            if assistant_message.content is not None and not isinstance(assistant_message.content, str):
                raw = assistant_message.content
                if isinstance(raw, dict):
                    assistant_message.content = raw.get("text", "") or raw.get("content", "") or json.dumps(raw)
                elif isinstance(raw, list):
                    # Multimodal content list — extract text parts
                    parts = []
                    for part in raw:
                        if isinstance(part, str):
                            parts.append(part)
                        elif isinstance(part, dict) and part.get("type") == "text":
                            parts.append(part.get("text", ""))
                        elif isinstance(part, dict) and "text" in part:
                            parts.append(str(part["text"]))
                    assistant_message.content = "\n".join(parts)
                else:
                    assistant_message.content = str(raw)

            try:
                from hermes_cli.plugins import (
                    has_hook,
                    invoke_hook as _invoke_hook,
                )
                if has_hook("post_api_request"):
                    _assistant_tool_calls = (
                        getattr(assistant_message, "tool_calls", None) or []
                    )
                    _assistant_text = assistant_message.content or ""
                    _api_ended_at = api_start_time + api_duration
                    _invoke_hook(
                        "post_api_request",
                        task_id=effective_task_id,
                        turn_id=turn_id,
                        api_request_id=api_request_id,
                        session_id=agent.session_id or "",
                        platform=agent.platform or "",
                        model=agent.model,
                        provider=agent.provider,
                        base_url=agent.base_url,
                        api_mode=agent.api_mode,
                        api_call_count=api_call_count,
                        api_duration=api_duration,
                        started_at=api_start_time,
                        ended_at=_api_ended_at,
                        finish_reason=finish_reason,
                        message_count=len(api_messages),
                        response_model=getattr(response, "model", None),
                        response=agent._api_response_payload_for_hook(
                            response,
                            assistant_message,
                            finish_reason=finish_reason,
                        ),
                        usage=agent._usage_summary_for_api_request_hook(response),
                        assistant_message=assistant_message,
                        assistant_content_chars=len(_assistant_text),
                        assistant_tool_call_count=len(_assistant_tool_calls),
                    )
            except Exception:
                pass

            # Handle assistant response
            if assistant_message.content and not agent.quiet_mode:
                if agent.verbose_logging:
                    agent._vprint(f"{agent.log_prefix}🤖 Assistant: {assistant_message.content}")
                else:
                    agent._vprint(f"{agent.log_prefix}🤖 Assistant: {assistant_message.content[:100]}{'...' if len(assistant_message.content) > 100 else ''}")

            # Notify progress callback of model's thinking (used by subagent
            # delegation to relay the child's reasoning to the parent display).
            if (assistant_message.content and agent.tool_progress_callback):
                _think_text = assistant_message.content.strip()
                # Strip reasoning XML tags that shouldn't leak to parent display
                _think_text = re.sub(
                    r'</?(?:REASONING_SCRATCHPAD|think|reasoning)>', '', _think_text
                ).strip()
                # For subagents: relay first line to parent display (existing behaviour).
                # For all agents with a structured callback: emit reasoning.available event.
                first_line = _think_text.split('\n')[0][:80] if _think_text else ""
                if first_line and getattr(agent, '_delegate_depth', 0) > 0:
                    try:
                        agent.tool_progress_callback("_thinking", first_line)
                    except Exception:
                        pass
                elif _think_text:
                    try:
                        agent.tool_progress_callback("reasoning.available", "_thinking", _think_text[:500], None)
                    except Exception:
                        pass
            
            # Check for incomplete <REASONING_SCRATCHPAD> (opened but never closed)
            # This means the model ran out of output tokens mid-reasoning — retry up to 2 times
            if has_incomplete_scratchpad(assistant_message.content or ""):
                agent._incomplete_scratchpad_retries += 1
                
                agent._buffer_vprint(f"⚠️  Incomplete <REASONING_SCRATCHPAD> detected (opened but never closed)")
                
                if agent._incomplete_scratchpad_retries <= 2:
                    agent._buffer_vprint(f"🔄 Retrying API call ({agent._incomplete_scratchpad_retries}/2)...")
                    # Don't add the broken message, just retry
                    continue
                else:
                    # Max retries - discard this turn and save as partial
                    agent._flush_status_buffer()
                    agent._vprint(f"{agent.log_prefix}❌ Max retries (2) for incomplete scratchpad. Saving as partial.", force=True)
                    agent._incomplete_scratchpad_retries = 0
                    
                    rolled_back_messages = agent._get_messages_up_to_last_assistant(messages)
                    agent._cleanup_task_resources(effective_task_id)
                    agent._persist_session(messages, conversation_history)
                    
                    return {
                        "final_response": None,
                        "messages": rolled_back_messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "partial": True,
                        "error": "Incomplete REASONING_SCRATCHPAD after 2 retries"
                    }
            
            # Reset incomplete scratchpad counter on clean response
            agent._incomplete_scratchpad_retries = 0

            if agent.api_mode == "codex_responses" and finish_reason == "incomplete":
                agent._codex_incomplete_retries += 1

                interim_msg = agent._build_assistant_message(assistant_message, finish_reason)
                interim_has_content = bool((interim_msg.get("content") or "").strip())
                interim_has_reasoning = bool(interim_msg.get("reasoning", "").strip()) if isinstance(interim_msg.get("reasoning"), str) else False
                interim_has_codex_reasoning = bool(interim_msg.get("codex_reasoning_items"))
                interim_has_codex_message_items = bool(interim_msg.get("codex_message_items"))

                if (
                    interim_has_content
                    or interim_has_reasoning
                    or interim_has_codex_reasoning
                    or interim_has_codex_message_items
                ):
                    last_msg = messages[-1] if messages else None
                    # Duplicate detection: two consecutive incomplete assistant
                    # messages with identical content AND reasoning are collapsed.
                    # For provider-state-only changes (encrypted reasoning
                    # items or replayable message ids/phases/statuses differ
                    # while visible content/reasoning are unchanged), compare
                    # those opaque payloads too so we don't silently drop the
                    # newer continuation state.
                    last_codex_items = last_msg.get("codex_reasoning_items") if isinstance(last_msg, dict) else None
                    interim_codex_items = interim_msg.get("codex_reasoning_items")
                    last_codex_message_items = last_msg.get("codex_message_items") if isinstance(last_msg, dict) else None
                    interim_codex_message_items = interim_msg.get("codex_message_items")
                    duplicate_interim = (
                        isinstance(last_msg, dict)
                        and last_msg.get("role") == "assistant"
                        and last_msg.get("finish_reason") == "incomplete"
                        and (last_msg.get("content") or "") == (interim_msg.get("content") or "")
                        and (last_msg.get("reasoning") or "") == (interim_msg.get("reasoning") or "")
                        and last_codex_items == interim_codex_items
                        and last_codex_message_items == interim_codex_message_items
                    )
                    if not duplicate_interim:
                        messages.append(interim_msg)
                        agent._emit_interim_assistant_message(interim_msg)

                if agent._codex_incomplete_retries < 3:
                    if not agent.quiet_mode:
                        agent._vprint(f"{agent.log_prefix}↻ Codex response incomplete; continuing turn ({agent._codex_incomplete_retries}/3)")
                    agent._session_messages = messages
                    continue

                agent._codex_incomplete_retries = 0
                agent._persist_session(messages, conversation_history)
                return {
                    "final_response": None,
                    "messages": messages,
                    "api_calls": api_call_count,
                    "completed": False,
                    "partial": True,
                    "error": "Codex response remained incomplete after 3 continuation attempts",
                }
            elif hasattr(agent, "_codex_incomplete_retries"):
                agent._codex_incomplete_retries = 0
            
            # Check for tool calls
            if assistant_message.tool_calls:
                if not agent.quiet_mode:
                    agent._vprint(f"{agent.log_prefix}🔧 Processing {len(assistant_message.tool_calls)} tool call(s)...")
                
                if agent.verbose_logging:
                    for tc in assistant_message.tool_calls:
                        logging.debug(f"Tool call: {tc.function.name} with args: {tc.function.arguments[:200]}...")
                
                # Validate tool call names - detect model hallucinations
                # Repair mismatched tool names before validating
                for tc in assistant_message.tool_calls:
                    if tc.function.name not in agent.valid_tool_names:
                        repaired = agent._repair_tool_call(tc.function.name)
                        if repaired:
                            print(f"{agent.log_prefix}🔧 Auto-repaired tool name: '{tc.function.name}' -> '{repaired}'")
                            tc.function.name = repaired
                invalid_tool_calls = [
                    tc.function.name for tc in assistant_message.tool_calls
                    if tc.function.name not in agent.valid_tool_names
                ]
                if invalid_tool_calls:
                    # Track retries for invalid tool calls
                    agent._invalid_tool_retries += 1

                    # Return helpful error to model — model can agent-correct next turn
                    available = ", ".join(sorted(agent.valid_tool_names))
                    invalid_name = invalid_tool_calls[0]
                    invalid_preview = invalid_name[:80] + "..." if len(invalid_name) > 80 else invalid_name
                    agent._buffer_vprint(f"⚠️  Unknown tool '{invalid_preview}' — sending error to model for agent-correction ({agent._invalid_tool_retries}/3)")

                    if agent._invalid_tool_retries >= 3:
                        agent._flush_status_buffer()
                        agent._vprint(f"{agent.log_prefix}❌ Max retries (3) for invalid tool calls exceeded. Stopping as partial.", force=True)
                        agent._invalid_tool_retries = 0
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": f"Model generated invalid tool call: {invalid_preview}"
                        }

                    assistant_msg = agent._build_assistant_message(assistant_message, finish_reason)
                    messages.append(assistant_msg)
                    for tc in assistant_message.tool_calls:
                        _tc_name = tc.function.name
                        if _tc_name not in agent.valid_tool_names:
                            # A blank/whitespace-only name is not a typo the
                            # model can fuzzy-correct toward a real tool — it is
                            # almost always a weak open model echoing tool-call
                            # XML/JSON it saw in file or tool output (#47967:
                            # <tool_call>/<invoke name=...> payloads in a file
                            # prime mimo/nemotron-class models to emit empty
                            # structured calls). Dumping the full tool catalog
                            # in that case feeds the priming loop more names to
                            # mimic and inflates context 3-4x across retries, so
                            # send a terse error that tells the model in-context
                            # tool-call syntax is DATA, not a call to make.
                            if not (_tc_name or "").strip():
                                content = (
                                    "Tool call rejected: the tool name was empty. "
                                    "If tool-call XML or JSON appeared in file "
                                    "contents or tool output, that is data — do "
                                    "not re-emit it as a tool call. To call a "
                                    "tool, use a valid name from your tool list; "
                                    "otherwise reply in plain text."
                                )
                            else:
                                content = f"Tool '{_tc_name}' does not exist. Available tools: {available}"
                        else:
                            content = "Skipped: another tool call in this turn used an invalid name. Please retry this tool call."
                        messages.append({
                            "role": "tool",
                            "name": tc.function.name,
                            "tool_call_id": tc.id,
                            "content": content,
                        })
                    continue
                # Reset retry counter on successful tool call validation
                agent._invalid_tool_retries = 0
                
                # Validate tool call arguments are valid JSON
                # Handle empty strings as empty objects (common model quirk)
                invalid_json_args = []
                for tc in assistant_message.tool_calls:
                    args = tc.function.arguments
                    if isinstance(args, (dict, list)):
                        tc.function.arguments = json.dumps(args)
                        continue
                    if args is not None and not isinstance(args, str):
                        tc.function.arguments = str(args)
                        args = tc.function.arguments
                    # Treat empty/whitespace strings as empty object
                    if not args or not args.strip():
                        tc.function.arguments = "{}"
                        continue
                    try:
                        json.loads(args)
                    except json.JSONDecodeError as e:
                        invalid_json_args.append((tc.function.name, str(e)))
                
                if invalid_json_args:
                    # Check if the invalid JSON is due to truncation rather
                    # than a model formatting mistake.  Routers sometimes
                    # rewrite finish_reason from "length" to "tool_calls",
                    # hiding the truncation from the length handler above.
                    # Detect truncation: args that don't end with } or ]
                    # (after stripping whitespace) are cut off mid-stream.
                    _truncated = any(
                        not (tc.function.arguments or "").rstrip().endswith(("}", "]"))
                        for tc in assistant_message.tool_calls
                        if tc.function.name in {n for n, _ in invalid_json_args}
                    )
                    if _truncated:
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  Truncated tool call arguments detected "
                            f"(finish_reason={finish_reason!r}) — refusing to execute.",
                            force=True,
                        )
                        agent._invalid_json_retries = 0
                        agent._cleanup_task_resources(effective_task_id)
                        agent._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": "Response truncated due to output length limit",
                        }

                    # Track retries for invalid JSON arguments
                    agent._invalid_json_retries += 1

                    tool_name, error_msg = invalid_json_args[0]
                    agent._buffer_vprint(f"⚠️  Invalid JSON in tool call arguments for '{tool_name}': {error_msg}")

                    if agent._invalid_json_retries < 3:
                        agent._buffer_vprint(f"🔄 Retrying API call ({agent._invalid_json_retries}/3)...")
                        # Don't add anything to messages, just retry the API call
                        continue
                    else:
                        # Instead of returning partial, inject tool error results so the model can recover.
                        # Using tool results (not user messages) preserves role alternation.
                        agent._buffer_vprint(f"⚠️  Injecting recovery tool results for invalid JSON...")
                        agent._invalid_json_retries = 0  # Reset for next attempt
                        
                        # Append the assistant message with its (broken) tool_calls
                        recovery_assistant = agent._build_assistant_message(assistant_message, finish_reason)
                        messages.append(recovery_assistant)
                        
                        # Respond with tool error results for each tool call
                        invalid_names = {name for name, _ in invalid_json_args}
                        for tc in assistant_message.tool_calls:
                            if tc.function.name in invalid_names:
                                err = next(e for n, e in invalid_json_args if n == tc.function.name)
                                tool_result = (
                                    f"Error: Invalid JSON arguments. {err}. "
                                    f"For tools with no required parameters, use an empty object: {{}}. "
                                    f"Please retry with valid JSON."
                                )
                            else:
                                tool_result = "Skipped: other tool call in this response had invalid JSON."
                            messages.append({
                                "role": "tool",
                                "name": tc.function.name,
                                "tool_call_id": tc.id,
                                "content": tool_result,
                            })
                        continue
                
                # Reset retry counter on successful JSON validation
                agent._invalid_json_retries = 0

                # ── Post-call guardrails ──────────────────────────
                assistant_message.tool_calls = agent._cap_delegate_task_calls(
                    assistant_message.tool_calls
                )
                assistant_message.tool_calls = agent._deduplicate_tool_calls(
                    assistant_message.tool_calls
                )

                assistant_msg = agent._build_assistant_message(assistant_message, finish_reason)
                
                # If this turn has both content AND tool_calls, capture the content
                # as a fallback final response. Common pattern: model delivers its
                # answer and calls memory/skill tools as a side-effect in the same
                # turn. If the follow-up turn after tools is empty, we use this.
                turn_content = assistant_message.content or ""
                if turn_content and agent._has_content_after_think_block(turn_content):
                    agent._last_content_with_tools = turn_content
                    # Only mute subsequent output when EVERY tool call in
                    # this turn is post-response housekeeping (memory, todo,
                    # skill_manage, etc.).  If any substantive tool is present
                    # (search_files, read_file, write_file, terminal, ...),
                    # keep output visible so the user sees progress.
                    _HOUSEKEEPING_TOOLS = frozenset({
                        "memory", "todo", "skill_manage", "session_search",
                    })
                    _all_housekeeping = all(
                        tc.function.name in _HOUSEKEEPING_TOOLS
                        for tc in assistant_message.tool_calls
                    )
                    agent._last_content_tools_all_housekeeping = _all_housekeeping
                    if _all_housekeeping and agent._has_stream_consumers():
                        agent._mute_post_response = True
                    elif agent._should_emit_quiet_tool_messages():
                        clean = agent._strip_think_blocks(turn_content).strip()
                        if clean:
                            agent._vprint(f"  ┊ 💬 {clean}")
                
                # Pop thinking-only prefill message(s) before appending
                # (tool-call path — same rationale as the final-response path).
                _had_prefill = False
                while (
                    messages
                    and isinstance(messages[-1], dict)
                    and messages[-1].get("_thinking_prefill")
                ):
                    messages.pop()
                    _had_prefill = True

                # Reset prefill counter when tool calls follow a prefill
                # recovery.  Without this, the counter accumulates across
                # the whole conversation — a model that intermittently
                # empties (empty → prefill → tools → empty → prefill →
                # tools) burns both prefill attempts and the third empty
                # gets zero recovery.  Resetting here treats each tool-
                # call success as a fresh start.
                if _had_prefill:
                    agent._thinking_prefill_retries = 0
                    agent._empty_content_retries = 0
                # Successful tool execution — reset the post-tool nudge
                # flag so it can fire again if the model goes empty on
                # a LATER tool round.
                agent._post_tool_empty_retried = False

                messages.append(assistant_msg)
                agent._emit_interim_assistant_message(assistant_msg)

                # Close any open streaming display (response box, reasoning
                # box) before tool execution begins.  Intermediate turns may
                # have streamed early content that opened the response box;
                # flushing here prevents it from wrapping tool feed lines.
                # Only signal the display callback — TTS (_stream_callback)
                # should NOT receive None (it uses None as end-of-stream).
                if agent.stream_delta_callback:
                    try:
                        agent.stream_delta_callback(None)
                    except Exception:
                        pass

                agent._execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count)

                if agent._tool_guardrail_halt_decision is not None:
                    decision = agent._tool_guardrail_halt_decision
                    _turn_exit_reason = "guardrail_halt"
                    final_response = agent._toolguard_controlled_halt_response(decision)
                    agent._emit_status(
                        f"⚠️ Tool guardrail halted {decision.tool_name}: {decision.code}"
                    )
                    messages.append({"role": "assistant", "content": final_response})
                    # Emit the halt message to the client so it's not
                    # indistinguishable from a crash.  The stream display
                    # was flushed (callback(None)) before tool execution,
                    # but the callback is still alive — fire the text
                    # through it so SSE/TUI clients see the explanation.
                    if final_response:
                        agent._safe_print(f"\n{final_response}\n")
                        if agent.stream_delta_callback:
                            try:
                                agent.stream_delta_callback(final_response)
                                agent.stream_delta_callback(None)
                            except Exception:
                                pass
                    break

                # Reset per-turn retry counters after successful tool
                # execution so a single truncation doesn't poison the
                # entire conversation.
                truncated_tool_call_retries = 0

                # Signal that a paragraph break is needed before the next
                # streamed text.  We don't emit it immediately because
                # multiple consecutive tool iterations would stack up
                # redundant blank lines.  Instead, _fire_stream_delta()
                # will prepend a single "\n\n" the next time real text
                # arrives.
                agent._stream_needs_break = True

                # Refund the iteration if the ONLY tool(s) called were
                # execute_code (programmatic tool calling).  These are
                # cheap RPC-style calls that shouldn't eat the budget.
                _tc_names = {tc.function.name for tc in assistant_message.tool_calls}
                if _tc_names == {"execute_code"}:
                    agent.iteration_budget.refund()
                
                # Use real token counts from the API response to decide
                # compression.  prompt_tokens + completion_tokens is the
                # actual context size the provider reported plus the
                # assistant turn — a tight lower bound for the next prompt.
                # Tool results appended above aren't counted yet, but the
                # threshold (default 50%) leaves ample headroom; if tool
                # results push past it, the next API call will report the
                # real total and trigger compression then.
                #
                # If last_prompt_tokens is 0 (stale after API disconnect
                # or provider returned no usage data), fall back to rough
                # estimate to avoid missing compression.  Without this,
                # a session can grow unbounded after disconnects because
                # should_compress(0) never fires.  (#2153)
                _compressor = agent.context_compressor
                if _compressor.last_prompt_tokens > 0:
                    # Only use prompt_tokens — completion/reasoning
                    # tokens don't consume context window space.
                    # Thinking models (GLM-5.1, QwQ, DeepSeek R1)
                    # inflate completion_tokens with reasoning,
                    # causing premature compression.  (#12026)
                    _real_tokens = _compressor.last_prompt_tokens
                elif _compressor.last_prompt_tokens == -1:
                    # Compression just ran and no API-reported prompt count
                    # has arrived yet. Avoid treating a schema-heavy rough
                    # post-compression estimate as real context pressure.
                    _real_tokens = 0
                else:
                    # Include tool schemas — with 50+ tools enabled
                    # these add 20-30K tokens the messages-only
                    # estimate misses, which can skip compression
                    # past the configured threshold (#14695).
                    _real_tokens = estimate_request_tokens_rough(
                        messages, tools=agent.tools or None
                    )

                if agent.compression_enabled and _compressor.should_compress(_real_tokens):
                    agent._safe_print("  ⟳ compacting context…")
                    messages, active_system_prompt, conversation_history = apply_turn_compression(
                        agent, messages, system_message,
                        approx_tokens=agent.context_compressor.last_prompt_tokens,
                        task_id=effective_task_id,
                    )
                
                # Save session log incrementally (so progress is visible even if interrupted)
                agent._session_messages = messages
                
                # Continue loop for next response
                continue
            
            else:
                # No tool calls - this is the final response
                final_response = assistant_message.content or ""
                
                # Fix: unmute output when entering the no-tool-call branch
                # so the user can see empty-response warnings and recovery
                # status messages.  _mute_post_response was set during a
                # prior housekeeping tool turn and should not silence the
                # final response path.
                agent._mute_post_response = False
                
                # Check if response only has think block with no actual content after it
                if not agent._has_content_after_think_block(final_response):
                    # ── Partial stream recovery ─────────────────────
                    # If content was already streamed to the user before
                    # the connection died, use it as the final response
                    # instead of falling through to prior-turn fallback
                    # or wasting API calls on retries.
                    _partial_streamed = (
                        getattr(agent, "_current_streamed_assistant_text", "") or ""
                    )
                    if agent._has_content_after_think_block(_partial_streamed):
                        _turn_exit_reason = "partial_stream_recovery"
                        _recovered = agent._strip_think_blocks(_partial_streamed).strip()
                        logger.info(
                            "Partial stream content delivered (%d chars) "
                            "— using as final response",
                            len(_recovered),
                        )
                        agent._emit_status(
                            "↻ Stream interrupted — using delivered content "
                            "as final response"
                        )
                        final_response = _recovered
                        agent._response_was_previewed = True
                        break

                    # If the previous turn already delivered real content alongside
                    # HOUSEKEEPING tool calls (e.g. "You're welcome!" + memory save),
                    # the model has nothing more to say. Use the earlier content
                    # immediately instead of wasting API calls on retries.
                    # NOTE: Only use this shortcut when ALL tools in that turn were
                    # housekeeping (memory, todo, etc.).  When substantive tools
                    # were called (terminal, search_files, etc.), the content was
                    # likely mid-task narration ("I'll scan the directory...") and
                    # the empty follow-up means the model choked — let the
                    # post-tool nudge below handle that instead of exiting early.
                    fallback = getattr(agent, '_last_content_with_tools', None)
                    if fallback and getattr(agent, '_last_content_tools_all_housekeeping', False):
                        _turn_exit_reason = "fallback_prior_turn_content"
                        logger.info("Empty follow-up after tool calls — using prior turn content as final response")
                        agent._emit_status("↻ Empty response after tool calls — using earlier content as final answer")
                        agent._last_content_with_tools = None
                        agent._last_content_tools_all_housekeeping = False
                        agent._empty_content_retries = 0
                        # Do NOT modify the assistant message content — the
                        # old code injected "Calling the X tools..." which
                        # poisoned the conversation history.  Just use the
                        # fallback text as the final response and break.
                        final_response = agent._strip_think_blocks(fallback).strip()
                        agent._response_was_previewed = True
                        break

                    # ── Post-tool-call empty response nudge ───────────
                    # The model returned empty after executing tool calls.
                    # This covers two cases:
                    #  (a) No prior-turn content at all — model went silent
                    #  (b) Prior turn had content + SUBSTANTIVE tools (the
                    #      fallback above was skipped because the content
                    #      was mid-task narration, not a final answer)
                    # Instead of giving up, nudge the model to continue by
                    # appending a user-level hint.  This is the #9400 case:
                    # weaker models (mimo-v2-pro, GLM-5, etc.) sometimes
                    # return empty after tool results instead of continuing
                    # to the next step.  One retry with a nudge usually
                    # fixes it.
                    _prior_was_tool = any(
                        m.get("role") == "tool"
                        for m in messages[-5:]  # check recent messages
                    )
                    # Detect Qwen3/Ollama-style in-content thinking blocks.
                    # Ollama puts <think> in the content field (not in
                    # reasoning_content), so _has_structured below would
                    # miss it.  We check here so thinking-only responses
                    # after tool calls route to prefill instead of nudge.
                    _has_inline_thinking = bool(
                        re.search(
                            r'<think>|<thinking>|<reasoning>',
                            final_response or "",
                            re.IGNORECASE,
                        )
                    )
                    if (
                        _prior_was_tool
                        and not getattr(agent, "_post_tool_empty_retried", False)
                        and not _has_inline_thinking  # thinking model still working — let prefill handle
                    ):
                        agent._post_tool_empty_retried = True
                        # Clear stale narration so it doesn't resurface
                        # on a later empty response after the nudge.
                        agent._last_content_with_tools = None
                        agent._last_content_tools_all_housekeeping = False
                        logger.info(
                            "Empty response after tool calls — nudging model "
                            "to continue processing"
                        )
                        agent._buffer_status(
                            "⚠️ Model returned empty after tool calls — "
                            "nudging to continue"
                        )
                        # Append the empty assistant message first so the
                        # message sequence stays valid:
                        #   tool(result) → assistant("(empty)") → user(nudge)
                        # Without this, we'd have tool → user which most
                        # APIs reject as an invalid sequence.
                        _nudge_msg = agent._build_assistant_message(assistant_message, finish_reason)
                        _nudge_msg["content"] = "(empty)"
                        _nudge_msg["_empty_recovery_synthetic"] = True
                        messages.append(_nudge_msg)
                        messages.append({
                            "role": "user",
                            "content": (
                                "You just executed tool calls but returned an "
                                "empty response. Please process the tool "
                                "results above and continue with the task."
                            ),
                            "_empty_recovery_synthetic": True,
                        })
                        continue

                    # ── Thinking-only prefill continuation ──────────
                    # The model produced structured reasoning (via API
                    # fields) but no visible text content.  Rather than
                    # giving up, append the assistant message as-is and
                    # continue — the model will see its own reasoning
                    # on the next turn and produce the text portion.
                    # Inspired by clawdbot's "incomplete-text" recovery.
                    # Also covers Qwen3/Ollama in-content <think> blocks
                    # (detected above as _has_inline_thinking).
                    _has_structured = bool(
                        getattr(assistant_message, "reasoning", None)
                        or getattr(assistant_message, "reasoning_content", None)
                        or getattr(assistant_message, "reasoning_details", None)
                        or _has_inline_thinking
                    )
                    if _has_structured and agent._thinking_prefill_retries < 2:
                        agent._thinking_prefill_retries += 1
                        logger.info(
                            "Thinking-only response (no visible content) — "
                            "prefilling to continue (%d/2)",
                            agent._thinking_prefill_retries,
                        )
                        agent._buffer_status(
                            f"↻ Thinking-only response — prefilling to continue "
                            f"({agent._thinking_prefill_retries}/2)"
                        )
                        interim_msg = agent._build_assistant_message(
                            assistant_message, "incomplete"
                        )
                        interim_msg["_thinking_prefill"] = True
                        messages.append(interim_msg)
                        agent._session_messages = messages
                        continue

                    # ── Empty response retry ──────────────────────
                    # Model returned nothing usable.  Retry up to 3
                    # times before attempting fallback.  This covers
                    # both truly empty responses (no content, no
                    # reasoning) AND reasoning-only responses after
                    # prefill exhaustion — models like mimo-v2-pro
                    # always populate reasoning fields via OpenRouter,
                    # so the old `not _has_structured` guard blocked
                    # retries for every reasoning model after prefill.
                    _truly_empty = not agent._strip_think_blocks(
                        final_response
                    ).strip()
                    _prefill_exhausted = (
                        _has_structured
                        and agent._thinking_prefill_retries >= 2
                    )
                    if _truly_empty and (not _has_structured or _prefill_exhausted) and agent._empty_content_retries < 3:
                        agent._empty_content_retries += 1
                        logger.warning(
                            "Empty response (no content or reasoning) — "
                            "retry %d/3 (model=%s)",
                            agent._empty_content_retries, agent.model,
                        )
                        agent._buffer_status(
                            f"⚠️ Empty response from model — retrying "
                            f"({agent._empty_content_retries}/3)"
                        )
                        continue

                    # ── Exhausted retries — try fallback provider ──
                    # Before giving up with "(empty)", attempt to
                    # switch to the next provider in the fallback
                    # chain.  This covers the case where a model
                    # (e.g. GLM-4.5-Air) consistently returns empty
                    # due to context degradation or provider issues.
                    if _truly_empty and agent._fallback_chain:
                        logger.warning(
                            "Empty response after %d retries — "
                            "attempting fallback (model=%s, provider=%s)",
                            agent._empty_content_retries, agent.model,
                            agent.provider,
                        )
                        agent._buffer_status(
                            "⚠️ Model returning empty responses — "
                            "switching to fallback provider..."
                        )
                        if agent._try_activate_fallback():
                            agent._empty_content_retries = 0
                            agent._buffer_status(
                                f"↻ Switched to fallback: {agent.model} "
                                f"({agent.provider})"
                            )
                            logger.info(
                                "Fallback activated after empty responses: "
                                "now using %s on %s",
                                agent.model, agent.provider,
                            )
                            continue

                    # Exhausted retries and fallback chain (or no
                    # fallback configured).  Fall through to the
                    # "(empty)" terminal.
                    # Surface the buffered retry/fallback trace so the
                    # user can see what was attempted before "(empty)".
                    agent._flush_status_buffer()
                    _turn_exit_reason = "empty_response_exhausted"
                    reasoning_text = agent._extract_reasoning(assistant_message)
                    agent._drop_trailing_empty_response_scaffolding(messages)
                    assistant_msg = agent._build_assistant_message(assistant_message, finish_reason)
                    assistant_msg["content"] = "(empty)"
                    # This is a user-facing failure sentinel for the gateway,
                    # not real assistant content. Persisting it makes later
                    # "continue" turns replay assistant("(empty)") as if it
                    # were a meaningful model response, which can keep long
                    # tool-heavy sessions stuck in empty-response loops.
                    assistant_msg["_empty_terminal_sentinel"] = True
                    messages.append(assistant_msg)

                    if reasoning_text:
                        reasoning_preview = reasoning_text[:500] + "..." if len(reasoning_text) > 500 else reasoning_text
                        logger.warning(
                            "Reasoning-only response (no visible content) "
                            "after exhausting retries and fallback. "
                            "Reasoning: %s", reasoning_preview,
                        )
                        agent._emit_status(
                            "⚠️ Model produced reasoning but no visible "
                            "response after all retries. Returning empty."
                        )
                    else:
                        logger.warning(
                            "Empty response (no content or reasoning) "
                            "after %d retries. No fallback available. "
                            "model=%s provider=%s",
                            agent._empty_content_retries, agent.model,
                            agent.provider,
                        )
                        agent._emit_status(
                            "❌ Model returned no content after all retries"
                            + (" and fallback attempts." if agent._fallback_chain else
                               ". No fallback providers configured.")
                        )

                    final_response = "(empty)"
                    break
                
                # Reset retry counter/signature on successful content
                agent._empty_content_retries = 0
                agent._thinking_prefill_retries = 0
                # Successful content reached — drop any buffered retry
                # status from earlier failed attempts in this turn.
                agent._clear_status_buffer()

                if (
                    agent.api_mode == "codex_responses"
                    and agent.valid_tool_names
                    and codex_ack_continuations < 2
                    and agent._looks_like_codex_intermediate_ack(
                        user_message=user_message,
                        assistant_content=final_response,
                        messages=messages,
                    )
                ):
                    codex_ack_continuations += 1
                    interim_msg = agent._build_assistant_message(assistant_message, "incomplete")
                    messages.append(interim_msg)
                    agent._emit_interim_assistant_message(interim_msg)

                    continue_msg = {
                        "role": "user",
                        "content": (
                            "[System: Continue now. Execute the required tool calls and only "
                            "send your final answer after completing the task.]"
                        ),
                    }
                    messages.append(continue_msg)
                    agent._session_messages = messages
                    continue

                codex_ack_continuations = 0

                if truncated_response_parts:
                    final_response = "".join(truncated_response_parts) + final_response
                    truncated_response_parts = []
                    length_continue_retries = 0
                
                final_response = agent._strip_think_blocks(final_response).strip()
                
                final_msg = agent._build_assistant_message(assistant_message, finish_reason)

                # Pop thinking-only prefill and empty-response retry
                # scaffolding before appending the final response.  These
                # internal turns are only for the next API retry and should
                # not become durable transcript context.
                while (
                    messages
                    and isinstance(messages[-1], dict)
                    and (
                        messages[-1].get("_thinking_prefill")
                        or messages[-1].get("_empty_recovery_synthetic")
                        or messages[-1].get("_empty_terminal_sentinel")
                    )
                ):
                    messages.pop()

                messages.append(final_msg)
                
                _turn_exit_reason = f"text_response(finish_reason={finish_reason})"
                if not agent.quiet_mode:
                    agent._safe_print(f"🎉 Conversation completed after {api_call_count} OpenAI-compatible API call(s)")
                break
            
        except Exception as e:
            error_msg = f"Error during OpenAI-compatible API call #{api_call_count}: {str(e)}"
            try:
                print(f"❌ {error_msg}")
            except (OSError, ValueError):
                logger.error(error_msg)

            # Emit the full traceback at ERROR level so it lands in both
            # agent.log AND errors.log.  Previously this was logged at DEBUG,
            # which meant intermittent outer-loop failures were unreproducible
            # — users would see a one-line summary on screen with no way to
            # recover the call site.  logger.exception() includes the
            # traceback automatically and emits at ERROR.
            logger.exception("Outer loop error in API call #%d", api_call_count)
            
            # If an assistant message with tool_calls was already appended,
            # the API expects a role="tool" result for every tool_call_id.
            # Fill in error results for any that weren't answered yet.
            for idx in range(len(messages) - 1, -1, -1):
                msg = messages[idx]
                if not isinstance(msg, dict):
                    break
                if msg.get("role") == "tool":
                    continue
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    answered_ids = {
                        m["tool_call_id"]
                        for m in messages[idx + 1:]
                        if isinstance(m, dict) and m.get("role") == "tool"
                    }
                    for tc in msg["tool_calls"]:
                        if not tc or not isinstance(tc, dict): continue
                        if tc["id"] not in answered_ids:
                            err_msg = {
                                "role": "tool",
                                "name": _ra().AIAgent._get_tool_call_name_static(tc),
                                "tool_call_id": tc["id"],
                                "content": f"Error executing tool: {error_msg}",
                            }
                            messages.append(err_msg)
                break
            
            # Non-tool errors don't need a synthetic message injected.
            # The error is already printed to the user (line above), and
            # the retry loop continues.  Injecting a fake user/assistant
            # message pollutes history, burns tokens, and risks violating
            # role-alternation invariants.

            # If we're near the limit, break to avoid infinite loops
            if api_call_count >= agent.max_iterations - 1:
                _turn_exit_reason = f"error_near_max_iterations({error_msg[:80]})"
                final_response = f"I apologize, but I encountered repeated errors: {error_msg}"
                # Append as assistant so the history stays valid for
                # session resume (avoids consecutive user messages).
                messages.append({"role": "assistant", "content": final_response})
                break
    
    # Post-loop turn finalization extracted to agent/turn_finalizer.finalize_turn
    # (god-file decomposition Phase 1 step 4). Behavior-neutral: the assembled
    # result dict is returned exactly as before.
    from agent.turn_finalizer import finalize_turn
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=api_call_count,
        interrupted=interrupted,
        failed=failed,
        messages=messages,
        conversation_history=conversation_history,
        effective_task_id=effective_task_id,
        turn_id=turn_id,
        user_message=user_message,
        original_user_message=original_user_message,
        _should_review_memory=_should_review_memory,
        _turn_exit_reason=_turn_exit_reason,
    )



__all__ = ["run_conversation"]
