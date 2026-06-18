"""The API-call + retry loop, extracted from ``conversation_loop.run_conversation``.

Plan 010c (stacked on 010a compression seam + 010b retry char tests):
:func:`run_api_call_with_retry` is the verbatim ``while retry_count <
max_retries`` loop that drives ONE model API call through retries, fallbacks,
compression, length-continuation and content-policy handling for a single
outer iteration of the conversation loop.

Signal protocol — :class:`ApiCallOutcome`:
  * Every in-loop ``return <result-dict>`` / ``_content_policy_blocked_result``
    that used to return from ``run_conversation`` now returns
    ``ApiCallOutcome(terminal_result=<that dict>, ...)``; the caller does
    ``if outcome.terminal_result is not None: return outcome.terminal_result``.
  * The success ``break`` and the natural while-exit fall through to
    ``ApiCallOutcome(terminal_result=None, response=response, ...)``.
  * ``_retry`` (:class:`TurnRetryState`) is MUTABLE and passed in — the loop
    mutates it and the caller reads ``restart_*`` flags back directly off it
    (they are NOT carried in the outcome).

Behaviour is preserved byte-for-byte: the loop body is a verbatim copy
(dedented 4 spaces) of the original inline loop, with ONLY the function-level
``return``s wrapped in ``_terminal(...)``.  ``break``/``continue`` target this
loop and are unchanged; the ``_perform_api_call`` / ``_stop_spinner`` closures
move wholesale (their ``return`` / ``nonlocal`` are unchanged).

The conversation_loop-internal helpers (``_content_policy_blocked_result``,
``_get_continuation_prompt`` …) and the ``apply_turn_compression`` seam are
resolved through a function-level ``from agent import conversation_loop`` to
avoid the import cycle (conversation_loop imports this module).  Binding
``apply_turn_compression = conversation_loop.apply_turn_compression`` at call
time keeps 010b's spy — which patches
``agent.conversation_loop.apply_turn_compression`` — intercepting the in-loop
compression calls.
"""

from __future__ import annotations

import json
import logging
import re
import ssl
import time
from dataclasses import dataclass
from typing import Any, List, Optional

from agent.error_classifier import FailoverReason, classify_api_error
from agent.message_sanitization import (
    _sanitize_messages_non_ascii,
    _sanitize_messages_surrogates,
    _sanitize_structure_non_ascii,
    _sanitize_structure_surrogates,
    _sanitize_tools_non_ascii,
    _strip_images_from_messages,
    _strip_non_ascii,
)
from agent.model_metadata import (
    get_context_length_from_provider_error,
    parse_available_output_tokens_from_error,
    save_context_length,
)
from agent.retry_utils import jittered_backoff
from agent.usage_pricing import estimate_usage_cost, normalize_usage
from hermes_constants import PARTIAL_STREAM_STUB_ID
from utils import base_url_host_matches, env_var_enabled

logger = logging.getLogger(__name__)


@dataclass
class ApiCallOutcome:
    """Result of one ``run_api_call_with_retry`` invocation (one outer iteration).

    ``terminal_result`` not None → ``run_conversation`` returns it immediately.
    Otherwise the API call produced ``response`` (or the caller's post-loop
    restart/guard handling takes over via the mutated ``_retry`` flags).
    """

    terminal_result: Optional[dict]
    response: Any
    messages: list
    active_system_prompt: Any
    conversation_history: Any
    api_call_count: int
    retry_count: int
    length_continue_retries: int
    compression_attempts: int
    truncated_tool_call_retries: int
    finish_reason: Any
    api_kwargs: Any
    api_duration: Any
    final_response: Any
    interrupted: bool
    failed: bool


def run_api_call_with_retry(
    agent,
    *,
    messages,
    active_system_prompt,
    system_message,
    conversation_history,
    api_messages,
    api_call_count,
    effective_task_id,
    approx_tokens,
    total_chars,
    original_user_message,
    turn_id,
    api_request_id,
    api_start_time,
    _retry,
    max_retries,
    max_compression_attempts,
    retry_count,
    length_continue_retries,
    compression_attempts,
    truncated_tool_call_retries,
    truncated_response_parts,
    response,
    api_kwargs,
    finish_reason,
    interrupted,
    failed,
    final_response,
    thinking_spinner,
):
    """Run the API-call retry loop for one outer conversation iteration.

    See module docstring for the signal protocol.  ``_retry`` is mutated in
    place; ``messages`` / ``api_messages`` / ``truncated_response_parts`` are
    mutated in place as in the original loop.
    """
    # Resolve conversation_loop-internal helpers + the compression seam lazily
    # to avoid the import cycle (conversation_loop imports this module).  Re-
    # binding the seam here keeps 010b's spy (which patches
    # ``agent.conversation_loop.apply_turn_compression``) intercepting.
    from agent import conversation_loop

    _content_policy_blocked_result = conversation_loop._content_policy_blocked_result
    _get_continuation_prompt = conversation_loop._get_continuation_prompt
    _image_error_max_dimension = conversation_loop._image_error_max_dimension
    _is_nous_inference_route = conversation_loop._is_nous_inference_route
    _print_billing_or_entitlement_guidance = conversation_loop._print_billing_or_entitlement_guidance
    _print_nous_entitlement_guidance = conversation_loop._print_nous_entitlement_guidance
    _billing_or_entitlement_message = conversation_loop._billing_or_entitlement_message
    _try_refresh_nous_paid_entitlement_credentials = conversation_loop._try_refresh_nous_paid_entitlement_credentials
    _ra = conversation_loop._ra
    sanitize_active_system_prompt = conversation_loop.sanitize_active_system_prompt
    apply_turn_compression = conversation_loop.apply_turn_compression
    _CONTENT_POLICY_RECOVERY_HINT = conversation_loop._CONTENT_POLICY_RECOVERY_HINT
    INTERRUPT_WAITING_FOR_MODEL_PREFIX = conversation_loop.INTERRUPT_WAITING_FOR_MODEL_PREFIX

    def _terminal(result_dict):
        """Wrap a run_conversation result dict as a terminal ApiCallOutcome,
        snapshotting current loop state (read as free vars at call time)."""
        return ApiCallOutcome(
            terminal_result=result_dict,
            response=response,
            messages=messages,
            active_system_prompt=active_system_prompt,
            conversation_history=conversation_history,
            api_call_count=api_call_count,
            retry_count=retry_count,
            length_continue_retries=length_continue_retries,
            compression_attempts=compression_attempts,
            truncated_tool_call_retries=truncated_tool_call_retries,
            finish_reason=finish_reason,
            api_kwargs=api_kwargs,
            api_duration=api_duration,
            final_response=final_response,
            interrupted=interrupted,
            failed=failed,
        )

    # ``api_duration`` is set on the success path inside the loop (and read by
    # the caller after a successful response).  Initialise so ``_terminal`` and
    # the fall-through return always have a value.
    api_duration = None

    while retry_count < max_retries:
        # ── Nous Portal rate limit guard ──────────────────────
        # If another session already recorded that Nous is rate-
        # limited, skip the API call entirely.  Each attempt
        # (including SDK-level retries) counts against RPH and
        # deepens the rate limit hole.
        if agent.provider == "nous":
            try:
                from agent.nous_rate_guard import (
                    nous_rate_limit_remaining,
                    format_remaining as _fmt_nous_remaining,
                )
                _nous_remaining = nous_rate_limit_remaining()
                if _nous_remaining is not None and _nous_remaining > 0:
                    _nous_msg = (
                        f"Nous Portal rate limit active — "
                        f"resets in {_fmt_nous_remaining(_nous_remaining)}."
                    )
                    agent._buffer_vprint(
                        f"⏳ {_nous_msg} Trying fallback..."
                    )
                    agent._buffer_status(f"⏳ {_nous_msg}")
                    if agent._try_activate_fallback():
                        retry_count = 0
                        compression_attempts = 0
                        _retry.primary_recovery_attempted = False
                        continue
                    # No fallback available — surface buffered context
                    # so user sees the rate-limit message that led here.
                    agent._flush_status_buffer()
                    agent._persist_session(messages, conversation_history)
                    return _terminal({
                        "final_response": (
                            f"⏳ {_nous_msg}\n\n"
                            "No fallback provider available. "
                            "Try again after the reset, or add a "
                            "fallback provider in config.yaml."
                        ),
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "failed": True,
                        "error": _nous_msg,
                    })
            except ImportError:
                pass
            except Exception:
                pass  # Never let rate guard break the agent loop

        try:
            agent._reset_stream_delivery_tracking()
            # api_messages is built once, before this retry loop, while the
            # primary provider is active.  A mid-conversation fallback can
            # switch to a require-side provider (DeepSeek / Kimi / MiMo) that
            # rejects assistant turns lacking reasoning_content.  Re-apply the
            # echo-back pad for the *current* provider here (idempotent no-op
            # unless the active provider needs it) so the fallback request
            # isn't sent with stale, primary-shaped reasoning fields.
            agent._reapply_reasoning_echo_for_provider(api_messages)
            api_kwargs = agent._build_api_kwargs(api_messages)
            if agent._force_ascii_payload:
                _sanitize_structure_non_ascii(api_kwargs)
            if agent.api_mode == "codex_responses":
                api_kwargs = agent._get_transport().preflight_kwargs(api_kwargs, allow_stream=False)
            try:
                from hermes_cli.middleware import apply_llm_request_middleware

                _llm_request_mw = apply_llm_request_middleware(
                    api_kwargs,
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
                )
                api_kwargs = _llm_request_mw.payload
                _original_api_kwargs = _llm_request_mw.original_payload
                _llm_middleware_trace = _llm_request_mw.trace
            except Exception:
                _original_api_kwargs = dict(api_kwargs)
                _llm_middleware_trace = []

            try:
                from hermes_cli.plugins import (
                    has_hook,
                    invoke_hook as _invoke_hook,
                )
                if has_hook("pre_api_request"):
                    request_messages = api_kwargs.get("messages")
                    if not isinstance(request_messages, list):
                        request_messages = api_kwargs.get("input")
                    if not isinstance(request_messages, list):
                        request_messages = api_messages
                    # Shallow-copy the outer list so plugins that retain the
                    # reference for async snapshotting don't observe later
                    # mutations of api_messages.  The inner dicts are not
                    # mutated by the agent loop, so a shallow copy is
                    # sufficient; a deepcopy would walk every tool result
                    # and base64 image on every API call.
                    #
                    # The ``request_messages`` and ``conversation_history``
                    # kwargs below are pre-existing raw passthroughs
                    # consumed by the bundled langfuse plugin
                    # (``plugins/observability/langfuse/__init__.py:_coerce_request_messages``).
                    # They predate ``request`` and are intentionally NOT
                    # sanitised — secrets are not expected here because
                    # ``api_kwargs`` is the same object passed to the
                    # provider client.  New consumers should read the
                    # sanitised view from ``request["body"]["messages"]``.
                    _request_payload = agent._api_request_payload_for_hook(api_kwargs)
                    _invoke_hook(
                        "pre_api_request",
                        task_id=effective_task_id,
                        turn_id=turn_id,
                        api_request_id=api_request_id,
                        session_id=agent.session_id or "",
                        user_message=original_user_message,
                        conversation_history=list(messages),
                        platform=agent.platform or "",
                        model=agent.model,
                        provider=agent.provider,
                        base_url=agent.base_url,
                        api_mode=agent.api_mode,
                        api_call_count=api_call_count,
                        request_messages=list(request_messages)
                        if isinstance(request_messages, list)
                        else [],
                        message_count=len(api_messages),
                        tool_count=len(agent.tools or []),
                        approx_input_tokens=approx_tokens,
                        request_char_count=total_chars,
                        max_tokens=agent.max_tokens,
                        started_at=api_start_time,
                        middleware_trace=list(_llm_middleware_trace),
                        request=_request_payload,
                    )
            except Exception:
                pass

            if env_var_enabled("HERMES_DUMP_REQUESTS"):
                agent._dump_api_request_debug(api_kwargs, reason="preflight")

            # Always prefer the streaming path — even without stream
            # consumers.  Streaming gives us fine-grained health
            # checking (90s stale-stream detection, 60s read timeout)
            # that the non-streaming path lacks.  Without this,
            # subagents and other quiet-mode callers can hang
            # indefinitely when the provider keeps the connection
            # alive with SSE pings but never delivers a response.
            # The streaming path is a no-op for callbacks when no
            # consumers are registered, and falls back to non-
            # streaming automatically if the provider doesn't
            # support it.
            def _stop_spinner():
                nonlocal thinking_spinner
                if thinking_spinner:
                    thinking_spinner.stop("")
                    thinking_spinner = None
                if agent.thinking_callback:
                    agent.thinking_callback("")

            _use_streaming = True
            # Provider signaled "stream not supported" on a previous
            # attempt — switch to non-streaming for the rest of this
            # session instead of re-failing every retry.
            if getattr(agent, "_disable_streaming", False):
                _use_streaming = False
            # CopilotACPClient communicates via subprocess stdio and
            # returns a plain SimpleNamespace — not an iterable
            # stream.  Mirror the ACP exclusion used for Responses
            # API upgrade (lines ~1083-1085).
            elif (
                agent.provider == "copilot-acp"
                or str(agent.base_url or "").lower().startswith("acp://copilot")
                or str(agent.base_url or "").lower().startswith("acp+tcp://")
            ):
                _use_streaming = False
            elif not agent._has_stream_consumers():
                # No display/TTS consumer. Still prefer streaming for
                # health checking, but skip for Mock clients in tests
                # (mocks return SimpleNamespace, not stream iterators).
                from unittest.mock import Mock
                if isinstance(getattr(agent, "client", None), Mock):
                    _use_streaming = False

            def _perform_api_call(next_api_kwargs):
                if _use_streaming:
                    return agent._interruptible_streaming_api_call(
                        next_api_kwargs, on_first_delta=_stop_spinner
                    )
                return agent._interruptible_api_call(next_api_kwargs)

            from hermes_cli.middleware import run_llm_execution_middleware

            response = run_llm_execution_middleware(
                api_kwargs,
                _perform_api_call,
                original_request=_original_api_kwargs,
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
                middleware_trace=list(_llm_middleware_trace),
            )

            api_duration = time.time() - api_start_time

            # Stop thinking spinner silently -- the response box or tool
            # execution messages that follow are more informative.
            if thinking_spinner:
                thinking_spinner.stop("")
                thinking_spinner = None
            if agent.thinking_callback:
                agent.thinking_callback("")

            if not agent.quiet_mode:
                agent._vprint(f"{agent.log_prefix}⏱️  API call completed in {api_duration:.2f}s")

            if agent.verbose_logging:
                # Log response with provider info if available
                resp_model = getattr(response, 'model', 'N/A') if response else 'N/A'
                logging.debug(f"API Response received - Model: {resp_model}, Usage: {response.usage if hasattr(response, 'usage') else 'N/A'}")

            # Validate response shape before proceeding
            response_invalid = False
            error_details = []
            if agent.api_mode == "codex_responses":
                _ct_v = agent._get_transport()
                if not _ct_v.validate_response(response):
                    if response is None:
                        response_invalid = True
                        error_details.append("response is None")
                    else:
                        # Provider returned a terminal failure (e.g. quota exhaustion).
                        # Treat as invalid so the fallback chain is triggered instead of
                        # letting the error bubble up outside the retry/fallback loop.
                        _codex_resp_status = str(getattr(response, "status", "") or "").strip().lower()
                        if _codex_resp_status in {"failed", "cancelled"}:
                            _codex_error_obj = getattr(response, "error", None)
                            _codex_error_msg = (
                                _codex_error_obj.get("message") if isinstance(_codex_error_obj, dict)
                                else str(_codex_error_obj) if _codex_error_obj
                                else f"Responses API returned status '{_codex_resp_status}'"
                            )
                            logger.warning(
                                "Codex response status='%s' (error=%s). Routing to fallback. %s",
                                _codex_resp_status, _codex_error_msg,
                                agent._client_log_context(),
                            )
                            response_invalid = True
                            error_details.append(f"response.status={_codex_resp_status}: {_codex_error_msg}")
                        else:
                            # output_text fallback: stream backfill may have failed
                            # but normalize can still recover from output_text
                            _out_text = getattr(response, "output_text", None)
                            _out_text_stripped = _out_text.strip() if isinstance(_out_text, str) else ""
                            if _out_text_stripped:
                                logger.debug(
                                    "Codex response.output is empty but output_text is present "
                                    "(%d chars); deferring to normalization.",
                                    len(_out_text_stripped),
                                )
                            else:
                                _resp_status = getattr(response, "status", None)
                                _resp_incomplete = getattr(response, "incomplete_details", None)
                                logger.warning(
                                    "Codex response.output is empty after stream backfill "
                                    "(status=%s, incomplete_details=%s, model=%s). %s",
                                    _resp_status, _resp_incomplete,
                                    getattr(response, "model", None),
                                    f"api_mode={agent.api_mode} provider={agent.provider}",
                                )
                                response_invalid = True
                                error_details.append("response.output is empty")
            elif agent.api_mode == "anthropic_messages":
                _tv = agent._get_transport()
                if not _tv.validate_response(response):
                    response_invalid = True
                    if response is None:
                        error_details.append("response is None")
                    else:
                        error_details.append("response.content invalid (not a non-empty list)")
            elif agent.api_mode == "bedrock_converse":
                _btv = agent._get_transport()
                if not _btv.validate_response(response):
                    response_invalid = True
                    if response is None:
                        error_details.append("response is None")
                    else:
                        error_details.append("Bedrock response invalid (no output or choices)")
            else:
                _ctv = agent._get_transport()
                if not _ctv.validate_response(response):
                    response_invalid = True
                    if response is None:
                        error_details.append("response is None")
                    elif not hasattr(response, 'choices'):
                        error_details.append("response has no 'choices' attribute")
                    elif response.choices is None:
                        error_details.append("response.choices is None")
                    else:
                        error_details.append("response.choices is empty")

            if response_invalid:
                agent._invoke_api_request_error_hook(
                    task_id=effective_task_id,
                    turn_id=turn_id,
                    api_request_id=api_request_id,
                    api_call_count=api_call_count,
                    api_start_time=api_start_time,
                    api_kwargs=api_kwargs,
                    error_type="InvalidAPIResponse",
                    error_message=", ".join(error_details) or "Invalid API response",
                    status_code=getattr(getattr(response, "error", None), "code", None),
                    retry_count=retry_count,
                    max_retries=max_retries,
                    retryable=True,
                    reason="invalid_response",
                )
                # Stop spinner silently — retry status is now buffered
                # and only surfaced if every retry+fallback exhausts.
                if thinking_spinner:
                    thinking_spinner.stop("")
                    thinking_spinner = None
                if agent.thinking_callback:
                    agent.thinking_callback("")

                # Invalid response — could be rate limiting, provider timeout,
                # upstream server error, or malformed response.
                retry_count += 1

                # Eager fallback: empty/malformed responses are a common
                # rate-limit symptom.  Switch to fallback immediately
                # rather than retrying with extended backoff.
                if agent._fallback_index < len(agent._fallback_chain):
                    agent._buffer_status("⚠️ Empty/malformed response — switching to fallback...")
                if agent._try_activate_fallback():
                    retry_count = 0
                    compression_attempts = 0
                    _retry.primary_recovery_attempted = False
                    continue

                # Check for error field in response (some providers include this)
                error_msg = "Unknown"
                provider_name = "Unknown"
                if response and hasattr(response, 'error') and response.error:
                    error_msg = str(response.error)
                    # Try to extract provider from error metadata
                    if hasattr(response.error, 'metadata') and response.error.metadata:
                        provider_name = response.error.metadata.get('provider_name', 'Unknown')
                elif response and hasattr(response, 'message') and response.message:
                    error_msg = str(response.message)

                # Try to get provider from model field (OpenRouter often returns actual model used)
                if provider_name == "Unknown" and response and hasattr(response, 'model') and response.model:
                    provider_name = f"model={response.model}"

                # Check for x-openrouter-provider or similar metadata
                if provider_name == "Unknown" and response:
                    # Log all response attributes for debugging
                    resp_attrs = {k: str(v)[:100] for k, v in vars(response).items() if not k.startswith('_')}
                    if agent.verbose_logging:
                        logging.debug(f"Response attributes for invalid response: {resp_attrs}")

                # Extract error code from response for contextual diagnostics
                _resp_error_code = None
                if response and hasattr(response, 'error') and response.error:
                    _code_raw = getattr(response.error, 'code', None)
                    if _code_raw is None and isinstance(response.error, dict):
                        _code_raw = response.error.get('code')
                    if _code_raw is not None:
                        try:
                            _resp_error_code = int(_code_raw)
                        except (TypeError, ValueError):
                            pass

                # Build a human-readable failure hint from the error code
                # and response time, instead of always assuming rate limiting.
                if _resp_error_code == 524:
                    _failure_hint = f"upstream provider timed out (Cloudflare 524, {api_duration:.0f}s)"
                elif _resp_error_code == 504:
                    _failure_hint = f"upstream gateway timeout (504, {api_duration:.0f}s)"
                elif _resp_error_code == 429:
                    _failure_hint = f"rate limited by upstream provider (429)"
                elif _resp_error_code in {500, 502}:
                    _failure_hint = f"upstream server error ({_resp_error_code}, {api_duration:.0f}s)"
                elif _resp_error_code in {503, 529}:
                    _failure_hint = f"upstream provider overloaded ({_resp_error_code})"
                elif _resp_error_code is not None:
                    _failure_hint = f"upstream error (code {_resp_error_code}, {api_duration:.0f}s)"
                elif api_duration < 10:
                    _failure_hint = f"fast response ({api_duration:.1f}s) — likely rate limited"
                elif api_duration > 60:
                    _failure_hint = f"slow response ({api_duration:.0f}s) — likely upstream timeout"
                else:
                    _failure_hint = f"response time {api_duration:.1f}s"

                agent._buffer_vprint(f"⚠️  Invalid API response (attempt {retry_count}/{max_retries}): {', '.join(error_details)}")
                agent._buffer_vprint(f"   🏢 Provider: {provider_name}")
                cleaned_provider_error = agent._clean_error_message(error_msg)
                agent._buffer_vprint(f"   📝 Provider message: {cleaned_provider_error}")
                agent._buffer_vprint(f"   ⏱️  {_failure_hint}")

                if retry_count >= max_retries:
                    # Try fallback before giving up
                    if agent._has_pending_fallback():
                        agent._buffer_status(f"⚠️ Max retries ({max_retries}) for invalid responses — trying fallback...")
                    if agent._try_activate_fallback():
                        retry_count = 0
                        compression_attempts = 0
                        _retry.primary_recovery_attempted = False
                        continue
                    # Terminal — flush buffered retry trace so user sees what happened.
                    agent._flush_status_buffer()
                    agent._emit_status(f"❌ Max retries ({max_retries}) exceeded for invalid responses. Giving up.")
                    logger.error(f"{agent.log_prefix}Invalid API response after {max_retries} retries.")
                    agent._persist_session(messages, conversation_history)
                    return _terminal({
                        "messages": messages,
                        "completed": False,
                        "api_calls": api_call_count,
                        "error": f"Invalid API response after {max_retries} retries: {_failure_hint}",
                        "failed": True  # Mark as failure for filtering
                    })

                # Backoff before retry — jittered exponential: 5s base, 120s cap
                wait_time = jittered_backoff(retry_count, base_delay=5.0, max_delay=120.0)
                agent._buffer_vprint(f"⏳ Retrying in {wait_time:.1f}s ({_failure_hint})...")
                logger.warning(f"Invalid API response (retry {retry_count}/{max_retries}): {', '.join(error_details)} | Provider: {provider_name}")

                # Sleep in small increments to stay responsive to interrupts
                sleep_end = time.time() + wait_time
                _backoff_touch_counter = 0
                while time.time() < sleep_end:
                    if agent._interrupt_requested:
                        agent._vprint(f"{agent.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                        agent._persist_session(messages, conversation_history)
                        agent.clear_interrupt()
                        return _terminal({
                            "final_response": f"Operation interrupted during retry ({_failure_hint}, attempt {retry_count}/{max_retries}).",
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "interrupted": True,
                        })
                    time.sleep(0.2)
                    # Touch activity every ~30s so the gateway's inactivity
                    # monitor knows we're alive during backoff waits.
                    _backoff_touch_counter += 1
                    if _backoff_touch_counter % 150 == 0:  # 150 × 0.2s = 30s
                        agent._touch_activity(
                            f"retry backoff ({retry_count}/{max_retries}), "
                            f"{int(sleep_end - time.time())}s remaining"
                        )
                continue  # Retry the API call

            # Check finish_reason before proceeding
            if agent.api_mode == "codex_responses":
                status = getattr(response, "status", None)
                incomplete_details = getattr(response, "incomplete_details", None)
                incomplete_reason = None
                if isinstance(incomplete_details, dict):
                    incomplete_reason = incomplete_details.get("reason")
                else:
                    incomplete_reason = getattr(incomplete_details, "reason", None)
                if status == "incomplete" and incomplete_reason in {"max_output_tokens", "length"}:
                    finish_reason = "length"
                else:
                    finish_reason = "stop"
            elif agent.api_mode == "anthropic_messages":
                _tfr = agent._get_transport()
                finish_reason = _tfr.map_finish_reason(response.stop_reason)
            elif agent.api_mode == "bedrock_converse":
                # Bedrock response already normalized at dispatch — use transport
                _bt_fr = agent._get_transport()
                _bedrock_result = _bt_fr.normalize_response(response)
                finish_reason = _bedrock_result.finish_reason
            else:
                _cc_fr = agent._get_transport()
                _finish_result = _cc_fr.normalize_response(response)
                finish_reason = _finish_result.finish_reason
                assistant_message = _finish_result
                if agent._should_treat_stop_as_truncated(
                    finish_reason,
                    assistant_message,
                    messages,
                ):
                    agent._vprint(
                        f"{agent.log_prefix}⚠️  Treating suspicious Ollama/GLM stop response as truncated",
                        force=True,
                    )
                    finish_reason = "length"

            # ── Content-policy refusal (HTTP 200) ──────────────────
            # The model — or the provider's safety system — returned a
            # *successful* response whose stop/finish reason is a refusal:
            # Anthropic ``stop_reason="refusal"`` → ``content_filter``;
            # OpenAI / portal ``finish_reason="content_filter"`` or a
            # populated ``message.refusal`` (mapped in the chat_completions
            # transport); Bedrock ``guardrail_intervened``. The content is
            # typically empty, so without this branch the response falls
            # through to the empty-response / invalid-response retry loops
            # and is mis-surfaced as "rate limited" / "no content after
            # retries" — burning paid attempts reproducing a deterministic
            # refusal. Surface it clearly and stop. Mirrors the
            # exception-based ``content_policy_blocked`` recovery: try a
            # configured fallback once, otherwise return the refusal.
            if finish_reason == "content_filter":
                _refusal_transport = agent._get_transport()
                if agent.api_mode == "anthropic_messages":
                    _refusal_result = _refusal_transport.normalize_response(
                        response, strip_tool_prefix=agent._is_anthropic_oauth
                    )
                else:
                    _refusal_result = _refusal_transport.normalize_response(response)
                _refusal_text = (getattr(_refusal_result, "content", None) or "").strip()
                # Some refusals carry the explanation only in the reasoning
                # channel; fall back to it so the user sees *something*.
                if not _refusal_text:
                    _refusal_text = (agent._extract_reasoning(_refusal_result) or "").strip()

                agent._invoke_api_request_error_hook(
                    task_id=effective_task_id,
                    turn_id=turn_id,
                    api_request_id=api_request_id,
                    api_call_count=api_call_count,
                    api_start_time=api_start_time,
                    api_kwargs=api_kwargs,
                    error_type="ContentPolicyBlocked",
                    error_message=_refusal_text or "model declined to respond (content_filter)",
                    status_code=None,
                    retry_count=retry_count,
                    max_retries=max_retries,
                    retryable=False,
                    reason=FailoverReason.content_policy_blocked.value,
                )

                if thinking_spinner:
                    thinking_spinner.stop("")
                    thinking_spinner = None
                if agent.thinking_callback:
                    agent.thinking_callback("")

                # Deterministic for the unchanged prompt — never retry.
                # Try a configured fallback once (a different model may not
                # refuse); otherwise surface the refusal terminally.
                if agent._has_pending_fallback():
                    agent._buffer_status(
                        "⚠️ Model declined to respond (safety refusal) — trying fallback..."
                    )
                if agent._try_activate_fallback():
                    retry_count = 0
                    compression_attempts = 0
                    _retry.primary_recovery_attempted = False
                    continue

                agent._flush_status_buffer()
                _refusal_log = (
                    _refusal_text[:500] + "..."
                    if len(_refusal_text) > 500
                    else _refusal_text
                )
                logger.warning(
                    "%sModel declined to respond (finish_reason=content_filter). "
                    "model=%s provider=%s refusal=%s",
                    agent.log_prefix, agent.model, agent.provider,
                    _refusal_log or "(no text)",
                )
                agent._emit_status(
                    "⚠️ The model declined to respond to this request (safety refusal)."
                )

                _refusal_detail = (
                    f"Model's explanation: {_refusal_text}"
                    if _refusal_text
                    else "The model returned no explanation."
                )
                _refusal_response = (
                    "⚠️  The model declined to respond to this request "
                    "(safety refusal — not a Hermes/gateway failure).\n\n"
                    f"{_refusal_detail}\n\n"
                    f"{_CONTENT_POLICY_RECOVERY_HINT}"
                )

                agent._cleanup_task_resources(effective_task_id)
                agent._persist_session(messages, conversation_history)
                return _terminal(_content_policy_blocked_result(
                    messages,
                    api_call_count,
                    final_response=_refusal_response,
                    error_detail=_refusal_text or "model declined (content_filter)",
                ))

            if finish_reason == "length":
                if getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID:
                    agent._vprint(
                        f"{agent.log_prefix}⚠️  Stream interrupted by network error "
                        f"(finish_reason='length' on partial-stream-stub)",
                        force=True,
                    )
                else:
                    agent._vprint(
                        f"{agent.log_prefix}⚠️  Response truncated "
                        f"(finish_reason='length') - model hit max output tokens",
                        force=True,
                    )

                # Normalize the truncated response to a single OpenAI-style
                # message shape so text-continuation and tool-call retry
                # work uniformly across chat_completions, bedrock_converse,
                # and anthropic_messages.  For Anthropic we use the same
                # adapter the agent loop already relies on so the rebuilt
                # interim assistant message is byte-identical to what
                # would have been appended in the non-truncated path.
                _trunc_msg = None
                _trunc_transport = agent._get_transport()
                if agent.api_mode == "anthropic_messages":
                    _trunc_result = _trunc_transport.normalize_response(
                        response, strip_tool_prefix=agent._is_anthropic_oauth
                    )
                else:
                    _trunc_result = _trunc_transport.normalize_response(response)
                _trunc_msg = _trunc_result

                _trunc_content = getattr(_trunc_msg, "content", None) if _trunc_msg else None
                _trunc_has_tool_calls = bool(getattr(_trunc_msg, "tool_calls", None)) if _trunc_msg else False

                # ── Detect thinking-budget exhaustion ──────────────
                # When the model spends ALL output tokens on reasoning
                # and has none left for the response, continuation
                # retries are pointless.  Detect this early and give a
                # targeted error instead of wasting 3 API calls.
                # A response is "thinking exhausted" only when the model
                # actually produced reasoning blocks but no visible text after
                # them.  Models that do not use <think> tags (e.g. GLM-4.7 on
                # NVIDIA Build, minimax) may return content=None or an empty
                # string for unrelated reasons — treat those as normal
                # truncations that deserve continuation retries, not as
                # thinking-budget exhaustion.
                _has_think_tags = bool(
                    _trunc_content and re.search(
                        r'<(?:think|thinking|reasoning|REASONING_SCRATCHPAD)[^>]*>',
                        _trunc_content,
                        re.IGNORECASE,
                    )
                )
                _thinking_exhausted = (
                    not _trunc_has_tool_calls
                    and _has_think_tags
                    and (
                        (_trunc_content is not None and not agent._has_content_after_think_block(_trunc_content))
                        or _trunc_content is None
                    )
                )

                if _thinking_exhausted:
                    _exhaust_error = (
                        "Model used all output tokens on reasoning with none left "
                        "for the response. Try lowering reasoning effort or "
                        "increasing max_tokens."
                    )
                    agent._vprint(
                        f"{agent.log_prefix}💭 Reasoning exhausted the output token budget — "
                        f"no visible response was produced.",
                        force=True,
                    )
                    # Return a user-friendly message as the response so
                    # CLI (response box) and gateway (chat message) both
                    # display it naturally instead of a suppressed error.
                    _exhaust_response = (
                        "⚠️ **Thinking Budget Exhausted**\n\n"
                        "The model used all its output tokens on reasoning "
                        "and had none left for the actual response.\n\n"
                        "To fix this:\n"
                        "→ Lower reasoning effort: `/thinkon low` or `/thinkon minimal`\n"
                        "→ Or switch to a larger/non-reasoning model with `/model`"
                    )
                    agent._cleanup_task_resources(effective_task_id)
                    agent._persist_session(messages, conversation_history)
                    return _terminal({
                        "final_response": _exhaust_response,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "partial": True,
                        "error": _exhaust_error,
                    })

                if agent.api_mode in {"chat_completions", "bedrock_converse", "anthropic_messages"}:
                    assistant_message = _trunc_msg
                    if assistant_message is not None and not _trunc_has_tool_calls:
                        length_continue_retries += 1
                        interim_msg = agent._build_assistant_message(assistant_message, finish_reason)
                        messages.append(interim_msg)
                        if assistant_message.content:
                            truncated_response_parts.append(assistant_message.content)

                        if length_continue_retries < 3:
                            _is_partial_stream_stub = (
                                getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID
                            )
                            _dropped_tools = getattr(
                                response, "_dropped_tool_names", None
                            )

                            if _is_partial_stream_stub and _dropped_tools:
                                _tool_list = ", ".join(_dropped_tools[:3])
                                agent._vprint(
                                    f"{agent.log_prefix}↻ Stream interrupted mid "
                                    f"tool-call ({_tool_list}) — requesting "
                                    f"chunked retry "
                                    f"({length_continue_retries}/3)..."
                                )
                            elif _is_partial_stream_stub:
                                agent._vprint(
                                    f"{agent.log_prefix}↻ Stream interrupted — "
                                    f"requesting continuation "
                                    f"({length_continue_retries}/3)..."
                                )
                            else:
                                agent._vprint(
                                    f"{agent.log_prefix}↻ Requesting continuation "
                                    f"({length_continue_retries}/3)..."
                                )

                            _continue_content = _get_continuation_prompt(
                                _is_partial_stream_stub, _dropped_tools
                            )
                            continue_msg = {
                                "role": "user",
                                "content": _continue_content,
                            }
                            messages.append(continue_msg)
                            agent._session_messages = messages
                            _retry.restart_with_length_continuation = True
                            break

                        partial_response = agent._strip_think_blocks("".join(truncated_response_parts)).strip()
                        agent._cleanup_task_resources(effective_task_id)
                        agent._persist_session(messages, conversation_history)
                        return _terminal({
                            "final_response": partial_response or None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": "Response remained truncated after 3 continuation attempts",
                        })

                if agent.api_mode in {"chat_completions", "bedrock_converse", "anthropic_messages"}:
                    assistant_message = _trunc_msg
                    if assistant_message is not None and _trunc_has_tool_calls:
                        _is_stub_stall = (
                            getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID
                        )
                        if truncated_tool_call_retries < 3:
                            truncated_tool_call_retries += 1
                            if _is_stub_stall:
                                # The stream broke mid tool-call (network /
                                # peer-closed connection), not a real output
                                # cap — say so instead of "max output tokens".
                                agent._buffer_vprint(
                                    f"⚠️  Stream interrupted mid tool-call — "
                                    f"retrying ({truncated_tool_call_retries}/3)..."
                                )
                            else:
                                agent._buffer_vprint(
                                    f"⚠️  Truncated tool call detected — "
                                    f"retrying API call "
                                    f"({truncated_tool_call_retries}/3)..."
                                )
                            # Boost max_tokens on each retry so the model has
                            # more room to complete the tool-call JSON. A
                            # network stall doesn't need a bigger budget, but
                            # a genuine output-cap truncation does, and the
                            # boost is harmless for the stall case.
                            _tc_boost_base = agent.max_tokens if agent.max_tokens else 4096
                            _tc_boost = _tc_boost_base * (truncated_tool_call_retries + 1)
                            _tc_requested_cap = agent._requested_output_cap_from_api_kwargs(api_kwargs)
                            if _tc_requested_cap is not None:
                                _tc_boost = max(_tc_boost, _tc_requested_cap)
                            _tc_boost_cap = max(32768, _tc_requested_cap or 0)
                            agent._ephemeral_max_output_tokens = min(_tc_boost, _tc_boost_cap)
                            # Don't append the broken response to messages;
                            # just re-run the same API call from the current
                            # message state, giving the model another chance.
                            continue
                        agent._flush_status_buffer()
                        if _is_stub_stall:
                            agent._vprint(
                                f"{agent.log_prefix}⚠️  Stream kept dropping mid tool-call after 3 retries — the action was not executed.",
                                force=True,
                            )
                        else:
                            agent._vprint(
                                f"{agent.log_prefix}⚠️  Truncated tool call response detected again — refusing to execute incomplete tool arguments.",
                                force=True,
                            )
                        agent._cleanup_task_resources(effective_task_id)
                        agent._persist_session(messages, conversation_history)
                        return _terminal({
                            "final_response": None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": (
                                "Stream repeatedly dropped mid tool-call (network); "
                                "the tool was not executed"
                                if _is_stub_stall
                                else "Response truncated due to output length limit"
                            ),
                        })

                # If we have prior messages, roll back to last complete state
                if len(messages) > 1:
                    agent._vprint(f"{agent.log_prefix}   ⏪ Rolling back to last complete assistant turn")
                    rolled_back_messages = agent._get_messages_up_to_last_assistant(messages)

                    agent._cleanup_task_resources(effective_task_id)
                    agent._persist_session(messages, conversation_history)

                    return _terminal({
                        "final_response": None,
                        "messages": rolled_back_messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "partial": True,
                        "error": "Response truncated due to output length limit"
                    })
                else:
                    # First message was truncated - mark as failed
                    agent._flush_status_buffer()
                    agent._vprint(f"{agent.log_prefix}❌ First response truncated - cannot recover", force=True)
                    agent._persist_session(messages, conversation_history)
                    return _terminal({
                        "final_response": None,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "failed": True,
                        "error": "First response truncated due to output length limit"
                    })

            # Track actual token usage from response for context management
            if hasattr(response, 'usage') and response.usage:
                canonical_usage = normalize_usage(
                    response.usage,
                    provider=agent.provider,
                    api_mode=agent.api_mode,
                )
                prompt_tokens = canonical_usage.prompt_tokens
                completion_tokens = canonical_usage.output_tokens
                total_tokens = canonical_usage.total_tokens
                # Forward canonical token + cache buckets so context engines
                # can make decisions on cache hit ratios / reasoning costs,
                # not just legacy aggregate tokens. Legacy keys stay for
                # back-compat with engines that only read prompt/completion/total.
                usage_dict = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "input_tokens": canonical_usage.input_tokens,
                    "output_tokens": canonical_usage.output_tokens,
                    "cache_read_tokens": canonical_usage.cache_read_tokens,
                    "cache_write_tokens": canonical_usage.cache_write_tokens,
                    "reasoning_tokens": canonical_usage.reasoning_tokens,
                }
                agent.context_compressor.update_from_response(usage_dict)

                # Cache discovered context length after successful call.
                # Only persist limits confirmed by the provider (parsed
                # from the error message), not guessed probe tiers.
                if getattr(agent.context_compressor, "_context_probed", False):
                    ctx = agent.context_compressor.context_length
                    if getattr(agent.context_compressor, "_context_probe_persistable", False):
                        save_context_length(agent.model, agent.base_url, ctx)
                        agent._safe_print(f"{agent.log_prefix}💾 Cached context length: {ctx:,} tokens for {agent.model}")
                    agent.context_compressor._context_probed = False
                    agent.context_compressor._context_probe_persistable = False

                agent.session_prompt_tokens += prompt_tokens
                agent.session_completion_tokens += completion_tokens
                agent.session_total_tokens += total_tokens
                agent.session_api_calls += 1
                agent.session_input_tokens += canonical_usage.input_tokens
                agent.session_output_tokens += canonical_usage.output_tokens
                agent.session_cache_read_tokens += canonical_usage.cache_read_tokens
                agent.session_cache_write_tokens += canonical_usage.cache_write_tokens
                agent.session_reasoning_tokens += canonical_usage.reasoning_tokens

                # Log API call details for debugging/observability
                _cache_pct = ""
                if canonical_usage.cache_read_tokens and prompt_tokens:
                    _cache_pct = f" cache={canonical_usage.cache_read_tokens}/{prompt_tokens} ({100*canonical_usage.cache_read_tokens/prompt_tokens:.0f}%)"
                logger.info(
                    "API call #%d: model=%s provider=%s in=%d out=%d total=%d latency=%.1fs%s",
                    agent.session_api_calls, agent.model, agent.provider or "unknown",
                    prompt_tokens, completion_tokens, total_tokens,
                    api_duration, _cache_pct,
                )

                cost_result = estimate_usage_cost(
                    agent.model,
                    canonical_usage,
                    provider=agent.provider,
                    base_url=agent.base_url,
                    api_key=getattr(agent, "api_key", ""),
                )
                if cost_result.amount_usd is not None:
                    agent.session_estimated_cost_usd += float(cost_result.amount_usd)
                agent.session_cost_status = cost_result.status
                agent.session_cost_source = cost_result.source

                # Persist token counts to session DB for /insights.
                # Do this for every platform with a session_id so non-CLI
                # sessions (gateway, cron, delegated runs) cannot lose
                # token/accounting data if a higher-level persistence path
                # is skipped or fails. Gateway/session-store writes use
                # absolute totals, so they safely overwrite these per-call
                # deltas instead of double-counting them.
                if agent._session_db and agent.session_id:
                    try:
                        # Ensure the session row exists before attempting UPDATE.
                        # Under concurrent load (cron/kanban), the initial
                        # _ensure_db_session() may have failed due to SQLite
                        # locking.  Retry here so per-call token deltas are
                        # not silently lost (UPDATE on a non-existent row
                        # affects 0 rows without error).
                        if not agent._session_db_created:
                            agent._ensure_db_session()
                        agent._session_db.update_token_counts(
                            agent.session_id,
                            input_tokens=canonical_usage.input_tokens,
                            output_tokens=canonical_usage.output_tokens,
                            cache_read_tokens=canonical_usage.cache_read_tokens,
                            cache_write_tokens=canonical_usage.cache_write_tokens,
                            reasoning_tokens=canonical_usage.reasoning_tokens,
                            estimated_cost_usd=float(cost_result.amount_usd)
                            if cost_result.amount_usd is not None else None,
                            cost_status=cost_result.status,
                            cost_source=cost_result.source,
                            billing_provider=agent.provider,
                            billing_base_url=agent.base_url,
                            billing_mode="subscription_included"
                            if cost_result.status == "included" else None,
                            model=agent.model,
                            api_call_count=1,
                        )
                    except Exception as e:
                        # Log token persistence failures so they're
                        # visible in agent.log — silent loss here is
                        # the root cause of undercounted analytics.
                        logger.debug(
                            "Token persistence failed (session=%s, tokens=%d): %s",
                            agent.session_id, total_tokens, e,
                        )

                if agent.verbose_logging:
                    logging.debug(f"Token usage: prompt={usage_dict['prompt_tokens']:,}, completion={usage_dict['completion_tokens']:,}, total={usage_dict['total_tokens']:,}")

                # Surface cache hit stats for any provider that reports
                # them — not just those where we inject cache_control
                # markers.  OpenAI/Kimi/DeepSeek/Qwen all do automatic
                # server-side prefix caching and return
                # ``prompt_tokens_details.cached_tokens``; users
                # previously could not see their cache % because this
                # line was gated on ``_use_prompt_caching``, which is
                # only True for Anthropic-style marker injection.
                # ``canonical_usage`` is already normalised from all
                # three API shapes (Anthropic / Codex / OpenAI-chat)
                # so we can rely on its values directly.
                cached = canonical_usage.cache_read_tokens
                written = canonical_usage.cache_write_tokens
                prompt = usage_dict["prompt_tokens"]
                if (cached or written) and not agent.quiet_mode:
                    hit_pct = (cached / prompt * 100) if prompt > 0 else 0
                    agent._vprint(
                        f"{agent.log_prefix}   💾 Cache: "
                        f"{cached:,}/{prompt:,} tokens "
                        f"({hit_pct:.0f}% hit, {written:,} written)"
                    )

            _retry.has_retried_429 = False  # Reset on success
            # Note: don't clear the retry buffer here — an "API call
            # success" only means we got bytes back, not that we got
            # usable content. Empty responses still loop through the
            # empty-retry path below; the buffer is cleared when
            # genuinely successful content is detected later (~L4127).
            # Clear Nous rate limit state on successful request —
            # proves the limit has reset and other sessions can
            # resume hitting Nous.
            if agent.provider == "nous":
                try:
                    from agent.nous_rate_guard import clear_nous_rate_limit
                    clear_nous_rate_limit()
                except Exception:
                    pass
            agent._touch_activity(f"API call #{api_call_count} completed")
            break  # Success, exit retry loop

        except InterruptedError:
            if thinking_spinner:
                thinking_spinner.stop("")
                thinking_spinner = None
            if agent.thinking_callback:
                agent.thinking_callback("")
            api_elapsed = time.time() - api_start_time
            agent._vprint(f"{agent.log_prefix}⚡ Interrupted during API call.", force=True)
            agent._persist_session(messages, conversation_history)
            interrupted = True
            final_response = f"{INTERRUPT_WAITING_FOR_MODEL_PREFIX}{api_elapsed:.1f}s elapsed)."
            break

        except Exception as api_error:
            # Stop spinner silently — retry status is buffered and
            # only flushed when every retry+fallback is exhausted.
            if thinking_spinner:
                thinking_spinner.stop("")
                thinking_spinner = None
            if agent.thinking_callback:
                agent.thinking_callback("")

            # -----------------------------------------------------------
            # UnicodeEncodeError recovery.  Two common causes:
            #   1. Lone surrogates (U+D800..U+DFFF) from clipboard paste
            #      (Google Docs, rich-text editors) — sanitize and retry.
            #   2. ASCII codec on systems with LANG=C or non-UTF-8 locale
            #      (e.g. Chromebooks) — any non-ASCII character fails.
            #      Detect via the error message mentioning 'ascii' codec.
            # We sanitize messages in-place and may retry twice:
            # first to strip surrogates, then once more for pure
            # ASCII-only locale sanitization if needed.
            # -----------------------------------------------------------
            if isinstance(api_error, UnicodeEncodeError) and getattr(agent, '_unicode_sanitization_passes', 0) < 2:
                _err_str = str(api_error).lower()
                _is_ascii_codec = "'ascii'" in _err_str or "ascii" in _err_str
                # Detect surrogate errors — utf-8 codec refusing to
                # encode U+D800..U+DFFF.  The error text is:
                #   "'utf-8' codec can't encode characters in position
                #    N-M: surrogates not allowed"
                _is_surrogate_error = (
                    "surrogate" in _err_str
                    or ("'utf-8'" in _err_str and not _is_ascii_codec)
                )
                # Sanitize surrogates from both the canonical `messages`
                # list AND `api_messages` (the API-copy, which may carry
                # `reasoning_content`/`reasoning_details` transformed
                # from `reasoning` — fields the canonical list doesn't
                # have directly).  Also clean `api_kwargs` if built and
                # `prefill_messages` if present.  Mirrors the ASCII
                # codec recovery below.
                _surrogates_found = _sanitize_messages_surrogates(messages)
                if isinstance(api_messages, list):
                    if _sanitize_messages_surrogates(api_messages):
                        _surrogates_found = True
                if isinstance(api_kwargs, dict):
                    if _sanitize_structure_surrogates(api_kwargs):
                        _surrogates_found = True
                if isinstance(getattr(agent, "prefill_messages", None), list):
                    if _sanitize_messages_surrogates(agent.prefill_messages):
                        _surrogates_found = True
                # Gate the retry on the error type, not on whether we
                # found anything — _force_ascii_payload / the extended
                # surrogate walker above cover all known paths, but a
                # new transformed field could still slip through.  If
                # the error was a surrogate encode failure, always let
                # the retry run; the proactive sanitizer at line ~8781
                # runs again on the next iteration.  Bounded by
                # _unicode_sanitization_passes < 2 (outer guard).
                if _surrogates_found or _is_surrogate_error:
                    agent._unicode_sanitization_passes += 1
                    if _surrogates_found:
                        agent._buffer_vprint(
                            f"⚠️  Stripped invalid surrogate characters from messages. Retrying..."
                        )
                    else:
                        agent._buffer_vprint(
                            f"⚠️  Surrogate encoding error — retrying after full-payload sanitization..."
                        )
                    continue
                if _is_ascii_codec:
                    agent._force_ascii_payload = True
                    # ASCII codec: the system encoding can't handle
                    # non-ASCII characters at all. Sanitize all
                    # non-ASCII content from messages/tool schemas and retry.
                    # Sanitize both the canonical `messages` list and
                    # `api_messages` (the API-copy built before the retry
                    # loop, which may contain extra fields like
                    # reasoning_content that are not in `messages`).
                    _messages_sanitized = _sanitize_messages_non_ascii(messages)
                    if isinstance(api_messages, list):
                        _sanitize_messages_non_ascii(api_messages)
                    # Also sanitize the last api_kwargs if already built,
                    # so a leftover non-ASCII value in a transformed field
                    # (e.g. extra_body, reasoning_content) doesn't survive
                    # into the next attempt via _build_api_kwargs cache paths.
                    if isinstance(api_kwargs, dict):
                        _sanitize_structure_non_ascii(api_kwargs)
                    _prefill_sanitized = False
                    if isinstance(getattr(agent, "prefill_messages", None), list):
                        _prefill_sanitized = _sanitize_messages_non_ascii(agent.prefill_messages)

                    _tools_sanitized = False
                    if isinstance(getattr(agent, "tools", None), list):
                        _tools_sanitized = _sanitize_tools_non_ascii(agent.tools)

                    active_system_prompt, _system_sanitized = sanitize_active_system_prompt(
                        agent, active_system_prompt, strip_non_ascii=_strip_non_ascii,
                    )
                    if isinstance(getattr(agent, "ephemeral_system_prompt", None), str):
                        _sanitized_ephemeral = _strip_non_ascii(agent.ephemeral_system_prompt)
                        if _sanitized_ephemeral != agent.ephemeral_system_prompt:
                            agent.ephemeral_system_prompt = _sanitized_ephemeral
                            _system_sanitized = True

                    _headers_sanitized = False
                    _default_headers = (
                        agent._client_kwargs.get("default_headers")
                        if isinstance(getattr(agent, "_client_kwargs", None), dict)
                        else None
                    )
                    if isinstance(_default_headers, dict):
                        _headers_sanitized = _sanitize_structure_non_ascii(_default_headers)

                    # Sanitize the API key — non-ASCII characters in
                    # credentials (e.g. ʋ instead of v from a bad
                    # copy-paste) cause httpx to fail when encoding
                    # the Authorization header as ASCII.  This is the
                    # most common cause of persistent UnicodeEncodeError
                    # that survives message/tool sanitization (#6843).
                    _credential_sanitized = False
                    _raw_key = getattr(agent, "api_key", None) or ""
                    # Entra ID bearer providers are callables — their
                    # minted JWTs are always ASCII, so no sanitization
                    # is needed (and ``_strip_non_ascii`` would crash
                    # on a callable input).
                    if _raw_key and isinstance(_raw_key, str):
                        _clean_key = _strip_non_ascii(_raw_key)
                        if _clean_key != _raw_key:
                            agent.api_key = _clean_key
                            if isinstance(getattr(agent, "_client_kwargs", None), dict):
                                agent._client_kwargs["api_key"] = _clean_key
                            # Also update the live client — it holds its
                            # own copy of api_key which auth_headers reads
                            # dynamically on every request.
                            if getattr(agent, "client", None) is not None and hasattr(agent.client, "api_key"):
                                agent.client.api_key = _clean_key
                            _credential_sanitized = True
                            agent._vprint(
                                f"{agent.log_prefix}⚠️  API key contained non-ASCII characters "
                                f"(bad copy-paste?) — stripped them. If auth fails, "
                                f"re-copy the key from your provider's dashboard.",
                                force=True,
                            )

                    # Always retry on ASCII codec detection —
                    # _force_ascii_payload guarantees the full
                    # api_kwargs payload is sanitized on the
                    # next iteration (line ~8475).  Even when
                    # per-component checks above find nothing
                    # (e.g. non-ASCII only in api_messages'
                    # reasoning_content), the flag catches it.
                    # Bounded by _unicode_sanitization_passes < 2.
                    agent._unicode_sanitization_passes += 1
                    _any_sanitized = (
                        _messages_sanitized
                        or _prefill_sanitized
                        or _tools_sanitized
                        or _system_sanitized
                        or _headers_sanitized
                        or _credential_sanitized
                    )
                    if _any_sanitized:
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  System encoding is ASCII — stripped non-ASCII characters from request payload. Retrying...",
                            force=True,
                        )
                    else:
                        agent._vprint(
                            f"{agent.log_prefix}⚠️  System encoding is ASCII — enabling full-payload sanitization for retry...",
                            force=True,
                        )
                    continue

            # ── Image-rejection recovery ──────────────────────────────
            # Some providers (mlx-lm, text-only endpoints, text-only
            # fallbacks on multimodal models) reject any message that
            # contains image_url content with a 4xx error like
            # "Only 'text' content type is supported."  On first hit,
            # strip all images from the message list, mark the session
            # as vision-unsupported, and retry with text only.
            #
            # Detection is best-effort English phrase matching — a
            # locale-translated or heavily-reworded upstream error
            # will bypass this guard and fall through to the normal
            # error handler.  Expand the phrase list when new
            # provider wordings are observed in the wild.
            _err_body = ""
            try:
                _err_body = str(getattr(api_error, "body", None) or
                                getattr(api_error, "message", None) or
                                str(api_error))
            except Exception:
                pass
            _err_status = getattr(api_error, "status_code", None)
            _IMAGE_REJECTION_PHRASES = (
                "only 'text' content type is supported",
                "only text content type is supported",
                "image_url is not supported",
                "image content is not supported",
                "multimodal is not supported",
                "multimodal content is not supported",
                "multimodal input is not supported",
                "vision is not supported",
                "vision input is not supported",
                "does not support images",
                "does not support image input",
                "does not support multimodal",
                "does not support vision",
                "model does not support image",
                # ChatGPT-account Codex backend
                # (https://chatgpt.com/backend-api/codex) rejects
                # data:image/...base64 URLs in input_image fields
                # with HTTP 400 "Invalid 'input[N].content[K].image_url'.
                # Expected a valid URL, but got a value with an
                # invalid format." The OpenAI Responses API on the
                # public endpoint accepts data URLs, but the
                # ChatGPT-account variant does not. Without this
                # phrase the agent cascaded into compression /
                # context-too-large recovery instead of just
                # stripping the images. Match is narrow on
                # purpose — keyed on the field-path apostrophe so
                # we don't false-trip on other URL validation
                # errors. (issue #23570)
                "image_url'. expected",
                # DeepSeek's OpenAI-compatible API reports text-only
                # request-body variants as:
                # "unknown variant `image_url`, expected `text`".
                "unknown variant `image_url`, expected `text`",
                "unknown variant image_url, expected text",
            )
            _err_lower = _err_body.lower()
            _looks_like_image_rejection = any(
                p in _err_lower for p in _IMAGE_REJECTION_PHRASES
            )
            # 4xx-only gate: never interpret 5xx/timeout as "server
            # said no to images" — those are transient and must
            # route to the normal retry path.
            _status_ok = _err_status is None or (400 <= int(_err_status) < 500)
            if (
                getattr(agent, "_vision_supported", True)
                and _looks_like_image_rejection
                and _status_ok
            ):
                agent._vision_supported = False
                _imgs_removed = _strip_images_from_messages(messages)
                if isinstance(api_messages, list):
                    _strip_images_from_messages(api_messages)
                agent._vprint(
                    f"{agent.log_prefix}⚠️  Server rejected image content — "
                    f"switching to text-only mode for this session"
                    + (". Stripped images from history and retrying." if _imgs_removed else "."),
                    force=True,
                )
                continue

            status_code = getattr(api_error, "status_code", None)
            error_context = agent._extract_api_error_context(api_error)

            # ── Classify the error for structured recovery decisions ──
            _compressor = getattr(agent, "context_compressor", None)
            _ctx_len = getattr(_compressor, "context_length", 200000) if _compressor else 200000
            classified = classify_api_error(
                api_error,
                provider=getattr(agent, "provider", "") or "",
                model=getattr(agent, "model", "") or "",
                approx_tokens=approx_tokens,
                context_length=_ctx_len,
                num_messages=len(api_messages) if api_messages else 0,
            )
            logger.debug(
                "Error classified: reason=%s status=%s retryable=%s compress=%s rotate=%s fallback=%s",
                classified.reason.value, classified.status_code,
                classified.retryable, classified.should_compress,
                classified.should_rotate_credential, classified.should_fallback,
            )
            agent._invoke_api_request_error_hook(
                task_id=effective_task_id,
                turn_id=turn_id,
                api_request_id=api_request_id,
                api_call_count=api_call_count,
                api_start_time=api_start_time,
                api_kwargs=api_kwargs,
                error_type=type(api_error).__name__,
                error_message=str(api_error),
                status_code=status_code,
                retry_count=retry_count,
                max_retries=max_retries,
                retryable=classified.retryable,
                reason=classified.reason.value,
            )

            if (
                classified.reason == FailoverReason.billing
                and _is_nous_inference_route(
                    getattr(agent, "provider", "") or "",
                    getattr(agent, "base_url", "") or "",
                )
                and not _retry.nous_paid_entitlement_refresh_attempted
            ):
                _retry.nous_paid_entitlement_refresh_attempted = True
                if _try_refresh_nous_paid_entitlement_credentials(agent):
                    agent._vprint(
                        f"{agent.log_prefix}🔐 Nous paid access verified — "
                        "refreshed runtime credentials and retrying request...",
                        force=True,
                    )
                    continue

            recovered_with_pool, _retry.has_retried_429 = agent._recover_with_credential_pool(
                status_code=status_code,
                has_retried_429=_retry.has_retried_429,
                classified_reason=classified.reason,
                error_context=error_context,
            )
            if recovered_with_pool:
                continue

            # Image-too-large recovery: shrink oversized native image
            # parts in-place and retry once.  Triggered by Anthropic's
            # per-image 5 MB ceiling (400 with "image exceeds 5 MB
            # maximum") or any other provider that complains about
            # image size.  If shrink fails or a second attempt still
            # fails, fall through to normal error handling.
            if (
                classified.reason == FailoverReason.image_too_large
                and not _retry.image_shrink_retry_attempted
            ):
                _retry.image_shrink_retry_attempted = True
                image_max_dimension = _image_error_max_dimension(api_error) or 8000
                if agent._try_shrink_image_parts_in_messages(
                    api_messages,
                    max_dimension=image_max_dimension,
                ):
                    agent._vprint(
                        f"{agent.log_prefix}📐 Image(s) exceeded provider size limit — "
                        f"shrank and retrying...",
                        force=True,
                    )
                    continue
                else:
                    logger.info(
                        "image-shrink recovery: no data-URL image parts found "
                        "or shrink didn't reduce size; surfacing original error."
                    )

            # Multimodal-tool-content recovery: providers that follow
            # the OpenAI spec strictly (tool message content must be a
            # string) reject our list-type content with a 400.  Strip
            # image parts from any list-type tool messages, mark the
            # (provider, model) as no-list-tool-content for the rest
            # of this session so future tool results preemptively
            # downgrade, and retry once.  See issue #27344.
            if (
                classified.reason == FailoverReason.multimodal_tool_content_unsupported
                and not _retry.multimodal_tool_content_retry_attempted
            ):
                _retry.multimodal_tool_content_retry_attempted = True
                if agent._try_strip_image_parts_from_tool_messages(api_messages):
                    agent._vprint(
                        f"{agent.log_prefix}📐 Provider rejected list-type tool content — "
                        f"downgraded screenshots to text and retrying...",
                        force=True,
                    )
                    continue
                else:
                    logger.info(
                        "multimodal-tool-content recovery: no list-type tool "
                        "messages with image parts found; surfacing original error."
                    )

            # Anthropic OAuth subscription rejected the 1M-context beta
            # header ("long context beta is not yet available for this
            # subscription"). Disable the beta for the rest of this
            # session, rebuild the client, and retry once.  1M-capable
            # subscriptions never hit this branch — they accept the
            # beta and keep full 1M context.  See PR #17680 for the
            # original report (we chose reactive recovery over the
            # proposed unconditional omit so capable subscriptions
            # don't silently lose the capability).
            if (
                classified.reason == FailoverReason.oauth_long_context_beta_forbidden
                and agent.api_mode == "anthropic_messages"
                and agent._is_anthropic_oauth
                and not _retry.oauth_1m_beta_retry_attempted
            ):
                _retry.oauth_1m_beta_retry_attempted = True
                if not getattr(agent, "_oauth_1m_beta_disabled", False):
                    agent._oauth_1m_beta_disabled = True
                    try:
                        agent._anthropic_client.close()
                    except Exception:
                        pass
                    agent._rebuild_anthropic_client()
                    agent._vprint(
                        f"{agent.log_prefix}🔕 OAuth subscription doesn't support "
                        f"the 1M-context beta — disabled for this session and retrying...",
                        force=True,
                    )
                    continue

            if (
                agent.api_mode == "codex_responses"
                and agent.provider in {"openai-codex", "xai-oauth"}
                and status_code == 401
                and not _retry.codex_auth_retry_attempted
            ):
                _retry.codex_auth_retry_attempted = True
                if agent._try_refresh_codex_client_credentials(force=True):
                    _label = "xAI OAuth" if agent.provider == "xai-oauth" else "Codex"
                    agent._buffer_vprint(f"🔐 {_label} auth refreshed after 401. Retrying request...")
                    continue
            if (
                agent.api_mode == "chat_completions"
                and agent.provider == "nous"
                and status_code == 401
                and not _retry.nous_auth_retry_attempted
            ):
                _retry.nous_auth_retry_attempted = True
                if agent._try_refresh_nous_client_credentials(force=True):
                    print(f"{agent.log_prefix}🔐 Nous agent key refreshed after 401. Retrying request...")
                    continue
                # Credential refresh didn't help — show diagnostic info.
                # Most common causes: Portal OAuth expired/revoked,
                # account out of credits, or agent key blocked.
                from hermes_constants import display_hermes_home as _dhh_fn
                _dhh = _dhh_fn()
                _body_text = ""
                try:
                    _body = getattr(api_error, "body", None) or getattr(api_error, "response", None)
                    if _body is not None:
                        _body_text = str(_body)[:200]
                except Exception:
                    pass
                print(f"{agent.log_prefix}🔐 Nous 401 — Portal authentication failed.")
                if _body_text:
                    print(f"{agent.log_prefix}   Response: {_body_text}")
                if not _print_nous_entitlement_guidance(agent, "Nous model access"):
                    print(f"{agent.log_prefix}   Most likely: Portal OAuth expired, account out of credits, or agent key revoked.")
                print(f"{agent.log_prefix}   Troubleshooting:")
                print(f"{agent.log_prefix}     • Re-authenticate: hermes auth add nous")
                print(f"{agent.log_prefix}     • Check credits / billing: https://portal.nousresearch.com")
                print(f"{agent.log_prefix}     • Verify stored credentials: {_dhh}/auth.json")
                print(f"{agent.log_prefix}     • Switch providers temporarily: /model <model> --provider openrouter")
            if (
                agent.provider == "copilot"
                and status_code == 401
                and not _retry.copilot_auth_retry_attempted
            ):
                _retry.copilot_auth_retry_attempted = True
                if agent._try_refresh_copilot_client_credentials():
                    agent._buffer_vprint(f"🔐 Copilot credentials refreshed after 401. Retrying request...")
                    continue
            if (
                agent.api_mode == "anthropic_messages"
                and status_code == 401
                and hasattr(agent, '_anthropic_api_key')
                and not _retry.anthropic_auth_retry_attempted
            ):
                _retry.anthropic_auth_retry_attempted = True
                from agent.anthropic_adapter import _is_oauth_token
                from agent.azure_identity_adapter import is_token_provider
                if agent._try_refresh_anthropic_client_credentials():
                    print(f"{agent.log_prefix}🔐 Anthropic credentials refreshed after 401. Retrying request...")
                    continue
                # Credential refresh didn't help — show diagnostic info
                key = agent._anthropic_api_key
                print(f"{agent.log_prefix}🔐 Anthropic 401 — authentication failed.")
                if is_token_provider(key):
                    # Azure Foundry Entra ID — the bearer token is
                    # minted per-request by an httpx event hook on a
                    # custom http_client passed to the SDK. The 401
                    # means Azure rejected the JWT (RBAC role missing,
                    # az login expired, IMDS unreachable, etc.).
                    print(f"{agent.log_prefix}   Auth method: Microsoft Entra ID (httpx event hook)")
                    print(f"{agent.log_prefix}   Run `hermes doctor` for credential-chain diagnostics, or")
                    print(f"{agent.log_prefix}   `az login` if your developer session expired.")
                else:
                    auth_method = "Bearer (OAuth/setup-token)" if _is_oauth_token(key) else "x-api-key (API key)"
                    print(f"{agent.log_prefix}   Auth method: {auth_method}")
                    print(f"{agent.log_prefix}   Token prefix: {key[:12]}..." if isinstance(key, str) and len(key) > 12 else f"{agent.log_prefix}   Token: (empty or short)")
                print(f"{agent.log_prefix}   Troubleshooting:")
                from hermes_constants import display_hermes_home as _dhh_fn
                _dhh = _dhh_fn()
                print(f"{agent.log_prefix}     • Check ANTHROPIC_TOKEN in {_dhh}/.env for Hermes-managed OAuth/setup tokens")
                print(f"{agent.log_prefix}     • Check ANTHROPIC_API_KEY in {_dhh}/.env for API keys or legacy token values")
                print(f"{agent.log_prefix}     • For API keys: verify at https://platform.claude.com/settings/keys")
                print(f"{agent.log_prefix}     • For Claude Code: run 'claude /login' to refresh, then retry")
                print(f"{agent.log_prefix}     • Legacy cleanup: hermes config set ANTHROPIC_TOKEN \"\"")
                print(f"{agent.log_prefix}     • Clear stale keys: hermes config set ANTHROPIC_API_KEY \"\"")

            # Thinking block signature recovery.
            #
            # Anthropic signs thinking blocks against the full turn
            # content. Any upstream mutation (context compression,
            # session truncation, message merging) invalidates the
            # signature and the API replies HTTP 400 ("invalid
            # signature" or "cannot be modified"). Recovery strips
            # ``reasoning_details`` so the retry sends no thinking
            # blocks at all. One-shot per outer loop.
            #
            # The strip targets ``api_messages``, which is the
            # API-call-time list that ``_build_api_kwargs`` consumes
            # on every retry. ``api_messages`` was populated once at
            # the start of the turn from shallow copies of
            # ``messages``, so mutating it does not touch the
            # canonical store. The previous implementation popped
            # ``reasoning_details`` from ``messages`` instead, which
            # had two problems: ``api_messages`` carried its own
            # reference to the field through the shallow copy, so the
            # retry's wire payload still included thinking blocks and
            # the recovery never reached the API; and the mutation
            # persisted into ``state.db`` through any subsequent
            # ``_persist_session`` call, permanently corrupting the
            # conversation. Future turns would replay the stripped
            # state, hit the same 400, and the agent would terminate
            # with ``max_retries_exhausted``, often spawning
            # cascading compaction-ended sessions chained off the
            # corrupted parent.
            if (
                classified.reason == FailoverReason.thinking_signature
                and not _retry.thinking_sig_retry_attempted
            ):
                _retry.thinking_sig_retry_attempted = True
                _api_stripped = 0
                for _m in api_messages:
                    if isinstance(_m, dict) and "reasoning_details" in _m:
                        _m.pop("reasoning_details", None)
                        _api_stripped += 1
                agent._vprint(
                    f"{agent.log_prefix}⚠️  Thinking block signature invalid, "
                    f"stripped reasoning_details from api_messages for retry...",
                    force=True,
                )
                logger.warning(
                    "%sThinking block signature recovery: stripped "
                    "reasoning_details from %d api_messages "
                    "(canonical messages unchanged)",
                    agent.log_prefix, _api_stripped,
                )
                continue

            # ── Invalid encrypted reasoning replay recovery ───────
            # OpenAI Responses API surfaces (and some compatible relays)
            # return HTTP 400 ``invalid_encrypted_content`` when a
            # replayed ``codex_reasoning_items`` blob from a previous
            # turn fails verification (provider rotated the encryption
            # key, the route doesn't actually persist reasoning state,
            # etc.).  Recovery: disable replay for the rest of the
            # session, strip cached items from history, retry once.
            # One-shot — if a second 400 fires we fall through to the
            # normal retry/backoff path.  Only fires for codex_responses
            # mode with at least one assistant message that has cached
            # ``codex_reasoning_items``; without replay state, the
            # error is unrelated to our cache so the normal retry path
            # handles it (the provider is rejecting something else).
            if (
                classified.reason == FailoverReason.invalid_encrypted_content
                and not _retry.invalid_encrypted_content_retry_attempted
                and agent.api_mode == "codex_responses"
                and bool(getattr(agent, "_codex_reasoning_replay_enabled", True))
                and any(
                    isinstance(_m, dict)
                    and _m.get("role") == "assistant"
                    and isinstance(_m.get("codex_reasoning_items"), list)
                    and _m.get("codex_reasoning_items")
                    for _m in messages
                )
            ):
                _retry.invalid_encrypted_content_retry_attempted = True
                replay_stats = agent._disable_codex_reasoning_replay(messages)
                agent._vprint(
                    f"{agent.log_prefix}⚠️  Encrypted reasoning replay was rejected by the provider — "
                    f"disabled replay and stripped {replay_stats['items']} item(s) from "
                    f"{replay_stats['messages']} message(s), retrying...",
                    force=True,
                )
                logger.warning(
                    "%sInvalid encrypted reasoning recovery: disabled replay and stripped %d items from %d messages",
                    agent.log_prefix,
                    replay_stats["items"],
                    replay_stats["messages"],
                )
                continue

            # ── llama.cpp grammar-parse recovery ──────────────────
            # llama.cpp's ``json-schema-to-grammar`` converter rejects
            # regex escape classes (``\d``, ``\w``, ``\s``) and most
            # ``format`` values in tool schemas.  MCP servers emit
            # these routinely for date/phone/email params.  Recovery:
            # strip ``pattern``/``format`` from ``agent.tools`` and
            # retry once.  We keep the keywords by default so cloud
            # providers get the full prompting hints; this branch
            # fires only for users on llama.cpp's OAI server.
            if (
                classified.reason == FailoverReason.llama_cpp_grammar_pattern
                and not _retry.llama_cpp_grammar_retry_attempted
            ):
                _retry.llama_cpp_grammar_retry_attempted = True
                try:
                    from tools.schema_sanitizer import strip_pattern_and_format
                    _, _stripped = strip_pattern_and_format(agent.tools)
                except Exception as _strip_exc:  # pragma: no cover — defensive
                    logger.warning(
                        "%sllama.cpp grammar recovery: strip helper failed: %s",
                        agent.log_prefix, _strip_exc,
                    )
                    _stripped = 0
                if _stripped:
                    agent._vprint(
                        f"{agent.log_prefix}⚠️  llama.cpp rejected tool schema grammar — "
                        f"stripped {_stripped} pattern/format keyword(s), retrying...",
                        force=True,
                    )
                    logger.warning(
                        "%sllama.cpp grammar recovery: stripped %d "
                        "pattern/format keyword(s) from tool schemas",
                        agent.log_prefix, _stripped,
                    )
                    continue
                # No keywords found to strip — fall through to normal
                # retry path rather than loop forever on the same error.
                logger.warning(
                    "%sllama.cpp grammar error but no pattern/format "
                    "keywords to strip — falling through to normal retry",
                    agent.log_prefix,
                )

            retry_count += 1
            elapsed_time = time.time() - api_start_time
            agent._touch_activity(
                f"API error recovery (attempt {retry_count}/{max_retries})"
            )

            error_type = type(api_error).__name__
            error_msg = str(api_error).lower()
            _error_summary = agent._summarize_api_error(api_error)
            logger.warning(
                "API call failed (attempt %s/%s) error_type=%s %s summary=%s",
                retry_count,
                max_retries,
                error_type,
                agent._client_log_context(),
                _error_summary,
            )

            _provider = getattr(agent, "provider", "unknown")
            _base = getattr(agent, "base_url", "unknown")
            _model = getattr(agent, "model", "unknown")
            _status_code_str = f" [HTTP {status_code}]" if status_code else ""
            agent._buffer_vprint(f"⚠️  API call failed (attempt {retry_count}/{max_retries}): {error_type}{_status_code_str}")
            agent._buffer_vprint(f"   🔌 Provider: {_provider}  Model: {_model}")
            agent._buffer_vprint(f"   🌐 Endpoint: {_base}")
            agent._buffer_vprint(f"   📝 Error: {_error_summary}")
            if status_code and status_code < 500:
                _err_body = getattr(api_error, "body", None)
                _err_body_str = str(_err_body)[:300] if _err_body else None
                if _err_body_str:
                    agent._buffer_vprint(f"   📋 Details: {_err_body_str}")
            agent._buffer_vprint(f"   ⏱️  Elapsed: {elapsed_time:.2f}s  Context: {len(api_messages)} msgs, ~{approx_tokens:,} tokens")

            # Actionable hint for OpenRouter "no tool endpoints" error.
            # Buffered like the rest of the retry trace — surfaced only
            # if every retry+fallback exhausts.  Avoids spamming users
            # who recover automatically via fallback.
            if (
                agent._is_openrouter_url()
                and "support tool use" in error_msg
            ):
                agent._buffer_vprint(
                    f"   💡 No OpenRouter providers for {_model} support tool calling with your current settings."
                )
                if agent.providers_allowed:
                    agent._buffer_vprint(
                        f"      Your provider_routing.only restriction is filtering out tool-capable providers."
                    )
                    agent._buffer_vprint(
                        f"      Try removing the restriction or adding providers that support tools for this model."
                    )
                agent._buffer_vprint(
                    f"      Check which providers support tools: https://openrouter.ai/models/{_model}"
                )

            # Check for interrupt before deciding to retry
            if agent._interrupt_requested:
                agent._vprint(f"{agent.log_prefix}⚡ Interrupt detected during error handling, aborting retries.", force=True)
                agent._persist_session(messages, conversation_history)
                agent.clear_interrupt()
                return _terminal({
                    "final_response": f"Operation interrupted: handling API error ({error_type}: {agent._clean_error_message(str(api_error))}).",
                    "messages": messages,
                    "api_calls": api_call_count,
                    "completed": False,
                    "interrupted": True,
                })

            # Check for 413 payload-too-large BEFORE generic 4xx handler.
            # A 413 is a payload-size error — the correct response is to
            # compress history and retry, not abort immediately.
            status_code = getattr(api_error, "status_code", None)

            # ── Respect disabled auto-compaction on overflow ──────
            # Ported from anomalyco/opencode#30749.  When the user has
            # turned auto-compaction off (``compression.enabled: false``),
            # NO automatic compaction trigger may fire — including the
            # provider/request-size overflow recovery paths below
            # (long-context-tier 429, 413 payload-too-large, and
            # context-overflow).  Without this guard the proactive
            # threshold path correctly honours the setting (see the
            # preflight check and the post-response ``should_compress``
            # gate) but a provider overflow error would still silently
            # compress + rotate the session, bypassing the user's
            # explicit choice.  Surface a terminal error instead so the
            # user can compact manually (``/compress``), start fresh
            # (``/new``), switch to a larger-context model, or reduce
            # attachments.  Forced compaction via ``/compress``
            # (``force=True``) is unaffected — it never reaches this loop.
            _overflow_reasons = {
                FailoverReason.long_context_tier,
                FailoverReason.payload_too_large,
                FailoverReason.context_overflow,
            }
            if (
                classified.reason in _overflow_reasons
                and not getattr(agent, "compression_enabled", True)
            ):
                agent._flush_status_buffer()
                agent._vprint(
                    f"{agent.log_prefix}❌ Context overflow, but auto-compaction is disabled "
                    f"(compression.enabled: false).",
                    force=True,
                )
                agent._vprint(
                    f"{agent.log_prefix}   💡 Run /compress to compact manually, /new to start fresh, "
                    f"switch to a larger-context model, or reduce attachments.",
                    force=True,
                )
                logger.error(
                    f"{agent.log_prefix}Context overflow ({classified.reason.value}) with "
                    f"auto-compaction disabled — not compressing."
                )
                agent._persist_session(messages, conversation_history)
                return _terminal({
                    "messages": messages,
                    "completed": False,
                    "api_calls": api_call_count,
                    "error": (
                        "Context overflow and auto-compaction is disabled "
                        "(compression.enabled: false). Run /compress to compact manually, "
                        "/new to start fresh, or switch to a larger-context model."
                    ),
                    "partial": True,
                    "failed": True,
                    "compaction_disabled": True,
                })

            # ── Anthropic Sonnet long-context tier gate ───────────
            # Anthropic returns HTTP 429 "Extra usage is required for
            # long context requests" when a Claude Max (or similar)
            # subscription doesn't include the 1M-context tier.  This
            # is NOT a transient rate limit — retrying or switching
            # credentials won't help.  Reduce context to 200k (the
            # standard tier) and compress.
            if classified.reason == FailoverReason.long_context_tier:
                _reduced_ctx = 200000
                compressor = agent.context_compressor
                old_ctx = compressor.context_length
                if old_ctx > _reduced_ctx:
                    compressor.update_model(
                        model=agent.model,
                        context_length=_reduced_ctx,
                        base_url=agent.base_url,
                        api_key=getattr(agent, "api_key", ""),
                        provider=agent.provider,
                        api_mode=agent.api_mode,
                    )
                    # Context probing flags — only set on built-in
                    # compressor (plugin engines manage their own).
                    if hasattr(compressor, "_context_probed"):
                        compressor._context_probed = True
                        # Don't persist — this is a subscription-tier
                        # limitation, not a model capability.  If the
                        # user later enables extra usage the 1M limit
                        # should come back automatically.
                        compressor._context_probe_persistable = False
                    agent._buffer_vprint(
                        f"⚠️  Anthropic long-context tier "
                        f"requires extra usage — reducing context: "
                        f"{old_ctx:,} → {_reduced_ctx:,} tokens"
                    )

                compression_attempts += 1
                if compression_attempts <= max_compression_attempts:
                    original_len = len(messages)
                    messages, active_system_prompt, conversation_history = apply_turn_compression(
                        agent, messages, system_message,
                        approx_tokens=approx_tokens, task_id=effective_task_id,
                    )
                    if len(messages) < original_len or old_ctx > _reduced_ctx:
                        agent._buffer_status(
                            f"🗜️ Context reduced to {_reduced_ctx:,} tokens "
                            f"(was {old_ctx:,}), retrying..."
                        )
                        time.sleep(2)
                        _retry.restart_with_compressed_messages = True
                        break
                # Fall through to normal error handling if compression
                # is exhausted or didn't help.

            # Eager fallback for rate-limit errors (429 or quota exhaustion).
            # When a fallback model is configured, switch immediately instead
            # of burning through retries with exponential backoff -- the
            # primary provider won't recover within the retry window.
            is_rate_limited = classified.reason in {
                FailoverReason.rate_limit,
                FailoverReason.billing,
            }
            if is_rate_limited and agent._fallback_index < len(agent._fallback_chain):
                # Don't eagerly fallback if credential pool rotation may
                # still recover.  See _pool_may_recover_from_rate_limit
                # for the single-credential-pool and CloudCode-quota
                # exceptions.  Fixes #11314 and #13636.
                pool_may_recover = _ra()._pool_may_recover_from_rate_limit(
                    agent._credential_pool,
                    provider=agent.provider,
                    base_url=getattr(agent, "base_url", None),
                )
                if not pool_may_recover:
                    if classified.reason == FailoverReason.billing:
                        agent._buffer_status(
                            "⚠️ Billing or credits exhausted — switching to fallback provider..."
                        )
                    else:
                        agent._buffer_status("⚠️ Rate limited — switching to fallback provider...")
                    if agent._try_activate_fallback(reason=classified.reason):
                        retry_count = 0
                        compression_attempts = 0
                        _retry.primary_recovery_attempted = False
                        continue

            # ── Nous Portal: record rate limit & skip retries ─────
            # When Nous returns a 429 that is a genuine account-
            # level rate limit, record the reset time to a shared
            # file so ALL sessions (cron, gateway, auxiliary) know
            # not to pile on, then skip further retries -- each
            # one burns another RPH request and deepens the hole.
            # The retry loop's top-of-iteration guard will catch
            # this on the next pass and try fallback or bail.
            #
            # IMPORTANT: Nous Portal multiplexes multiple upstream
            # providers (DeepSeek, Kimi, MiMo, Hermes).  A 429 can
            # also mean an UPSTREAM provider is out of capacity
            # for one specific model -- transient, clears in
            # seconds, nothing to do with the caller's quota.
            # Tripping the cross-session breaker on that would
            # block every Nous model for minutes.  We use
            # ``is_genuine_nous_rate_limit`` to tell the two
            # apart via the 429's own x-ratelimit-* headers and
            # the last-known-good state captured on the previous
            # successful response.
            if (
                is_rate_limited
                and agent.provider == "nous"
                and classified.reason == FailoverReason.rate_limit
                and not recovered_with_pool
            ):
                _genuine_nous_rate_limit = False
                try:
                    from agent.nous_rate_guard import (
                        is_genuine_nous_rate_limit,
                        record_nous_rate_limit,
                    )
                    _err_resp = getattr(api_error, "response", None)
                    _err_hdrs = (
                        getattr(_err_resp, "headers", None)
                        if _err_resp else None
                    )
                    _genuine_nous_rate_limit = is_genuine_nous_rate_limit(
                        headers=_err_hdrs,
                        last_known_state=agent._rate_limit_state,
                    )
                    if _genuine_nous_rate_limit:
                        record_nous_rate_limit(
                            headers=_err_hdrs,
                            error_context=error_context,
                        )
                    else:
                        logger.info(
                            "Nous 429 looks like upstream capacity "
                            "(no exhausted bucket in headers or "
                            "last-known state) -- not tripping "
                            "cross-session breaker."
                        )
                except Exception:
                    pass
                if _genuine_nous_rate_limit:
                    # Re-enter the loop exactly once so the
                    # top-of-loop Nous guard handles fallback or
                    # bails cleanly. (Setting retry_count to
                    # max_retries would make the while condition
                    # false immediately and the guard would never
                    # run -- no fallback, generic exhaustion error.)
                    retry_count = max(0, max_retries - 1)
                    continue
                # Upstream capacity 429: fall through to normal
                # retry logic.  A different model (or the same
                # model a moment later) will typically succeed.

            is_payload_too_large = (
                classified.reason == FailoverReason.payload_too_large
            )

            # Actionable hint for GitHub Models (Azure) 413 errors.
            # The free tier enforces a hard 8K token cap per request,
            # which Hermes' system prompt + tool schemas alone exceed.
            # Compression can't help — the floor is the system prompt
            # itself, not the conversation — so surface a clear "not
            # compatible" message instead of looping into three futile
            # compression attempts.
            if (
                status_code == 413
                and isinstance(agent.base_url, str)
                and "models.inference.ai.azure.com" in agent.base_url
            ):
                agent._vprint(
                    f"{agent.log_prefix}   💡 GitHub Models free tier (models.inference.ai.azure.com) caps every",
                    force=True,
                )
                agent._vprint(
                    f"{agent.log_prefix}      request at ~8K tokens. Hermes' system prompt + tool schemas baseline",
                    force=True,
                )
                agent._vprint(
                    f"{agent.log_prefix}      exceeds that floor, so this endpoint cannot run an agentic loop.",
                    force=True,
                )
                agent._vprint(
                    f"{agent.log_prefix}      Use the `copilot` provider with a Copilot subscription token (`hermes",
                    force=True,
                )
                agent._vprint(
                    f"{agent.log_prefix}      setup` → GitHub Copilot), or pick any other provider.",
                    force=True,
                )

            if is_payload_too_large:
                compression_attempts += 1
                if compression_attempts > max_compression_attempts:
                    # Terminal — surface the buffered retry trace.
                    agent._flush_status_buffer()
                    agent._vprint(f"{agent.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached for payload-too-large error.", force=True)
                    agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                    logger.error(f"{agent.log_prefix}413 compression failed after {max_compression_attempts} attempts.")
                    agent._persist_session(messages, conversation_history)
                    return _terminal({
                        "messages": messages,
                        "completed": False,
                        "api_calls": api_call_count,
                        "error": f"Request payload too large: max compression attempts ({max_compression_attempts}) reached.",
                        "partial": True,
                        "failed": True,
                        "compression_exhausted": True,
                    })
                agent._buffer_status(f"⚠️  Request payload too large (413) — compression attempt {compression_attempts}/{max_compression_attempts}...")

                original_len = len(messages)
                messages, active_system_prompt, conversation_history = apply_turn_compression(
                    agent, messages, system_message,
                    approx_tokens=approx_tokens, task_id=effective_task_id,
                )

                if len(messages) < original_len:
                    agent._buffer_status(f"🗜️ Compressed {original_len} → {len(messages)} messages, retrying...")
                    time.sleep(2)  # Brief pause between compression retries
                    _retry.restart_with_compressed_messages = True
                    break
                else:
                    # Terminal — surface buffered context so the user
                    # sees what compression attempts were made.
                    agent._flush_status_buffer()
                    agent._vprint(f"{agent.log_prefix}❌ Payload too large and cannot compress further.", force=True)
                    agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                    logger.error(f"{agent.log_prefix}413 payload too large. Cannot compress further.")
                    agent._persist_session(messages, conversation_history)
                    return _terminal({
                        "messages": messages,
                        "completed": False,
                        "api_calls": api_call_count,
                        "error": "Request payload too large (413). Cannot compress further.",
                        "partial": True,
                        "failed": True,
                        "compression_exhausted": True,
                    })

            # Check for context-length errors BEFORE generic 4xx handler.
            # The classifier detects context overflow from: explicit error
            # messages, generic 400 + large session heuristic (#1630), and
            # server disconnect + large session pattern (#2153).
            is_context_length_error = (
                classified.reason == FailoverReason.context_overflow
            )

            if is_context_length_error:
                compressor = agent.context_compressor
                old_ctx = compressor.context_length

                # ── Distinguish two very different errors ───────────
                # 1. "Prompt too long": the INPUT exceeds the context window.
                #    Fix: reduce context_length + compress history.
                # 2. "max_tokens too large": input is fine, but
                #    input_tokens + requested max_tokens > context_window.
                #    Fix: reduce max_tokens (the OUTPUT cap) for this call.
                #    Do NOT shrink context_length — the window is unchanged.
                #
                # Note: max_tokens = output token cap (one response).
                #       context_length = total window (input + output combined).
                available_out = parse_available_output_tokens_from_error(error_msg)
                if available_out is not None:
                    # Error is purely about the output cap being too large.
                    # Cap output to the available space and retry without
                    # touching context_length or triggering compression.
                    safe_out = max(1, available_out - 64)  # small safety margin
                    agent._ephemeral_max_output_tokens = safe_out
                    agent._buffer_vprint(
                        f"⚠️  Output cap too large for current prompt — "
                        f"retrying with max_tokens={safe_out:,} "
                        f"(available_tokens={available_out:,}; context_length unchanged at {old_ctx:,})"
                    )
                    # Still count against compression_attempts so we don't
                    # loop forever if the error keeps recurring.
                    compression_attempts += 1
                    if compression_attempts > max_compression_attempts:
                        agent._flush_status_buffer()
                        agent._vprint(f"{agent.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                        agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                        logger.error(f"{agent.log_prefix}Context compression failed after {max_compression_attempts} attempts.")
                        agent._persist_session(messages, conversation_history)
                        return _terminal({
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached.",
                            "partial": True,
                            "failed": True,
                            "compression_exhausted": True,
                        })
                    _retry.restart_with_compressed_messages = True
                    break

                # Error is about the INPUT being too large.  Only reduce
                # context_length when the provider explicitly reports the
                # real lower limit.  If the provider only says "input
                # exceeds the context window", keep the configured window
                # and try compression; guessing probe tiers can incorrectly
                # turn a user-configured 1M window into 256K/128K/64K.
                new_ctx = get_context_length_from_provider_error(error_msg, old_ctx)
                _provider_lower = (getattr(agent, "provider", "") or "").lower()
                _base_lower = (getattr(agent, "base_url", "") or "").rstrip("/").lower()
                is_minimax_provider = (
                    _provider_lower in {"minimax", "minimax-cn"}
                    or _base_lower.startswith((
                        "https://api.minimax.io/anthropic",
                        "https://api.minimaxi.com/anthropic",
                    ))
                )
                minimax_delta_only_overflow = (
                    is_minimax_provider
                    and new_ctx is None
                    and "context window exceeds limit (" in error_msg
                )

                if new_ctx is not None:
                    agent._buffer_vprint(f"Context limit detected from API: {new_ctx:,} tokens (was {old_ctx:,})")
                    compressor.update_model(
                        model=agent.model,
                        context_length=new_ctx,
                        base_url=agent.base_url,
                        api_key=getattr(agent, "api_key", ""),
                        provider=agent.provider,
                        api_mode=agent.api_mode,
                    )
                    # Context probing flags — only set on built-in
                    # compressor (plugin engines manage their own).  This
                    # value came from the provider, so it is safe to cache.
                    if hasattr(compressor, "_context_probed"):
                        compressor._context_probed = True
                        compressor._context_probe_persistable = True
                    agent._buffer_vprint(f"⚠️  Context length exceeded — using provider limit: {old_ctx:,} → {new_ctx:,} tokens")
                elif minimax_delta_only_overflow:
                    agent._buffer_vprint(
                        f"Provider reported overflow amount only; "
                        f"keeping context_length at {old_ctx:,} tokens and compressing."
                    )
                else:
                    agent._buffer_vprint(
                        f"⚠️  Context length exceeded, but provider did not report a max context length; "
                        f"keeping context_length at {old_ctx:,} tokens and compressing."
                    )

                compression_attempts += 1
                if compression_attempts > max_compression_attempts:
                    agent._flush_status_buffer()
                    agent._vprint(f"{agent.log_prefix}❌ Max compression attempts ({max_compression_attempts}) reached.", force=True)
                    agent._vprint(f"{agent.log_prefix}   💡 Try /new to start a fresh conversation, or /compress to retry compression.", force=True)
                    logger.error(f"{agent.log_prefix}Context compression failed after {max_compression_attempts} attempts.")
                    agent._persist_session(messages, conversation_history)
                    return _terminal({
                        "messages": messages,
                        "completed": False,
                        "api_calls": api_call_count,
                        "error": f"Context length exceeded: max compression attempts ({max_compression_attempts}) reached.",
                        "partial": True,
                        "failed": True,
                        "compression_exhausted": True,
                    })
                agent._buffer_status(f"🗜️ Context too large (~{approx_tokens:,} tokens) — compressing ({compression_attempts}/{max_compression_attempts})...")

                original_len = len(messages)
                messages, active_system_prompt, conversation_history = apply_turn_compression(
                    agent, messages, system_message,
                    approx_tokens=approx_tokens, task_id=effective_task_id,
                )

                if len(messages) < original_len or new_ctx and new_ctx < old_ctx:
                    if len(messages) < original_len:
                        agent._buffer_status(f"🗜️ Compressed {original_len} → {len(messages)} messages, retrying...")
                    time.sleep(2)  # Brief pause between compression retries
                    _retry.restart_with_compressed_messages = True
                    break
                else:
                    # Can't compress further and already at minimum tier
                    agent._flush_status_buffer()
                    agent._vprint(f"{agent.log_prefix}❌ Context length exceeded and cannot compress further.", force=True)
                    agent._vprint(f"{agent.log_prefix}   💡 The conversation has accumulated too much content. Try /new to start fresh, or /compress to manually trigger compression.", force=True)
                    logger.error(f"{agent.log_prefix}Context length exceeded: {approx_tokens:,} tokens. Cannot compress further.")
                    agent._persist_session(messages, conversation_history)
                    return _terminal({
                        "messages": messages,
                        "completed": False,
                        "api_calls": api_call_count,
                        "error": f"Context length exceeded ({approx_tokens:,} tokens). Cannot compress further.",
                        "partial": True,
                        "failed": True,
                        "compression_exhausted": True,
                    })

            # Check for non-retryable client errors.  The classifier
            # already accounts for 413, 429, 529 (transient), context
            # overflow, and generic-400 heuristics.  Local validation
            # errors (ValueError, TypeError) are programming bugs.
            # Exclude UnicodeEncodeError — it's a ValueError subclass
            # but is handled separately by the surrogate sanitization
            # path above.  Exclude json.JSONDecodeError — also a
            # ValueError subclass, but it indicates a transient
            # provider/network failure (malformed response body,
            # truncated stream, routing layer corruption), not a
            # local programming bug, and should be retried (#14782).
            is_local_validation_error = (
                isinstance(api_error, (ValueError, TypeError))
                and not isinstance(
                    api_error, (UnicodeEncodeError, json.JSONDecodeError)
                )
                # ssl.SSLError (and its subclass SSLCertVerificationError)
                # inherits from OSError *and* ValueError via Python MRO,
                # so the isinstance(ValueError) check above would
                # misclassify a TLS transport failure as a local
                # programming bug and abort without retrying.  Exclude
                # ssl.SSLError explicitly so the error classifier's
                # retryable=True mapping takes effect instead.
                and not isinstance(api_error, ssl.SSLError)
                # Provider/SDK "NoneType is not iterable" failures are
                # shape mismatches from upstream (e.g. chatgpt.com Codex
                # backend response.completed.output=null) — not local
                # programming bugs.  Even after #33042 made our own
                # consumer immune, third-party shims and mocked clients
                # can still surface this shape via TypeError.  Treat
                # them as retryable so the error classifier's normal
                # retry/fallback path runs instead of killing the turn
                # as non-retryable (which left Telegram users staring
                # at a bare "Non-retryable error" with no recovery).
                and not (
                    isinstance(api_error, TypeError)
                    and "nonetype" in str(api_error).lower()
                    and "not iterable" in str(api_error).lower()
                )
            )
            # ``FailoverReason.billing`` (HTTP 402) is NOT in this
            # exclusion set.  By the time we reach this block:
            #   • credential-pool rotation (line ~2031) has already
            #     fired for billing and either ``continue``d or
            #     returned (False, ...) — pool is exhausted or absent.
            #   • the eager-fallback branch above (line ~2422) also
            #     fires on billing and ``continue``s if a fallback
            #     provider is configured.
            # Falling through to here means BOTH recovery paths
            # gave up.  Treating 402 as retryable from this point
            # just burns more paid requests against a depleted
            # balance with no recovery mechanism left — see #31273
            # (real-world: ~$40 in 48h on a 24/7 gateway).  Aborting
            # mirrors how 401/403 (also ``should_fallback=True``)
            # already behave once their recovery paths have failed.
            is_client_error = (
                is_local_validation_error
                or (
                    not classified.retryable
                    and not classified.should_compress
                    and classified.reason not in {
                        FailoverReason.rate_limit,
                        FailoverReason.overloaded,
                        FailoverReason.context_overflow,
                        FailoverReason.payload_too_large,
                        FailoverReason.long_context_tier,
                        FailoverReason.thinking_signature,
                    }
                )
            ) and not is_context_length_error

            if is_client_error:
                # Try fallback before aborting — a different provider may
                # not have the same issue (rate limit, auth, etc.). Only
                # announce the attempt when a fallback chain actually
                # exists; otherwise "trying fallback..." is a lie and the
                # session looks like it's recovering when it's about to
                # abort silently (#35314, #17446).
                if agent._has_pending_fallback():
                    if classified.reason == FailoverReason.content_policy_blocked:
                        agent._buffer_status("⚠️ Provider safety filter blocked this request — trying fallback...")
                    else:
                        agent._buffer_status(f"⚠️ Non-retryable error (HTTP {status_code}) — trying fallback...")
                if agent._try_activate_fallback():
                    retry_count = 0
                    compression_attempts = 0
                    _retry.primary_recovery_attempted = False
                    continue
                if api_kwargs is not None:
                    agent._dump_api_request_debug(
                        api_kwargs, reason="non_retryable_client_error", error=api_error,
                    )
                # Terminal — flush buffered context so the user sees
                # what was tried before the abort.
                agent._flush_status_buffer()
                if classified.reason == FailoverReason.content_policy_blocked:
                    agent._emit_status(
                        f"❌ Provider safety filter blocked this request: "
                        f"{agent._summarize_api_error(api_error)}"
                    )
                else:
                    agent._emit_status(
                        f"❌ Non-retryable error (HTTP {status_code}): "
                        f"{agent._summarize_api_error(api_error)}"
                    )
                agent._vprint(f"{agent.log_prefix}❌ Non-retryable client error (HTTP {status_code}). Aborting.", force=True)
                agent._vprint(f"{agent.log_prefix}   🔌 Provider: {_provider}  Model: {_model}", force=True)
                agent._vprint(f"{agent.log_prefix}   🌐 Endpoint: {_base}", force=True)
                # Actionable guidance for common auth errors
                if classified.is_auth or classified.reason == FailoverReason.billing:
                    if classified.reason == FailoverReason.billing and _print_billing_or_entitlement_guidance(
                        agent,
                        capability="model access",
                        provider=_provider,
                        base_url=str(_base),
                        model=_model,
                    ):
                        pass
                    elif _provider == "nous" and _print_nous_entitlement_guidance(
                        agent,
                        "Nous model access",
                    ):
                        pass
                    elif _provider in {"openai-codex", "xai-oauth", "nous"} and status_code == 401:
                        if _provider == "openai-codex":
                            agent._vprint(f"{agent.log_prefix}   💡 Codex OAuth token was rejected (HTTP 401). Your token may have been", force=True)
                            agent._vprint(f"{agent.log_prefix}      refreshed by another client (Codex CLI, VS Code). To fix:", force=True)
                            agent._vprint(f"{agent.log_prefix}      1. Run `codex` in your terminal to generate fresh tokens.", force=True)
                            agent._vprint(f"{agent.log_prefix}      2. Then run `hermes auth` to re-authenticate.", force=True)
                        elif _provider == "xai-oauth":
                            agent._vprint(f"{agent.log_prefix}   💡 xAI OAuth token was rejected (HTTP 401). To fix:", force=True)
                            agent._vprint(f"{agent.log_prefix}      re-authenticate with xAI Grok OAuth (SuperGrok / Premium+) from `hermes model`.", force=True)
                        else:  # nous
                            agent._vprint(f"{agent.log_prefix}   💡 Nous Portal OAuth token was rejected (HTTP 401). Your token may be", force=True)
                            agent._vprint(f"{agent.log_prefix}      expired, revoked, or your account may be out of credits. To fix:", force=True)
                            agent._vprint(f"{agent.log_prefix}      1. Re-authenticate: hermes portal", force=True)
                            agent._vprint(f"{agent.log_prefix}      2. Check your portal account: https://portal.nousresearch.com", force=True)
                            # ``:free`` is OpenRouter slug syntax; Nous Portal will reject
                            # the model name even after a successful re-auth.
                            if isinstance(_model, str) and _model.endswith(":free"):
                                agent._vprint(f"{agent.log_prefix}      ⚠️  Note: `{_model}` looks like an OpenRouter slug (`:free` suffix).", force=True)
                                agent._vprint(f"{agent.log_prefix}         Nous Portal won't recognize that model name. Either switch to a", force=True)
                                agent._vprint(f"{agent.log_prefix}         Nous catalog model, or run `/model openrouter:{_model}` to use OpenRouter.", force=True)
                    else:
                        agent._vprint(f"{agent.log_prefix}   💡 Your API key was rejected by the provider. Check:", force=True)
                        agent._vprint(f"{agent.log_prefix}      • Is the key valid? Run: hermes setup", force=True)
                        agent._vprint(f"{agent.log_prefix}      • Does your account have access to {_model}?", force=True)
                        if base_url_host_matches(str(_base), "openrouter.ai"):
                            agent._vprint(f"{agent.log_prefix}      • Check credits: https://openrouter.ai/settings/credits", force=True)
                else:
                    agent._vprint(f"{agent.log_prefix}   💡 This type of error won't be fixed by retrying.", force=True)
                # Content-policy blocks deserve their own actionable
                # guidance — neither "fix your API key" nor "retry won't
                # help" tells the user what to actually do. The provider
                # has refused this specific prompt, so the recovery is
                # either a rephrase or routing to a different model.
                if classified.reason == FailoverReason.content_policy_blocked:
                    agent._vprint(
                        f"{agent.log_prefix}   💡 The provider's safety filter rejected this specific prompt.",
                        force=True,
                    )
                    agent._vprint(
                        f"{agent.log_prefix}      • Try rephrasing the request, narrowing the context, or splitting into smaller steps.",
                        force=True,
                    )
                    agent._vprint(
                        f"{agent.log_prefix}      • Configure a fallback provider so future blocks route automatically:",
                        force=True,
                    )
                    agent._vprint(
                        f"{agent.log_prefix}        hermes fallback add   (interactive picker — same as `hermes model`)",
                        force=True,
                    )
                logger.error(f"{agent.log_prefix}Non-retryable client error: {api_error}")
                # Skip session persistence when the error is likely
                # context-overflow related (status 400 + large session).
                # Persisting the failed user message would make the
                # session even larger, causing the same failure on the
                # next attempt. (#1630)
                if status_code == 400 and (approx_tokens > 50000 or len(api_messages) > 80):
                    agent._vprint(
                        f"{agent.log_prefix}⚠️  Skipping session persistence "
                        f"for large failed session to prevent growth loop.",
                        force=True,
                    )
                else:
                    agent._persist_session(messages, conversation_history)
                if classified.reason == FailoverReason.content_policy_blocked:
                    _summary = agent._summarize_api_error(api_error)
                    _policy_response = (
                        "⚠️  The model provider's safety filter blocked this request "
                        "(not a Hermes/gateway failure).\n\n"
                        f"Provider message: {_summary}\n\n"
                        f"{_CONTENT_POLICY_RECOVERY_HINT}"
                    )
                    return _terminal(_content_policy_blocked_result(
                        messages,
                        api_call_count,
                        final_response=_policy_response,
                        error_detail=_summary,
                    ))
                return _terminal({
                    "final_response": None,
                    "messages": messages,
                    "api_calls": api_call_count,
                    "completed": False,
                    "failed": True,
                    "error": str(api_error),
                })

            if retry_count >= max_retries:
                # Before falling back, try rebuilding the primary
                # client once for transient transport errors (stale
                # connection pool, TCP reset).  Only attempted once
                # per API call block.
                if not _retry.primary_recovery_attempted and agent._try_recover_primary_transport(
                    api_error, retry_count=retry_count, max_retries=max_retries,
                ):
                    _retry.primary_recovery_attempted = True
                    retry_count = 0
                    continue
                # Try fallback before giving up entirely
                if agent._has_pending_fallback():
                    agent._buffer_status(f"⚠️ Max retries ({max_retries}) exhausted — trying fallback...")
                if agent._try_activate_fallback():
                    retry_count = 0
                    compression_attempts = 0
                    _retry.primary_recovery_attempted = False
                    continue
                # Terminal — flush buffered retry/fallback trace.
                agent._flush_status_buffer()
                _final_summary = agent._summarize_api_error(api_error)
                _billing_guidance = ""
                if classified.reason == FailoverReason.billing:
                    agent._emit_status(f"❌ Billing or credits exhausted — {_final_summary}")
                    _billing_guidance = _billing_or_entitlement_message(
                        capability="model access",
                        provider=_provider,
                        base_url=str(_base),
                        model=_model,
                    )
                    _print_billing_or_entitlement_guidance(
                        agent,
                        capability="model access",
                        provider=_provider,
                        base_url=str(_base),
                        model=_model,
                    )
                elif is_rate_limited:
                    agent._emit_status(f"❌ Rate limited after {max_retries} retries — {_final_summary}")
                else:
                    agent._emit_status(f"❌ API failed after {max_retries} retries — {_final_summary}")
                agent._vprint(f"{agent.log_prefix}   💀 Final error: {_final_summary}", force=True)

                # Detect SSE stream-drop pattern (e.g. "Network
                # connection lost") and surface actionable guidance.
                # This typically happens when the model generates a
                # very large tool call (write_file with huge content)
                # and the proxy/CDN drops the stream mid-response.
                _is_stream_drop = (
                    not getattr(api_error, "status_code", None)
                    and any(p in error_msg for p in (
                        "connection lost", "connection reset",
                        "connection closed", "network connection",
                        "network error", "terminated",
                    ))
                )
                if _is_stream_drop:
                    agent._vprint(
                        f"{agent.log_prefix}   💡 The provider's stream "
                        f"connection keeps dropping. This often happens "
                        f"when the model tries to write a very large "
                        f"file in a single tool call.",
                        force=True,
                    )
                    agent._vprint(
                        f"{agent.log_prefix}      Try asking the model "
                        f"to use execute_code with Python's open() for "
                        f"large files, or to write the file in smaller "
                        f"sections.",
                        force=True,
                    )

                logger.error(
                    "%sAPI call failed after %s retries. %s | provider=%s model=%s msgs=%s tokens=~%s",
                    agent.log_prefix, max_retries, _final_summary,
                    _provider, _model, len(api_messages), f"{approx_tokens:,}",
                )
                if api_kwargs is not None:
                    agent._dump_api_request_debug(
                        api_kwargs, reason="max_retries_exhausted", error=api_error,
                    )
                agent._persist_session(messages, conversation_history)
                if classified.reason == FailoverReason.billing:
                    _final_response = f"Billing or credits exhausted: {_final_summary}"
                    if _billing_guidance:
                        _final_response += f"\n\n{_billing_guidance}"
                else:
                    _final_response = f"API call failed after {max_retries} retries: {_final_summary}"
                if _is_stream_drop:
                    _final_response += (
                        "\n\nThe provider's stream connection keeps "
                        "dropping — this often happens when generating "
                        "very large tool call responses (e.g. write_file "
                        "with long content). Try asking me to use "
                        "execute_code with Python's open() for large "
                        "files, or to write in smaller sections."
                    )
                return _terminal({
                    "final_response": _final_response,
                    "messages": messages,
                    "api_calls": api_call_count,
                    "completed": False,
                    "failed": True,
                    "error": _final_summary,
                    # Surface the classified reason so callers (notably the
                    # kanban worker path in cli.py) can distinguish a
                    # transient throttle from a real failure and choose a
                    # different exit code. ``rate_limit`` / ``billing`` here
                    # mean "quota wall, not a task error".
                    "failure_reason": classified.reason.value,
                })

            # For rate limits, respect the Retry-After header if present
            _retry_after = None
            if is_rate_limited:
                _resp_headers = getattr(getattr(api_error, "response", None), "headers", None)
                if _resp_headers and hasattr(_resp_headers, "get"):
                    _ra_raw = _resp_headers.get("retry-after") or _resp_headers.get("Retry-After")
                    if _ra_raw:
                        try:
                            _retry_after = min(float(_ra_raw), 120)  # Cap at 2 minutes
                        except (TypeError, ValueError):
                            pass
            wait_time = _retry_after if _retry_after else jittered_backoff(retry_count, base_delay=2.0, max_delay=60.0)
            if is_rate_limited:
                agent._buffer_status(f"⏱️ Rate limited. Waiting {wait_time:.1f}s (attempt {retry_count + 1}/{max_retries})...")
            else:
                agent._buffer_status(f"⏳ Retrying in {wait_time:.1f}s (attempt {retry_count}/{max_retries})...")
            logger.warning(
                "Retrying API call in %ss (attempt %s/%s) %s error=%s",
                wait_time,
                retry_count,
                max_retries,
                agent._client_log_context(),
                api_error,
            )
            # Sleep in small increments so we can respond to interrupts quickly
            # instead of blocking the entire wait_time in one sleep() call
            sleep_end = time.time() + wait_time
            _backoff_touch_counter = 0
            while time.time() < sleep_end:
                if agent._interrupt_requested:
                    agent._vprint(f"{agent.log_prefix}⚡ Interrupt detected during retry wait, aborting.", force=True)
                    agent._persist_session(messages, conversation_history)
                    agent.clear_interrupt()
                    return _terminal({
                        "final_response": f"Operation interrupted: retrying API call after error (retry {retry_count}/{max_retries}).",
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "interrupted": True,
                    })
                time.sleep(0.2)  # Check interrupt every 200ms
                # Touch activity every ~30s so the gateway's inactivity
                # monitor knows we're alive during backoff waits.
                _backoff_touch_counter += 1
                if _backoff_touch_counter % 150 == 0:  # 150 × 0.2s = 30s
                    agent._touch_activity(
                        f"error retry backoff ({retry_count}/{max_retries}), "
                        f"{int(sleep_end - time.time())}s remaining"
                    )

    # Success ``break`` and natural while-exit fall through to here: not a
    # terminal result — hand the response (or None, for the caller's post-loop
    # restart / ``response is None`` guard) back to run_conversation.
    return ApiCallOutcome(
        terminal_result=None,
        response=response,
        messages=messages,
        active_system_prompt=active_system_prompt,
        conversation_history=conversation_history,
        api_call_count=api_call_count,
        retry_count=retry_count,
        length_continue_retries=length_continue_retries,
        compression_attempts=compression_attempts,
        truncated_tool_call_retries=truncated_tool_call_retries,
        finish_reason=finish_reason,
        api_kwargs=api_kwargs,
        api_duration=api_duration,
        final_response=final_response,
        interrupted=interrupted,
        failed=failed,
    )
