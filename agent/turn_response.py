"""Response-processing and tool-dispatch step for the agent turn loop.

Extracted from :func:`agent.conversation_loop.run_conversation` (god-file
decomposition; plan 010e). This module owns the *pure* response-handling
region: response normalization, incomplete-scratchpad / codex-incomplete
handling, tool-call validation + dispatch, compression, and the
no-tool-call / empty-response / final-response paths, including the
trailing ``except Exception`` outer-loop error handler.

The original inline region used ``break`` / ``continue`` / ``return`` to
steer the surrounding ``while`` loop in ``run_conversation``. Because a
helper function cannot drive its caller's loop, every WHILE-targeting exit
is converted to a :class:`TurnStep` carrying a :class:`TurnAction` plus the
full set of loop locals the caller must read back. Inner ``for``-loop
``break`` / ``continue`` statements are kept verbatim — they steer their own
loops, not the outer ``while``.

The compression path REBINDS ``messages`` / ``active_system_prompt`` and
sets ``conversation_history`` to ``None``, so by-reference mutation is
insufficient; those (and the other mutated locals) ride back on the
returned :class:`TurnStep`. All other mutated state lives on ``agent.*``
attributes, which persist across the call.

``run_agent`` is resolved lazily via :func:`agent.conversation_loop._ra`
(re-exported here) so monkeypatches on ``run_agent`` reach this code path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agent.model_metadata import estimate_request_tokens_rough
from agent.trajectory import has_incomplete_scratchpad
from agent.turn_compression import apply_turn_compression


class TurnAction(Enum):
    CONTINUE = "continue"
    BREAK = "break"
    RETURN = "return"


@dataclass
class TurnStep:
    """Signal returned by :func:`process_turn_response` telling
    ``run_conversation`` how to steer its ``while`` loop, plus the full
    carry-back of loop locals the caller must read back after the call.
    """

    action: TurnAction
    terminal_result: dict | None = None
    final_response: Any = None
    turn_exit_reason: str | None = None
    messages: list = field(default_factory=list)        # may be REBOUND by compression
    active_system_prompt: Any = None                    # may be REBOUND by compression
    conversation_history: Any = None                    # set None by compression
    finish_reason: str = "stop"
    length_continue_retries: int = 0
    truncated_tool_call_retries: int = 0
    truncated_response_parts: list = field(default_factory=list)
    codex_ack_continuations: int = 0


def process_turn_response(
    agent,
    *,
    response,
    api_call_count,
    effective_task_id,
    turn_id,
    api_start_time,
    api_duration,
    api_request_id,
    api_messages,
    user_message,
    system_message,
    messages,
    active_system_prompt,
    conversation_history,
    finish_reason,
    final_response,
    length_continue_retries,
    truncated_tool_call_retries,
    truncated_response_parts,
    codex_ack_continuations,
):
    """Process one model response: normalize, validate + dispatch tool
    calls (or handle the final / empty response), apply compression, and
    return a :class:`TurnStep` directing the caller's loop.

    Read-only inputs (never rebound here): ``response``, ``api_call_count``,
    ``effective_task_id``, ``turn_id``, ``api_start_time``, ``api_duration``,
    ``api_request_id``, ``api_messages``, ``user_message``, ``system_message``.

    The remaining parameters are the mutable loop locals; they are mutated
    and/or rebound here and returned via :class:`TurnStep` for the caller to
    read back. ``_turn_exit_reason`` is tracked locally and returned as
    ``turn_exit_reason``.
    """
    # Deferred import to avoid a circular import (conversation_loop imports
    # this module's caller wiring; _ra() lazy-loads run_agent).
    from agent.conversation_loop import _ra, logger

    _turn_exit_reason = None

    def _carryback():
        return {
            "final_response": final_response,
            "turn_exit_reason": _turn_exit_reason,
            "messages": messages,
            "active_system_prompt": active_system_prompt,
            "conversation_history": conversation_history,
            "finish_reason": finish_reason,
            "length_continue_retries": length_continue_retries,
            "truncated_tool_call_retries": truncated_tool_call_retries,
            "truncated_response_parts": truncated_response_parts,
            "codex_ack_continuations": codex_ack_continuations,
        }

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
                return TurnStep(TurnAction.CONTINUE, **_carryback())
            else:
                # Max retries - discard this turn and save as partial
                agent._flush_status_buffer()
                agent._vprint(f"{agent.log_prefix}❌ Max retries (2) for incomplete scratchpad. Saving as partial.", force=True)
                agent._incomplete_scratchpad_retries = 0

                rolled_back_messages = agent._get_messages_up_to_last_assistant(messages)
                agent._cleanup_task_resources(effective_task_id)
                agent._persist_session(messages, conversation_history)

                return TurnStep(TurnAction.RETURN, terminal_result={
                    "final_response": None,
                    "messages": rolled_back_messages,
                    "api_calls": api_call_count,
                    "completed": False,
                    "partial": True,
                    "error": "Incomplete REASONING_SCRATCHPAD after 2 retries"
                }, **_carryback())

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
                return TurnStep(TurnAction.CONTINUE, **_carryback())

            agent._codex_incomplete_retries = 0
            agent._persist_session(messages, conversation_history)
            return TurnStep(TurnAction.RETURN, terminal_result={
                "final_response": None,
                "messages": messages,
                "api_calls": api_call_count,
                "completed": False,
                "partial": True,
                "error": "Codex response remained incomplete after 3 continuation attempts",
            }, **_carryback())
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
                    return TurnStep(TurnAction.RETURN, terminal_result={
                        "final_response": None,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "partial": True,
                        "error": f"Model generated invalid tool call: {invalid_preview}"
                    }, **_carryback())

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
                return TurnStep(TurnAction.CONTINUE, **_carryback())
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
                    return TurnStep(TurnAction.RETURN, terminal_result={
                        "final_response": None,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "partial": True,
                        "error": "Response truncated due to output length limit",
                    }, **_carryback())

                # Track retries for invalid JSON arguments
                agent._invalid_json_retries += 1

                tool_name, error_msg = invalid_json_args[0]
                agent._buffer_vprint(f"⚠️  Invalid JSON in tool call arguments for '{tool_name}': {error_msg}")

                if agent._invalid_json_retries < 3:
                    agent._buffer_vprint(f"🔄 Retrying API call ({agent._invalid_json_retries}/3)...")
                    # Don't add anything to messages, just retry the API call
                    return TurnStep(TurnAction.CONTINUE, **_carryback())
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
                    return TurnStep(TurnAction.CONTINUE, **_carryback())

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
                return TurnStep(TurnAction.BREAK, **_carryback())

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
            return TurnStep(TurnAction.CONTINUE, **_carryback())

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
                    return TurnStep(TurnAction.BREAK, **_carryback())

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
                    return TurnStep(TurnAction.BREAK, **_carryback())

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
                    return TurnStep(TurnAction.CONTINUE, **_carryback())

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
                    return TurnStep(TurnAction.CONTINUE, **_carryback())

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
                    return TurnStep(TurnAction.CONTINUE, **_carryback())

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
                        return TurnStep(TurnAction.CONTINUE, **_carryback())

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
                return TurnStep(TurnAction.BREAK, **_carryback())

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
                return TurnStep(TurnAction.CONTINUE, **_carryback())

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
            return TurnStep(TurnAction.BREAK, **_carryback())

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
            return TurnStep(TurnAction.BREAK, **_carryback())

    # Region fell through (no WHILE-targeting exit hit) — continue the loop.
    return TurnStep(TurnAction.CONTINUE, **_carryback())


__all__ = ["TurnAction", "TurnStep", "process_turn_response"]
