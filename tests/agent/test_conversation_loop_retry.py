"""Characterization tests for the retry / error / compression paths of
``agent.conversation_loop.run_conversation`` (advisor plan 010b).

008 (``tests/agent/test_conversation_loop.py``) covers the happy path plus a
single 429 -> retry.  This file pins the *error-recovery* branches that the
010c extraction of the ~2,540-line retry loop must preserve:

  * 413 payload-too-large -> compress (via the 010a ``apply_turn_compression``
    seam) then retry and succeed.
  * 400 + context-length message -> compress then retry and succeed.
  * max_retries exhausted -> the outer handler RETURNS a failure dict
    (``completed`` falsy / ``failed`` set), it does NOT raise.
  * 5xx / timeout -> NOT routed through the 4xx compression path; pinned as
    "retryable server error" (the code retries and recovers).

These are CHARACTERIZATION tests: each assertion matches what the code does
today, established by running it and observing, not an idealized contract.
The text key in the result dict is ``"final_response"`` (NOT ``"response"``).

To keep the 008 file untouched, this file carries its OWN copy of the minimal
008 builders (``_text_response`` / ``_FakeClient`` / ``_make_agent``) plus an
error-constructor helper and a counting compression spy.  No conftest is
created or modified; no production code is touched.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Response/client builders (copied minimal subset of the 008 surface)
# ---------------------------------------------------------------------------


def _text_response(text: str):
    """A normal assistant turn with text and no tool calls (finish=stop)."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text, reasoning=None, tool_calls=[]),
                finish_reason="stop",
            )
        ],
        usage=None,
    )


def _api_error(message: str, *, status_code=None):
    """Minimal exception that drives the error classifier.

    ``agent.error_classifier._extract_status_code`` reads the ``.status_code``
    attribute, and ``classify_api_error`` matches the message text.  A bare
    ``RuntimeError`` with a ``status_code`` attribute is exactly what 008's
    429 test uses, so the classifier treats it like a real SDK error.
    """
    err = RuntimeError(message)
    if status_code is not None:
        err.status_code = status_code
    return err


class _FakeCompletions:
    """Returns scripted responses in order; records every create() call.

    A queued ``Exception`` item is raised (drives error paths); the last
    queued item repeats for any further calls.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        idx = min(len(self.calls) - 1, len(self._responses) - 1)
        item = self._responses[idx]
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(kwargs)
        return item


class _FakeClient:
    def __init__(self, responses):
        self.completions = _FakeCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def _make_agent(monkeypatch, responses, tool_names=("read_file",), max_iterations=3):
    """Build a real AIAgent wired to a fake OpenAI client (008's builder)."""
    from run_agent import AIAgent

    fake_client = _FakeClient(responses)
    monkeypatch.setattr("run_agent.OpenAI", lambda **kwargs: fake_client)
    monkeypatch.setattr(
        "run_agent.get_tool_definitions",
        lambda *args, **kwargs: [{"function": {"name": n}} for n in tool_names],
    )
    monkeypatch.setattr(
        "run_agent.handle_function_call",
        lambda name, args, task_id=None, **kwargs: json.dumps(
            {"ok": True, "tool": name, "args": args}
        ),
    )
    # Collapse retry backoff to a no-op so error-retry tests are instant.
    monkeypatch.setattr("agent.conversation_loop.jittered_backoff", lambda *a, **k: 0.0)

    agent = AIAgent(
        model="test-model",
        api_key="test-key-1234567890",
        base_url="http://localhost:8080/v1",
        platform="cli",
        max_iterations=max_iterations,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._disable_streaming = True
    agent.client = fake_client
    return agent, fake_client


def _install_compression_spy(monkeypatch):
    """Patch the 010a ``apply_turn_compression`` seam with a counting spy.

    The conversation loop imports ``apply_turn_compression`` into its own
    module namespace and calls it bare, so patching the name on
    ``agent.conversation_loop`` intercepts every in-loop compression call.

    The real seam returns ``(messages, active_system_prompt, conversation_history)``.
    The retry branches only set ``restart_with_compressed_messages`` (and thus
    actually retry) when ``len(messages) < original_len``, so the spy returns a
    SHORTENED message list to make the post-compression retry fire.  That
    shortened list is what gets re-sent to the fake client on the next attempt.
    """
    calls = {"count": 0}

    def _spy(agent, messages, system_message, *, approx_tokens, task_id):
        calls["count"] += 1
        # Drop ONE message so ``len(messages) < original_len`` is satisfied
        # (the retry gate that flips ``restart_with_compressed_messages``).
        # Removing exactly one keeps the user turn intact when possible.
        src = list(messages)
        compressed = src[1:] if len(src) > 1 else []
        return compressed, system_message, None

    monkeypatch.setattr(
        "agent.conversation_loop.apply_turn_compression", _spy
    )
    return calls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_413_payload_too_large_triggers_compression_then_succeeds(monkeypatch):
    """A 413 (payload_too_large) compresses via the seam, then the retry
    succeeds and the recovered text surfaces in ``final_response``."""
    agent, client = _make_agent(
        monkeypatch,
        [_api_error("request entity too large (413)", status_code=413),
         _text_response("ok recovered after compression")],
    )
    spy = _install_compression_spy(monkeypatch)

    result = agent.run_conversation("hi")

    assert isinstance(result, dict)
    assert spy["count"] >= 1, "413 must drive a compression via the seam"
    assert len(client.completions.calls) >= 2, (
        "the 413 call plus at least one post-compression retry should hit the client"
    )
    assert result["final_response"] is not None
    assert result["final_response"].startswith("ok recovered after compression")


def test_context_length_error_triggers_compression(monkeypatch):
    """A 400 whose message matches a context-overflow pattern compresses via
    the seam, then the retry succeeds."""
    agent, client = _make_agent(
        monkeypatch,
        [_api_error("prompt is too long: context length exceeded", status_code=400),
         _text_response("ok after context compression")],
    )
    spy = _install_compression_spy(monkeypatch)

    result = agent.run_conversation("hi")

    assert isinstance(result, dict)
    assert spy["count"] >= 1, "context-length error must drive a compression via the seam"
    assert len(client.completions.calls) >= 2
    assert result["final_response"] is not None
    assert result["final_response"].startswith("ok after context compression")


def test_max_retries_exhausted_returns_failure_dict(monkeypatch):
    """When a retriable error recurs past max_retries, run_conversation RETURNS
    a failure dict (does not raise): completed falsy, failed set,
    final_response present."""
    # 429 is retryable and rotates/falls back, but with no fallback chain and
    # no credential pool the loop just exhausts retries.  Queue enough copies
    # to outlast max_retries; the last item repeats for any further calls.
    err = _api_error("rate limit exceeded (429)", status_code=429)
    agent, client = _make_agent(monkeypatch, [err])
    # Keep the retry count small/deterministic if the agent exposes it; the
    # repeating queue guarantees exhaustion regardless of the configured max.
    max_retries = agent._api_max_retries

    result = agent.run_conversation("hi")

    assert isinstance(result, dict), "must RETURN a dict, not raise"
    assert not result.get("completed"), "exhausted retries -> completed falsy"
    assert result.get("failed"), "exhausted retries -> failed set"
    assert "final_response" in result and result["final_response"], (
        "a terminal failure still carries a final_response message"
    )
    assert len(client.completions.calls) >= max_retries, (
        f"the retriable error should have been attempted ~{max_retries} times"
    )


def test_5xx_not_treated_as_4xx(monkeypatch):
    """A 500 is classified as a retryable server_error, NOT routed through the
    4xx compression path; the loop retries and recovers without compressing."""
    agent, client = _make_agent(
        monkeypatch,
        [_api_error("internal server error (500)", status_code=500),
         _text_response("ok recovered from 5xx")],
    )
    spy = _install_compression_spy(monkeypatch)

    result = agent.run_conversation("hi")

    assert isinstance(result, dict)
    # CHARACTERIZATION: 5xx does NOT go through the compression branch.
    assert spy["count"] == 0, "a 500 must not trigger context compression"
    assert len(client.completions.calls) >= 2, (
        "the 500 call plus a retry that recovers should hit the client"
    )
    assert result["final_response"] is not None
    assert result["final_response"].startswith("ok recovered from 5xx")
