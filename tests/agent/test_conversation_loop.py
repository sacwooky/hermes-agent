"""Characterization tests for ``agent.conversation_loop.run_conversation``.

``run_conversation`` is the ~4,400-line single function that drives one
user turn through the agent (model call, tool dispatch, retries, fallbacks,
compression, post-turn hooks).  There was previously NO unit-level test
file for it.  These tests pin down its CURRENT observable behavior so a
later decomposition (advisor plan 010) can prove it didn't silently change
the contract.

These are CHARACTERIZATION tests: each assertion matches what the code
actually does today, not an idealized contract.  In particular:

  * The result dict's primary text key is ``"final_response"`` (NOT
    ``"response"`` — that name appears nowhere in the returned dict).
  * The loop is driven via ``AIAgent.run_conversation`` (a thin forwarder
    to ``agent.conversation_loop.run_conversation``).

Mock surface (mirrors the proven pattern in
``tests/run_agent/test_dict_tool_call_args.py``):

  * ``run_agent.OpenAI`` is patched with a fake client exposing
    ``chat.completions.create(**kwargs)`` so no network call happens.  The
    non-streaming path (``agent._disable_streaming = True``) routes the
    request through ``chat.completions.create``, and the ``messages`` kwarg
    it receives is the assembled request — that's our inspection point.
  * ``run_agent.handle_function_call`` is patched so tool dispatch is a
    pure function returning a JSON string.
  * ``run_agent.get_tool_definitions`` is patched to a minimal tool list.

No real APIs are invoked.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Response/client builders (OpenAI chat.completions shape)
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


def _tool_call(name: str, arguments):
    return SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _tool_call_response(name: str, arguments):
    """An assistant turn that requests a single tool call (finish=tool_calls)."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None, reasoning=None, tool_calls=[_tool_call(name, arguments)]
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=None,
    )


class _FakeCompletions:
    """Returns scripted responses in order; records every create() call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []  # captured kwargs per create() invocation

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


# ---------------------------------------------------------------------------
# Agent fixture / builder
# ---------------------------------------------------------------------------


def _make_agent(monkeypatch, responses, tool_names=("read_file",), max_iterations=3):
    """Build a real AIAgent wired to a fake OpenAI client.

    The fake client returns ``responses`` in sequence from
    ``chat.completions.create``.  Tool dispatch and tool definitions are
    patched so the loop never touches the network or the real tool layer.
    """
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
    # Ensure the captured client is the one in use.
    agent.client = fake_client
    return agent, fake_client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_conversation_single_turn_returns_response(monkeypatch):
    """A single text turn returns a dict whose final_response carries the text."""
    agent, _ = _make_agent(monkeypatch, [_text_response("Hello from mock")])

    result = agent.run_conversation("hi")

    assert isinstance(result, dict)
    # CHARACTERIZATION: the text key is "final_response", not "response".
    assert "final_response" in result
    assert isinstance(result["final_response"], str)
    assert result["final_response"].startswith("Hello from mock")


def test_run_conversation_no_tools_single_turn(monkeypatch):
    """With no tool calls in the response, no tool dispatch occurs and one
    LLM call produces the answer."""
    dispatched = []
    agent, client = _make_agent(monkeypatch, [_text_response("no tools needed")])
    # Re-apply a recording dispatch (the _make_agent patch installed its own).
    monkeypatch.setattr(
        "run_agent.handle_function_call",
        lambda name, args, task_id=None, **kwargs: dispatched.append(name) or "{}",
    )

    result = agent.run_conversation("answer directly")

    assert result["final_response"].startswith("no tools needed")
    assert dispatched == [], "no tool calls in the response -> no dispatch"
    assert len(client.completions.calls) == 1


def test_run_conversation_with_tool_call(monkeypatch):
    """A tool-call turn dispatches the tool once, then a follow-up text turn
    delivers the final answer."""
    dispatched = []

    agent, client = _make_agent(
        monkeypatch,
        [
            _tool_call_response("read_file", {"path": "README.md"}),
            _text_response("done reading"),
        ],
    )
    monkeypatch.setattr(
        "run_agent.handle_function_call",
        lambda name, args, task_id=None, **kwargs: dispatched.append((name, args))
        or json.dumps({"ok": True}),
    )

    result = agent.run_conversation("read the file")

    assert dispatched == [("read_file", {"path": "README.md"})], (
        "tool executor called exactly once with the requested args"
    )
    assert result["final_response"].startswith("done reading")
    # Two LLM calls: the tool-call turn + the final answer turn.
    assert len(client.completions.calls) == 2


def test_run_conversation_api_error_retries(monkeypatch):
    """A retriable (429) error on the first LLM call is retried; the second
    call succeeds and a result is RETURNED (not raised)."""
    rate_limit_error = RuntimeError("rate limit exceeded (429)")
    rate_limit_error.status_code = 429

    agent, client = _make_agent(
        monkeypatch,
        [rate_limit_error, _text_response("recovered after retry")],
    )

    # Must not raise; the loop classifies 429 as retryable and retries.
    result = agent.run_conversation("hi")

    assert isinstance(result, dict)
    assert len(client.completions.calls) >= 2, (
        "the first (429) call plus at least one retry should hit the client"
    )
    # The recovered text should surface in the final response.
    assert result["final_response"] is not None
    assert "recovered after retry" in result["final_response"]


def test_run_conversation_passes_history(monkeypatch):
    """A provided conversation_history is included in the messages sent to
    the LLM."""
    agent, client = _make_agent(monkeypatch, [_text_response("ack")])

    history = [
        {"role": "user", "content": "earlier question MARKER_PRIOR_USER"},
        {"role": "assistant", "content": "earlier answer MARKER_PRIOR_ASSISTANT"},
    ]

    agent.run_conversation("new question", conversation_history=history)

    assert client.completions.calls, "the LLM should have been called"
    sent_messages = client.completions.calls[0]["messages"]
    blob = json.dumps(sent_messages)
    assert "MARKER_PRIOR_USER" in blob, "prior user turn must be in the request"
    assert "MARKER_PRIOR_ASSISTANT" in blob, (
        "prior assistant turn must be in the request"
    )
    assert "new question" in blob, "the current user message must be in the request"


def test_run_conversation_stream_callback_called(monkeypatch):
    """When a stream_callback is supplied, the streaming path is taken and the
    callback receives text deltas at least once."""
    callback = MagicMock()

    # Streaming path: client.chat.completions.create(stream=True) returns an
    # iterable of chunks. Build chunks in OpenAI streaming delta shape.
    def _delta_chunk(text=None, finish=None):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=text, reasoning=None, tool_calls=None),
                    finish_reason=finish,
                )
            ],
            usage=None,
        )

    class _StreamingCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return iter(
                [
                    _delta_chunk("Hello "),
                    _delta_chunk("stream"),
                    _delta_chunk(finish="stop"),
                ]
            )

    class _StreamingClient:
        def __init__(self):
            self.completions = _StreamingCompletions()
            self.chat = SimpleNamespace(completions=self.completions)

    from run_agent import AIAgent

    streaming_client = _StreamingClient()
    monkeypatch.setattr("run_agent.OpenAI", lambda **kwargs: streaming_client)
    monkeypatch.setattr(
        "run_agent.get_tool_definitions",
        lambda *args, **kwargs: [{"function": {"name": "read_file"}}],
    )
    monkeypatch.setattr(
        "run_agent.handle_function_call",
        lambda name, args, task_id=None, **kwargs: "{}",
    )
    monkeypatch.setattr("agent.conversation_loop.jittered_backoff", lambda *a, **k: 0.0)

    agent = AIAgent(
        model="test-model",
        api_key="test-key-1234567890",
        base_url="http://localhost:8080/v1",
        platform="cli",
        max_iterations=3,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.client = streaming_client

    result = agent.run_conversation("hi", stream_callback=callback)

    assert isinstance(result, dict)
    assert callback.called, "stream_callback must receive at least one delta"
    # The deltas concatenate into the final response text.
    assert result["final_response"] is not None
    assert "Hello" in result["final_response"]


@pytest.mark.skip(
    reason="Interrupt is driven by run_agent._set_interrupt / a threading.Event "
    "checked deep inside the loop's per-iteration guards. Triggering it "
    "deterministically from a unit test requires racing the interrupt flag "
    "against the (mocked, instant) LLM call, which is inherently flaky. "
    "Interrupt-return behavior is covered by the dedicated integration tests "
    "tests/run_agent/test_concurrent_interrupt.py and "
    "tests/run_agent/test_real_interrupt_subagent.py."
)
def test_run_conversation_interrupt_returns_partial():
    pass
