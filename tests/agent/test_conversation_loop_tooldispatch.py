"""Characterization tests for the response-processing / tool-dispatch block of
``agent.conversation_loop.run_conversation`` (advisor plan 010d).

008 (``tests/agent/test_conversation_loop.py``) has exactly ONE tool-dispatch
test (``test_run_conversation_with_tool_call``).  After the 010c extraction of
the retry/error machinery, the remaining body of the outer turn loop is the
response-processing / tool-dispatch block (~998 lines, many control-flow
exits).  Extracting that block (010e) is the highest-risk step, so this file
adds the missing characterization FIRST.

These are CHARACTERIZATION tests: each assertion matches what the code does
TODAY (established by running it and observing), not an idealized contract.
The text key in the result dict is ``"final_response"`` (NOT ``"response"``).

Observed mechanics of the block under test (for the 010e extraction):

  * The block normalizes the API response, then on ``assistant_message.tool_calls``
    appends the assistant message to ``messages`` and dispatches each tool via
    ``AIAgent._execute_tool_calls`` -> ``run_agent.handle_function_call`` (one
    call per tool_call).  Tool results are appended to ``messages`` as
    ``role="tool"`` dicts BEFORE the next API call.
  * With tool_calls present the loop *continues* (next ``while`` iteration ->
    another API call).  A finish=stop / no-tool-call text turn sets
    ``final_response`` and *breaks* out to ``finalize_turn``.
  * When the outer ``while`` runs out (``api_call_count >= max_iterations`` with
    ``final_response`` still None), ``finalize_turn`` calls
    ``_handle_max_iterations`` (one extra toolless summary call) and then RETURNS
    a result dict with ``completed`` False — it does NOT hang or raise.

To keep the 008 / 010b files untouched, this file carries its OWN copy of the
minimal builders (``_text_response`` / ``_tool_call_response`` / ``_FakeClient``
/ ``_make_agent``).  No conftest is created or modified; no production code is
touched.
"""

from __future__ import annotations

import json
from types import SimpleNamespace


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


def _tool_call(name: str, arguments, call_id: str = "call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _tool_call_response(*calls, content=None):
    """An assistant turn that requests one or more tool calls.

    Each positional arg is a ``(name, arguments)`` tuple (or a pre-built
    tool-call namespace).  ``content`` lets a test pin the "content AND
    tool_calls together" branch.
    """
    tcs = []
    for idx, c in enumerate(calls):
        if isinstance(c, tuple):
            name, arguments = c
            tcs.append(_tool_call(name, arguments, call_id=f"call_{idx + 1}"))
        else:
            tcs.append(c)
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content, reasoning=None, tool_calls=tcs
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=None,
    )


class _FakeCompletions:
    """Returns scripted responses in order; records every create() call.

    The last queued item repeats for any further calls (so a never-ending
    tool-call stream can be expressed with a single queued response).
    """

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


def _make_agent(monkeypatch, responses, dispatched, tool_names=("read_file", "write_file"),
                max_iterations=3):
    """Build a real AIAgent wired to a fake OpenAI client (008's builder).

    ``dispatched`` is a caller-supplied list; the patched
    ``handle_function_call`` records ``(name, args)`` into it per dispatch so
    tests can assert call count / names / args.  The tool result is a JSON
    string (what the real dispatcher returns), so the loop appends a normal
    ``role="tool"`` message.
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
        lambda name, args, task_id=None, **kwargs: (
            dispatched.append((name, args)) or json.dumps({"ok": True, "tool": name})
        ),
    )
    # Collapse retry backoff to a no-op so any error path is instant.
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_multiple_tool_calls_in_one_response_all_dispatched(monkeypatch):
    """A single assistant response with 2 tool_calls dispatches the tool
    executor once PER tool (names + args captured), then a follow-up text turn
    yields the final_response."""
    dispatched: list = []
    agent, client = _make_agent(
        monkeypatch,
        [
            _tool_call_response(
                ("read_file", {"path": "a.txt"}),
                ("write_file", {"path": "b.txt", "data": "x"}),
            ),
            _text_response("both tools done"),
        ],
        dispatched,
    )

    result = agent.run_conversation("do two things")

    # One dispatch per tool_call in the batch.
    assert len(dispatched) == 2, "each tool_call in the batch must be dispatched once"
    names = [n for n, _ in dispatched]
    assert sorted(names) == ["read_file", "write_file"], (
        "both tool names dispatched (order may vary if batch parallelizes)"
    )
    by_name = dict(dispatched)
    assert by_name["read_file"] == {"path": "a.txt"}
    assert by_name["write_file"] == {"path": "b.txt", "data": "x"}
    # Follow-up text turn delivers the answer.
    assert result["final_response"].startswith("both tools done")
    # Tool-call turn + final answer turn.
    assert len(client.completions.calls) == 2


def test_tool_result_appended_to_messages(monkeypatch):
    """After a tool call, a role="tool" result message is appended to the
    running messages BEFORE the next API call.  Proven by inspecting the
    SECOND request's ``messages`` kwarg: it must contain a tool-result entry
    for the dispatched call_id that was absent from the first request."""
    dispatched: list = []
    agent, client = _make_agent(
        monkeypatch,
        [
            _tool_call_response(("read_file", {"path": "README.md"})),
            _text_response("read complete"),
        ],
        dispatched,
    )

    result = agent.run_conversation("read it")

    assert len(client.completions.calls) == 2, "tool turn + final answer turn"
    first_msgs = client.completions.calls[0]["messages"]
    second_msgs = client.completions.calls[1]["messages"]

    def _tool_results(msgs):
        return [m for m in msgs if isinstance(m, dict) and m.get("role") == "tool"]

    assert _tool_results(first_msgs) == [], (
        "no tool result exists before the tool is dispatched"
    )
    second_tool_results = _tool_results(second_msgs)
    assert second_tool_results, "the tool result must be appended before the 2nd API call"
    # The appended result carries the tool name and is bound to the call_id.
    blob = json.dumps(second_tool_results)
    assert "read_file" in blob, "tool result message records the tool name"
    assert "call_1" in json.dumps(second_msgs), (
        "the dispatched tool_call_id is present in the next request"
    )
    assert result["final_response"].startswith("read complete")


def test_tool_then_final_answer_two_turns(monkeypatch):
    """A tool-call turn followed by a text turn produces EXACTLY two API calls
    and returns the final text (the break-out-of-loop path)."""
    dispatched: list = []
    agent, client = _make_agent(
        monkeypatch,
        [
            _tool_call_response(("read_file", {"path": "x"})),
            _text_response("final answer text"),
        ],
        dispatched,
    )

    result = agent.run_conversation("go")

    assert len(client.completions.calls) == 2, "exactly two API calls"
    assert len(dispatched) == 1, "one tool dispatched on the first turn"
    assert result["final_response"].startswith("final answer text")
    assert result["completed"] is True, "a clean text-turn finish completes the turn"


def test_content_and_tool_calls_together(monkeypatch):
    """A response carrying BOTH content and tool_calls.  CHARACTERIZATION: the
    tool branch wins — the tool is dispatched and the loop continues; the
    same-turn content is captured as a fallback but the follow-up text turn's
    text is what surfaces in final_response."""
    dispatched: list = []
    agent, client = _make_agent(
        monkeypatch,
        [
            _tool_call_response(
                ("read_file", {"path": "z"}),
                content="here is my answer and I will also read a file",
            ),
            _text_response("post-tool final text"),
        ],
        dispatched,
    )

    result = agent.run_conversation("mixed turn")

    # The tool was still dispatched despite content being present.
    assert dispatched == [("read_file", {"path": "z"})]
    # The same-turn content was captured as the with-tools fallback.
    assert agent._last_content_with_tools == (
        "here is my answer and I will also read a file"
    )
    # The follow-up text turn supplies the surfaced final_response.
    assert result["final_response"].startswith("post-tool final text")
    assert len(client.completions.calls) == 2


def test_max_iterations_reached_during_tool_loop(monkeypatch):
    """If the model keeps returning tool calls forever, the outer loop hits
    max_iterations and run_conversation RETURNS a dict (does not hang/raise).

    CHARACTERIZATION: ``completed`` is falsy (api_call_count reached the cap),
    the result is a dict, and the tool was dispatched once per loop iteration.
    The single queued tool-call response repeats for every create() call."""
    dispatched: list = []
    agent, client = _make_agent(
        monkeypatch,
        # One never-ending tool-call response (repeats for every call).
        [_tool_call_response(("read_file", {"path": "loop"}))],
        dispatched,
        max_iterations=3,
    )

    result = agent.run_conversation("loop forever")

    assert isinstance(result, dict), "must RETURN a dict, not hang or raise"
    assert not result.get("completed"), (
        "hitting max_iterations -> completed is falsy"
    )
    # The block dispatched the tool once per main-loop iteration (max_iterations
    # of them); the extra summary call is toolless so it adds no dispatch.
    assert len(dispatched) == agent.max_iterations, (
        "one dispatch per main-loop iteration up to max_iterations"
    )
    assert result.get("turn_exit_reason"), "a terminal turn records an exit reason"
    # The client was called once per loop iteration plus the toolless summary.
    assert len(client.completions.calls) >= agent.max_iterations
