"""The failover sites in the split retry loop refresh the in-flight system message.

``try_activate_fallback`` rewrites the identity lines on ``_cached_system_prompt``;
``_sync_failover_system_message`` must then patch ``api_messages[0]`` so the retry
ships it. tests/agent/test_failover_identity.py covers the two helpers; these tests
cover the call sites in agent/turn_api_call.py and agent/turn_response.py, plus the
auth-failure failover (upstream f22dd8a75ac), which test_auth_provider_failover.py
only checks as a predicate. Without that block a 401 still fails over, later and
with no reason, through the generic non-retryable path; the test pins the auth route.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.error_classifier import FailoverReason
from run_agent import AIAgent

_REPO = Path(__file__).resolve().parents[2]
_FALLBACK_PROMPT = "You are the fallback."


@pytest.fixture
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a.tool_delay = 0
    a.compression_enabled = False
    a.save_trajectories = False
    a._fallback_chain = [{"provider": "openrouter", "model": "fallback/model"}]
    a._fallback_index = 0
    a._fallback_activated = False
    return a


def _response(content):
    msg = SimpleNamespace(content=content, tool_calls=None, reasoning=None,
                          reasoning_content=None, reasoning_details=None, role="assistant")
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                           model="test/model", usage=None)


class _HTTPError(RuntimeError):
    def __init__(self, status):
        super().__init__(f"Error code: {status} - provider error")
        self.status_code = status


def _run(agent, outcomes):
    """Run one turn whose API calls return/raise ``outcomes`` in order; return
    the system message content each call was sent, and whether fallback ran."""
    sent = []
    activated = []

    def _api_call(api_kwargs):
        sent.append(api_kwargs["messages"][0]["content"])
        result = outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def _fallback(reason=None):
        # What try_activate_fallback does to the prompt (via
        # rewrite_prompt_model_identity), reduced to a visible marker.
        activated.append(reason)
        agent._fallback_index = 1
        agent._fallback_activated = True
        agent.model, agent.provider = "fallback/model", "openrouter"
        agent._cached_system_prompt = _FALLBACK_PROMPT
        return True

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_interruptible_api_call", side_effect=_api_call),
        patch.object(agent, "_try_activate_fallback", side_effect=_fallback),
        patch.object(agent, "_try_refresh_nous_client_credentials", return_value=False),
        patch("run_agent.time.sleep", return_value=None),
    ):
        result = agent.run_conversation("which model are you?")
    return result, sent, activated


def test_rate_limit_failover_sends_fallback_identity(agent):
    result, sent, activated = _run(agent, [_HTTPError(429), _response("from fallback")])
    assert activated, "the 429 should fail over"
    assert result["final_response"] == "from fallback"
    assert sent[0] == "You are helpful."
    assert sent[-1].startswith(_FALLBACK_PROMPT)


def test_auth_failure_fails_over_and_sends_fallback_identity(agent):
    result, sent, activated = _run(agent, [_HTTPError(401), _response("from fallback")])
    assert activated == [FailoverReason.auth], "a 401 that refreshing cannot fix takes the auth failover"
    assert result["final_response"] == "from fallback"
    assert sent[-1].startswith(_FALLBACK_PROMPT)


def test_empty_response_failover_sends_fallback_identity(agent):
    agent.base_url = "http://127.0.0.1:1234/v1"
    empties = [_response(None) for _ in range(4)]
    result, sent, activated = _run(agent, empties + [_response("from fallback")])
    assert activated, "repeated empty responses should fail over"
    assert result["final_response"] == "from fallback"
    assert sent[-1].startswith(_FALLBACK_PROMPT)


def _fallback_ifs(tree):
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Call)
                and isinstance(node.test.func, ast.Attribute)
                and node.test.func.attr == "_try_activate_fallback"):
            yield node


@pytest.mark.parametrize("module", ["agent/turn_api_call.py", "agent/turn_response.py"])
def test_every_fallback_site_syncs_the_system_message(module):
    """Every ``if agent._try_activate_fallback(...)`` body starts by syncing, so a
    new failover site cannot ship the stale identity."""
    tree = ast.parse((_REPO / module).read_text(encoding="utf-8"))
    sites = list(_fallback_ifs(tree))
    assert sites, f"no failover sites found in {module}"
    for node in sites:
        first = node.body[0]
        ok = (isinstance(first, ast.Assign)
              and [getattr(t, "id", None) for t in first.targets] == ["active_system_prompt"]
              and isinstance(first.value, ast.Call)
              and getattr(first.value.func, "id", None) == "_sync_failover_system_message")
        assert ok, f"{module}:{node.lineno} fails over without _sync_failover_system_message"
