"""Tests for the clarify_gate plugin (fleet three-gate customization).

The plugin is loaded as ``hermes_plugins.clarify_gate`` by tests/conftest.py
(mirrors runtime discovery). These cover the seam-facing API that core files
delegate to, plus the fail-open contract that keeps the agent on upstream
behavior when the plugin is absent.
"""
from __future__ import annotations

import json

from hermes_plugins.clarify_gate import gate


class TestInvokeClarifyCallbackShim:
    def test_gate_aware_callback_receives_gate(self):
        seen = {}

        def cb(q, c, gate=False):
            seen["gate"] = gate
            return "ok"

        assert gate.invoke_clarify_callback(cb, "q", None, True) == "ok"
        assert seen["gate"] is True

    def test_legacy_two_arg_callback_not_passed_gate(self):
        def legacy(q, c):  # no gate param — must not crash
            return "ok"

        assert gate.invoke_clarify_callback(legacy, "q", ["a"], True) == "ok"

    def test_kwargs_callback_receives_gate(self):
        seen = {}

        def cb(q, c, **kw):
            seen["gate"] = kw.get("gate")
            return "x"

        gate.invoke_clarify_callback(cb, "q", None, True)
        assert seen["gate"] is True

    def test_genuine_typeerror_inside_callback_not_swallowed(self):
        def cb(q, c, gate=False):
            raise TypeError("boom from inside")

        try:
            gate.invoke_clarify_callback(cb, "q", None, True)
        except TypeError as e:
            assert "boom from inside" in str(e)
        else:
            raise AssertionError("TypeError raised inside the callback was swallowed")


class TestTimeoutsAndMessages:
    def test_resolve_cli_timeout_gate_vs_nongate(self):
        cfg = {"timeout": 120, "gate_timeout": 999}
        assert gate.resolve_cli_timeout(False, cfg) == 120
        assert gate.resolve_cli_timeout(True, cfg) == 999

    def test_resolve_cli_timeout_gate_default_when_missing(self):
        assert gate.resolve_cli_timeout(True, {}) == gate.DEFAULT_GATE_TIMEOUT

    def test_get_gate_timeout_default(self):
        assert gate.get_gate_timeout() == gate.DEFAULT_GATE_TIMEOUT == 86400

    def test_hold_messages_never_a_default(self):
        # Hold text must instruct re-ask / never-proceed, not a fabricated answer.
        for msg in (gate.HOLD_MESSAGE, gate.CLI_HOLD_MESSAGE):
            assert msg and "never" in msg.lower()
            assert "default" in msg.lower()

    def test_cli_timeout_message_distinguishes_gate(self):
        assert "gate" in gate.cli_timeout_message(True, 86400).lower()
        assert "gate" not in gate.cli_timeout_message(False, 120).lower()


class TestClarifyToolFailOpen:
    def test_clarify_tool_falls_back_when_plugin_absent(self, monkeypatch):
        """With the plugin un-importable, clarify_tool must still work and call
        the callback in its plain 2-arg upstream shape (gate ignored, never crash)."""
        import sys
        import tools.clarify_tool as ct

        # Hide the plugin so the seam's ``from hermes_plugins import clarify_gate``
        # raises ImportError and takes the upstream fallback path.
        monkeypatch.setitem(sys.modules, "hermes_plugins.clarify_gate", None)
        monkeypatch.setitem(sys.modules, "hermes_plugins", None)

        def cb(q, c):  # plain upstream callback
            return "decided"

        out = json.loads(ct.clarify_tool("pick", ["a", "b"], cb, gate=True))
        assert out["user_response"] == "decided"
