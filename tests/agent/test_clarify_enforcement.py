"""Tests for clarify-dialog enforcement (#3, run 2026-06-21-555).

Detector must be HIGH-PRECISION (no false positives — they cost a wasted
regeneration on the live core loop). Gating must honor kill-switches + bounds.
"""
# Relocated into the clarify_gate plugin; tests/conftest.py loads it as
# hermes_plugins.clarify_gate (mirrors runtime plugin discovery).
from hermes_plugins.clarify_gate import enforcement as ce
from hermes_plugins.clarify_gate.enforcement import (
    looks_like_inline_question_to_user as Q,
    should_enforce_dialog as S,
)


class _A:  # dummy agent; config is read globally via load_config, not off the agent
    pass


INLINE = "Here's my plan. Which direction do you want to go?"
STMT = "Done. Shipped and verified."

# SHOULD FIRE — inline solicitations to the user
POSITIVES = [
    "Please answer these 5:\n1. What should the MVP prove?\n2. Which input source?\n3. Who is the user?",
    "A few things:\n- Do you want auth in v1?\n- Should we include share links?",
    "Here's my plan. Which direction do you want to go?",
    "I can do A or B. Would you like me to start with the prototype?",
    "Got it.\n\nShould I move into the V18 discovery next?",
    "What platform should this target?\nAnd do you want dark mode?",
]
# SHOULD NOT FIRE — false-positive guards
NEGATIVES = [
    "",
    "Done. I shipped the fix and verified it.",
    "Why does this matter? Because the gate was skipped.",     # rhetorical, self-answered
    "The function returns None on timeout.",
    "Here's the code:\n```\nshould I run this?\n```\nDone.",    # question only in code
    "> should we proceed?\n\nThat quote is from the old doc.",  # question only in blockquote
    "I considered: is this safe? It is, because of the kill-switch.",
    "I updated reviewer and teaching-labs. Both show allowed.",
]


def test_detector_positives():
    for t in POSITIVES:
        assert Q(t), f"should have fired: {t!r}"


def test_detector_no_false_positives():
    for t in NEGATIVES:
        assert not Q(t), f"FALSE POSITIVE: {t!r}"


def test_default_on(monkeypatch):
    monkeypatch.setattr(ce, "_agent_cfg", lambda: {})
    assert S(_A(), INLINE, 0) is True


def test_statement_never_enforced(monkeypatch):
    monkeypatch.setattr(ce, "_agent_cfg", lambda: {})
    assert S(_A(), STMT, 0) is False


def test_bounded_retries(monkeypatch):
    monkeypatch.setattr(ce, "_agent_cfg", lambda: {})       # default max=1
    assert S(_A(), INLINE, 1) is False                      # already retried once


def test_custom_max_retries(monkeypatch):
    monkeypatch.setattr(ce, "_agent_cfg", lambda: {"clarify_enforcement_max_retries": 2})
    assert S(_A(), INLINE, 1) is True


def test_config_kill_switch(monkeypatch):
    monkeypatch.setattr(ce, "_agent_cfg", lambda: {"clarify_enforcement": False})
    assert S(_A(), INLINE, 0) is False


def test_env_kill_switch(monkeypatch):
    monkeypatch.setattr(ce, "_agent_cfg", lambda: {})
    monkeypatch.setenv("HERMES_CLARIFY_ENFORCEMENT", "off")
    assert S(_A(), INLINE, 0) is False


def test_fail_open_on_none(monkeypatch):
    monkeypatch.setattr(ce, "_agent_cfg", lambda: {})
    assert S(_A(), None, 0) is False
