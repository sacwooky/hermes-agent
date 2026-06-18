"""Tests for ADD-ON C v2 Phase 6 — L2 Robin Fusion orchestration (WI-C5).

Pure, LLM-free: an injected fake ``chat`` returns canned ``ChatResult`` per lane. Fixed
``fusion_run_id`` and ``backoff_s=(0,0,0)`` keep everything deterministic. The load-bearing
assertions are the fail-closed ones: below-quorum / empty / independence-violation / degraded-high
must NEVER produce a pass.
"""

from __future__ import annotations

import json

from hermes_cli.review_loop import fusion as F
from hermes_cli.review_loop import ninerouter as NR
from hermes_cli.review_loop import routing as R

FRID = "fusion-test0001"


def _r(content, ok=True, error=None):
    return NR.ChatResult(content=content, model="m", ok=ok, error=error)


def _juror(verdict="pass", severity="none"):
    return json.dumps({"verdict": verdict, "severity": severity, "findings": [], "summary": "ok"})


def _judge(verdict="pass", conf=0.9):
    return json.dumps({"verdict": verdict, "confidence": conf, "consensus": "agree",
                       "contradictions": [], "coverage_gaps": [], "rationale": "r"})


def _chat_for(mapping, default_ok=True):
    """Build a fake chat that dispatches by model id. mapping: {lane_substr: ChatResult|callable}."""
    judge_lanes = ("gpt-5.5", "opus")

    def _chat(model, messages, **kw):
        for key, val in mapping.items():
            if key in model:
                return val(model, messages, **kw) if callable(val) else val
        # default: jurors pass, judge passes
        if any(j in model for j in judge_lanes):
            return _r(_judge())
        return _r(_juror())
    return _chat


def _run(chat, *, author="claude", high_risk=False):
    return F.run_fusion("a real diff artifact", author_provider=author, fusion_run_id=FRID,
                        high_risk=high_risk, chat=chat, backoff_s=(0, 0, 0))


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_quorum_routine_pass():
    res = _run(_chat_for({}))  # all pass
    assert res.verdict == F.PASS
    assert res.degraded is False
    assert res.fusion_run_id == FRID
    assert len(res.lanes_run) == 3  # 2 jurors + judge (routine)


def test_quorum_high_risk_three_pass():
    res = _run(_chat_for({}), high_risk=True)
    assert res.verdict == F.PASS
    assert len(res.lanes_run) == 4  # 3 jurors + judge


def test_judge_block_over_jury_pass():
    chat = _chat_for({"gpt-5.5": _r(_judge(verdict="block", conf=0.8))})
    res = _run(chat)
    assert res.verdict == F.BLOCK


# ---------------------------------------------------------------------------
# Fail-closed (the critical guarantees)
# ---------------------------------------------------------------------------


def test_below_quorum_rejects_never_pass():
    # only one juror responds (gemini ok; openai + xai empty), judge would pass
    def chat(model, messages, **kw):
        if "gpt-5.5" in model:
            return _r(_judge())
        if "gemini" in model:
            return _r(_juror())
        return _r("", ok=False, error="down")
    res = _run(chat)
    assert res.verdict == F.REJECTED
    assert res.reject_reason.startswith("below_quorum")


def test_all_empty_fails_closed():
    res = _run(_chat_for({}, ) if False else (lambda *a, **k: _r("", ok=False, error="x")))
    assert res.verdict == F.REJECTED


def test_judge_unavailable_rejects():
    def chat(model, messages, **kw):
        if "gpt-5.5" in model or "opus" in model:
            return _r("", ok=False, error="judge down")
        return _r(_juror())
    res = _run(chat)
    assert res.verdict == F.REJECTED and res.reject_reason == "judge_unavailable"


def test_judge_unparseable_rejects():
    chat = _chat_for({"gpt-5.5": _r("not json")})
    res = _run(chat)
    assert res.verdict == F.REJECTED and res.reject_reason == "judge_unparseable"


def test_oversize_artifact_rejects():
    res = F.run_fusion("x" * 200000, author_provider="claude", fusion_run_id=FRID,
                       chat=_chat_for({}), max_artifact_chars=120000, backoff_s=(0, 0, 0))
    assert res.verdict == F.REJECTED and "exceeds_context" in res.reject_reason


def test_empty_artifact_rejects():
    res = F.run_fusion("   ", author_provider="claude", fusion_run_id=FRID, chat=_chat_for({}))
    assert res.verdict == F.REJECTED


def test_lane_independence_violation_rejects(monkeypatch):
    # Force a self-family seat into the jury → assert_lane_allowed raises → run rejected.
    # (No Grok here, so the grok-high-risk guard passes and the independence check is what fires.)
    monkeypatch.setattr(R, "select_jury",
                        lambda author, *, high_risk: [(R.CLAUDE, "cc/claude-opus-4-8", None),
                                                      (R.OPENAI, "cx/gpt-5.4-review", None)])
    res = _run(_chat_for({}), author="claude")
    assert res.verdict == F.REJECTED and "independence" in res.reject_reason


# ---------------------------------------------------------------------------
# Degraded / backfill
# ---------------------------------------------------------------------------


def test_backfill_uses_fallback_and_marks_degraded():
    # gemini primary (ag/gemini-3.1-pro-low) down → fallback (vertex/...) ok
    def chat(model, messages, **kw):
        if "gpt-5.5" in model:
            return _r(_judge())
        if "ag/gemini-3.1-pro-low" in model:
            return _r("", ok=False, error="antigravity down")
        return _r(_juror())  # vertex fallback + openai juror pass
    res = _run(chat)
    assert res.verdict == F.PASS
    assert res.degraded is True
    gem = [j for j in res.jury if j["family"] == R.GEMINI][0]
    assert gem["used_fallback"] is True


def test_degraded_routine_via_fallback_passes():
    # routine jury = {gemini, openai}; gemini primary down → fallback ok ⇒ 2 respond, degraded → PASS.
    def chat(model, messages, **kw):
        if "gpt-5.5" in model:
            return _r(_judge())
        if "ag/gemini-3.1-pro-low" in model:
            return _r("", ok=False, error="down")
        return _r(_juror())
    routine = _run(chat, high_risk=False)
    assert routine.verdict == F.PASS and routine.degraded is True


def test_high_risk_missing_juror_rejects():
    # high-risk needs 3; openai has NO fallback and is fully down ⇒ gemini+xai=2 < 3 ⇒ rejected.
    def chat(model, messages, **kw):
        if "gpt-5.5" in model:
            return _r(_judge())
        if "cx/gpt-5.4-review" in model:
            return _r("", ok=False, error="down")
        return _r(_juror())
    high = _run(chat, high_risk=True)
    assert high.verdict == F.REJECTED and high.reject_reason.startswith("below_quorum")


def test_fusion_run_id_deterministic():
    a = F.mint_fusion_run_id("t1", "abc", 1000)
    b = F.mint_fusion_run_id("t1", "abc", 1000)
    c = F.mint_fusion_run_id("t1", "abc", 1001)
    assert a == b and a != c and a.startswith("fusion-")
