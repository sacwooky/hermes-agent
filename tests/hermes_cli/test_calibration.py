"""Tests for ADD-ON C v2 Phase 8 — calibration / regression meta-loop (WI-C10). Pure, LLM-free."""

from __future__ import annotations

import json
from pathlib import Path

from hermes_cli.review_loop import calibration as C
from hermes_cli.review_loop.fusion import FusionResult


def _fr(jury_lanes, judge_lane="cx/gpt-5.5-review", run_id="fusion-x"):
    jury = [{"family": "f", "lane": lane, "ok": True, "verdict": "pass"} for lane in jury_lanes]
    return FusionResult(verdict="pass", confidence={}, jury=jury,
                        judge={"lane": judge_lane, "ok": True}, fusion_run_id=run_id,
                        lanes_run=jury_lanes + [judge_lane])


# --- review events / scorecard ---
def test_fusion_review_events_per_lane():
    fr = _fr(["ag/gemini-3.1-pro-low", "cx/gpt-5.4-review"])
    evs = C.fusion_review_events(fr)
    lanes = {e["review_lane"] for e in evs}
    assert lanes == {"ag/gemini-3.1-pro-low", "cx/gpt-5.4-review", "cx/gpt-5.5-review"}
    assert all(e["role"] == "reviewer" and e["kind"] == "review" for e in evs)


def test_review_events_skip_failed_jurors():
    fr = FusionResult(verdict="pass", confidence={},
                      jury=[{"lane": "ag/gemini-3.1-pro-low", "ok": True},
                            {"lane": "cx/gpt-5.4-review", "ok": False}],
                      judge={"lane": "cx/gpt-5.5-review", "ok": True}, fusion_run_id="r")
    lanes = {e["review_lane"] for e in C.fusion_review_events(fr)}
    assert "cx/gpt-5.4-review" not in lanes  # failed juror not counted as a review


def test_per_juror_scorecard_escape_rate():
    events = (C.fusion_review_events(_fr(["ag/gemini-3.1-pro-low"])) * 4  # 4 reviews each lane
              + C.g3_escape_events(["ag/gemini-3.1-pro-low"]))            # 1 escape on gemini
    sc = C.per_juror_scorecard(events)
    assert sc["ag/gemini-3.1-pro-low"]["reviews"] == 4
    assert sc["ag/gemini-3.1-pro-low"]["escapes"] == 1
    assert sc["ag/gemini-3.1-pro-low"]["escape_rate"] == 0.25


# --- emit + regression registry (fail-safe, env-gated) ---
def test_emit_role_events_writes(tmp_path):
    ok = C.emit_role_events(C.fusion_review_events(_fr(["cx/gpt-5.4-review"])), vault_root=str(tmp_path))
    assert ok
    stream = tmp_path / "metrics" / "autonomy" / "role-events.jsonl"
    rows = [json.loads(x) for x in stream.read_text().splitlines() if x.strip()]
    assert rows and all(r["role"] == "reviewer" for r in rows)


def test_emit_noop_without_vault(monkeypatch):
    monkeypatch.delenv("HERMES_LEARNING_VAULT_ROOT", raising=False)
    assert C.emit_role_events([{"role": "reviewer", "kind": "review", "review_lane": "x"}]) is False


def test_record_g3_escape_attributes_and_registers(tmp_path):
    fr = _fr(["ag/gemini-3.1-pro-low", "cx/gpt-5.4-review"], run_id="fusion-esc1")
    C.record_g3_escape(fr, task_id="t_epic", reject_reason="missed an auth bug", vault_root=str(tmp_path))
    d = tmp_path / "metrics" / "autonomy"
    escapes = [json.loads(x) for x in (d / "role-events.jsonl").read_text().splitlines() if x.strip()]
    assert all(e["kind"] == "review_escape" for e in escapes)
    assert {e["review_lane"] for e in escapes} == {"ag/gemini-3.1-pro-low", "cx/gpt-5.4-review", "cx/gpt-5.5-review"}
    cases = [json.loads(x) for x in (d / "fusion-regression-cases.jsonl").read_text().splitlines() if x.strip()]
    assert cases[-1]["fusion_run_id"] == "fusion-esc1" and "auth bug" in cases[-1]["reject_reason"]


# --- pinned judge contract ---
def test_judge_contract_pinned_ok():
    C.assert_judge_contract(C.JUDGE_CONTRACT["slug"], C.JUDGE_CONTRACT["rubric_version"])  # no raise


def test_judge_contract_drift_raises():
    import pytest
    with pytest.raises(ValueError):
        C.assert_judge_contract("cx/gpt-5.4-review", "fusion-v1")  # wrong slug
    with pytest.raises(ValueError):
        C.assert_judge_contract("cx/gpt-5.5-review", "fusion-v2")  # rubric bumped without amendment


# --- agreement bar (WI-C10 ship gate) ---
def test_agreement_rate_and_bar():
    labeled = [{"judge_verdict": "block", "human_label": "block"},
               {"judge_verdict": "pass", "human_label": "pass"},
               {"judge_verdict": "pass", "human_label": "block"},  # disagreement
               {"judge_verdict": "block", "human_label": "block"},
               {"judge_verdict": "pass", "human_label": "pass"}]
    assert C.agreement_rate(labeled) == 0.8
    assert C.meets_agreement_bar(labeled) is True            # 0.8 >= 0.80
    assert C.meets_agreement_bar(labeled, bar=0.9) is False
