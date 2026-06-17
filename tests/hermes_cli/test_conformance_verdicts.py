"""Tests for EPIC 2 — QA conformance axes in the G3 acceptance packet.

Covers:
- _harvest_conformance_verdicts returns empty dict when no events
- record_conformance_verdict writes an event readable by _harvest_conformance_verdicts
- Security FAIL blocks G3 acceptance creation (emits conformance_gate_block)
- Security PASS/skip allows G3 to be created normally
- Advisory axes (perf/a11y) FAIL do not block G3
- Conformance verdicts appear in acceptance_packet when present
- B4: author-aware guard rejects same-provider verdicts
- B4: cross-provider pairing (Claude-authored → Gemini lane accepted)
- B5: high-risk epic requires cross-check second opinion
- B5: cross-check independence violation blocks G3
- B6: security FAIL spawns fix story (WI-QA4 auto-spawn)
- B6: escalate-once after retry cap reached
- B6: fix-story is deduped (no duplicate spawn per retry slot)

LLM-free by design.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_autonomy as ka
from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Shared fixtures (same pattern as test_kanban_autonomy.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decompose_epic(conn):
    """Create a minimal epic + two child stories, returning (root_id, child_ids)."""
    root = kb.create_task(conn, title="Epic: conformance test", triage=True)
    children = [
        {"title": "build feature", "assignee": "builder", "parents": []},
        {"title": "qa feature", "assignee": "qa", "parents": [0],
         "work_item_type": "qa"},
    ]
    child_ids = kb.decompose_triage_task(
        conn, root, root_assignee="orchestrator", children=children,
        author="orchestrator",
    )
    return root, child_ids


def _finish_epic(conn, root, child_ids):
    """Complete all stories and the epic root."""
    for cid in child_ids:
        kb.complete_task(conn, cid, result="done",
                         metadata={"demo_url": "http://demo:9120/"})
    kb.complete_task(conn, root, result="epic complete")


# ---------------------------------------------------------------------------
# _harvest_conformance_verdicts — baseline
# ---------------------------------------------------------------------------


def test_harvest_verdicts_empty_when_no_events(kanban_home):
    with kb.connect() as conn:
        root, _ = _decompose_epic(conn)
        result = ka._harvest_conformance_verdicts(conn, root)
    assert result == {}


# ---------------------------------------------------------------------------
# record_conformance_verdict → _harvest_conformance_verdicts round-trip
# ---------------------------------------------------------------------------


def test_record_and_harvest_security_verdict(kanban_home):
    with kb.connect() as conn:
        root, _ = _decompose_epic(conn)
        ka.record_conformance_verdict(
            conn, root, axis="security", verdict="pass",
            lane="robin-security-scan", findings=[],
            run_record="runs/2026-06-17-900-sec.md",
        )
        verdicts = ka._harvest_conformance_verdicts(conn, root)
    assert "security" in verdicts
    sec = verdicts["security"]
    assert sec["verdict"] == "pass"
    assert sec["lane"] == "robin-security-scan"
    assert sec["signed"] is True
    assert sec["findings"] == []


def test_record_and_harvest_all_three_axes(kanban_home):
    with kb.connect() as conn:
        root, _ = _decompose_epic(conn)
        ka.record_conformance_verdict(
            conn, root, axis="security", verdict="pass",
            lane="robin-sec", findings=[],
        )
        ka.record_conformance_verdict(
            conn, root, axis="perf", verdict="fail",
            lane="robin-perf", findings=["p95 latency > 500ms"],
        )
        ka.record_conformance_verdict(
            conn, root, axis="a11y", verdict="skip",
            lane="robin-a11y", findings=[],
        )
        verdicts = ka._harvest_conformance_verdicts(conn, root)
    assert verdicts["security"]["verdict"] == "pass"
    assert verdicts["performance"]["verdict"] == "fail"
    assert verdicts["accessibility"]["verdict"] == "skip"


def test_record_conformance_verdict_raises_on_unknown_axis(kanban_home):
    with kb.connect() as conn:
        root, _ = _decompose_epic(conn)
        with pytest.raises(ValueError, match="Unknown conformance axis"):
            ka.record_conformance_verdict(
                conn, root, axis="bogus", verdict="pass", lane="test"
            )


def test_record_conformance_verdict_latest_wins(kanban_home):
    """If two verdicts for the same axis are recorded, the latest wins."""
    with kb.connect() as conn:
        root, _ = _decompose_epic(conn)
        ka.record_conformance_verdict(
            conn, root, axis="security", verdict="fail",
            lane="robin-sec", findings=["vuln-old"],
        )
        # Record a second (re-scan after fix)
        ka.record_conformance_verdict(
            conn, root, axis="security", verdict="pass",
            lane="robin-sec", findings=[],
        )
        verdicts = ka._harvest_conformance_verdicts(conn, root)
    # _has_event returns the latest (ORDER BY id DESC LIMIT 1)
    assert verdicts["security"]["verdict"] == "pass"


# ---------------------------------------------------------------------------
# Security gate: FAIL blocks G3 acceptance creation
# ---------------------------------------------------------------------------


def test_security_fail_blocks_g3_creation(kanban_home):
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        # Record a FAIL security verdict on the epic
        ka.record_conformance_verdict(
            conn, root, axis="security", verdict="fail",
            lane="robin-sec", findings=["SQL injection in /api/login"],
        )
        created = ka.generate_epic_acceptances(conn)
    # G3 acceptance must NOT be created when security fails
    assert created == []


def test_security_fail_emits_conformance_gate_block_event(kanban_home):
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        ka.record_conformance_verdict(
            conn, root, axis="security", verdict="fail",
            lane="robin-sec", findings=["XSS in search"],
        )
        ka.generate_epic_acceptances(conn)
        events = [e.kind for e in kb.list_events(conn, root)]
    assert "conformance_gate_block" in events


def test_security_fail_does_not_emit_acceptance_task_created(kanban_home):
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        ka.record_conformance_verdict(
            conn, root, axis="security", verdict="fail",
            lane="robin-sec", findings=["RCE"],
        )
        ka.generate_epic_acceptances(conn)
        events = [e.kind for e in kb.list_events(conn, root)]
    assert "acceptance_task_created" not in events


# ---------------------------------------------------------------------------
# Security PASS / skip allows G3 creation
# ---------------------------------------------------------------------------


def test_security_pass_allows_g3_creation(kanban_home):
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        ka.record_conformance_verdict(
            conn, root, axis="security", verdict="pass",
            lane="robin-sec", findings=[],
        )
        created = ka.generate_epic_acceptances(conn)
    assert len(created) == 1


def test_security_skip_allows_g3_creation(kanban_home):
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        ka.record_conformance_verdict(
            conn, root, axis="security", verdict="skip",
            lane="robin-sec", findings=[],
        )
        created = ka.generate_epic_acceptances(conn)
    assert len(created) == 1


def test_no_conformance_verdict_allows_g3_creation(kanban_home):
    """No security verdict recorded → treated as skip, G3 proceeds."""
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        created = ka.generate_epic_acceptances(conn)
    assert len(created) == 1


# ---------------------------------------------------------------------------
# Advisory axes (perf/a11y) do NOT block G3 even when FAIL
# ---------------------------------------------------------------------------


def test_perf_fail_does_not_block_g3(kanban_home):
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        ka.record_conformance_verdict(
            conn, root, axis="perf", verdict="fail",
            lane="robin-perf", findings=["p99 > 2s"],
        )
        created = ka.generate_epic_acceptances(conn)
    assert len(created) == 1


def test_a11y_fail_does_not_block_g3(kanban_home):
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        ka.record_conformance_verdict(
            conn, root, axis="a11y", verdict="fail",
            lane="robin-a11y", findings=["missing aria labels"],
        )
        created = ka.generate_epic_acceptances(conn)
    assert len(created) == 1


def test_perf_and_a11y_fail_together_does_not_block_g3(kanban_home):
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        ka.record_conformance_verdict(
            conn, root, axis="perf", verdict="fail",
            lane="robin-perf", findings=["slow"],
        )
        ka.record_conformance_verdict(
            conn, root, axis="a11y", verdict="fail",
            lane="robin-a11y", findings=["contrast ratio"],
        )
        created = ka.generate_epic_acceptances(conn)
    assert len(created) == 1


# ---------------------------------------------------------------------------
# Conformance verdicts appear in acceptance_packet when present
# ---------------------------------------------------------------------------


def test_conformance_verdicts_in_acceptance_packet(kanban_home):
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        ka.record_conformance_verdict(
            conn, root, axis="security", verdict="pass",
            lane="robin-sec", findings=[],
        )
        ka.record_conformance_verdict(
            conn, root, axis="perf", verdict="fail",
            lane="robin-perf", findings=["slow db queries"],
        )
        (acc_id,) = ka.generate_epic_acceptances(conn)
        approvals = kb.list_task_approvals(acc_id)
    artifacts = approvals[0]["artifacts"]
    assert "conformance_verdicts" in artifacts
    cv = artifacts["conformance_verdicts"]
    assert cv["security"]["verdict"] == "pass"
    assert cv["performance"]["verdict"] == "fail"
    assert "slow db queries" in cv["performance"]["findings"]


def test_conformance_verdicts_absent_when_no_events(kanban_home):
    """If no conformance events, the conformance_verdicts key is absent from packet."""
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        (acc_id,) = ka.generate_epic_acceptances(conn)
        approvals = kb.list_task_approvals(acc_id)
    artifacts = approvals[0]["artifacts"]
    assert "conformance_verdicts" not in artifacts


def test_conformance_verdict_body_mentions_advisory(kanban_home):
    """Advisory axes appear in the acceptance task body with advisory marker."""
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        ka.record_conformance_verdict(
            conn, root, axis="perf", verdict="fail",
            lane="robin-perf", findings=["p95 > 1s"],
        )
        (acc_id,) = ka.generate_epic_acceptances(conn)
        task = kb.get_task(conn, acc_id)
    assert "Conformance verdicts:" in (task.body or "")
    assert "advisory" in (task.body or "")
    assert "performance" in (task.body or "").lower()


# ---------------------------------------------------------------------------
# B4: Author-aware conformance guard
# ---------------------------------------------------------------------------

def test_author_aware_rejects_same_provider(kanban_home):
    """A conformance verdict where lane-provider == author-provider is rejected."""
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        # Simulate Claude-authored epic (default; no model_lane telemetry → "claude")
        # Try to record a verdict on the Claude lane (same as author → rejected)
        with pytest.raises(ValueError, match="conformance_verdict_rejected"):
            ka.record_conformance_verdict(
                conn, root, axis="security", verdict="pass",
                lane="claude_code",  # same as author_provider="claude" → rejected
            )


def test_author_aware_accepts_cross_provider(kanban_home):
    """Claude-authored epic accepts a Gemini conformance lane."""
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        # Claude-authored (default) → Gemini lane must be accepted
        ka.record_conformance_verdict(
            conn, root, axis="security", verdict="pass",
            lane="gemini_pro",  # cross-provider → accepted
        )
        verdicts = ka._harvest_conformance_verdicts(conn, root)
        assert verdicts["security"]["verdict"] == "pass"
        assert verdicts["security"]["lane_provider"] == "gemini"
        assert verdicts["security"]["author_provider"] == "claude"


def test_author_aware_gemini_authored_routes_to_claude(kanban_home):
    """Gemini-authored epic requires Claude lane; Gemini lane is rejected."""
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        # Inject a Gemini author telemetry event
        kb._append_event(conn, root, "story_completed", {
            "model_lane": "gemini_pro", "task_id": child_ids[0],
        })
        # Gemini lane should now be rejected (same provider)
        with pytest.raises(ValueError, match="conformance_verdict_rejected"):
            ka.record_conformance_verdict(
                conn, root, axis="security", verdict="pass",
                lane="gemini_pro",
            )
        # Claude lane should be accepted
        ka.record_conformance_verdict(
            conn, root, axis="security", verdict="pass",
            lane="claude_code",
        )
        verdicts = ka._harvest_conformance_verdicts(conn, root)
        assert verdicts["security"]["author_provider"] == "gemini"
        assert verdicts["security"]["lane_provider"] == "claude"


def test_author_aware_self_review_records_verdict_rejected_event(kanban_home):
    """Same-provider rejection emits a verdict_rejected event on the epic."""
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        with pytest.raises(ValueError):
            ka.record_conformance_verdict(
                conn, root, axis="a11y", verdict="pass",
                lane="claude_code",
            )
        events = kb.list_events(conn, root)
        kinds = [(getattr(e, "kind", "") or "") for e in events]
        assert "verdict_rejected" in kinds


# ---------------------------------------------------------------------------
# B5: High-risk cross-check
# ---------------------------------------------------------------------------

def test_high_risk_epic_awaits_crosscheck(kanban_home):
    """High-risk epic with only primary security verdict does not get G3."""
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        # Override title to include a high-risk keyword
        conn.execute("UPDATE tasks SET title = ? WHERE id = ?",
                     ("auth service epic", root))
        conn.commit()
        _finish_epic(conn, root, child_ids)
        ka.record_conformance_verdict(
            conn, root, axis="security", verdict="pass",
            lane="gemini_pro",
        )
        # No cross-check yet → no G3
        result = ka.generate_epic_acceptances(conn)
        assert result == []


def test_high_risk_epic_with_crosscheck_gets_g3(kanban_home):
    """High-risk epic with primary + cross-check both pass → G3 created."""
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        conn.execute("UPDATE tasks SET title = ? WHERE id = ?",
                     ("auth service epic", root))
        conn.commit()
        _finish_epic(conn, root, child_ids)
        ka.record_conformance_verdict(
            conn, root, axis="security", verdict="pass",
            lane="gemini_pro",
        )
        # Record cross-check from Claude lane (different provider)
        ka.record_conformance_verdict(
            conn, root, axis="security", verdict="pass",
            lane="claude_code",
            crosscheck=True,
        )
        result = ka.generate_epic_acceptances(conn)
        assert len(result) == 1


def test_crosscheck_same_provider_blocks_g3(kanban_home):
    """Cross-check from the same provider as primary is an independence violation."""
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        conn.execute("UPDATE tasks SET title = ? WHERE id = ?",
                     ("auth service epic", root))
        conn.commit()
        _finish_epic(conn, root, child_ids)
        # Primary: claude-authored epic reviewed by gemini
        ka.record_conformance_verdict(
            conn, root, axis="security", verdict="pass",
            lane="gemini_pro",
        )
        # Cross-check also from gemini → independence violation (bypass author check
        # by directly writing the event — simulates a forged or mis-routed verdict)
        kb._append_event(conn, root, "conformance_verdict_security_xcheck", {
            "verdict": "pass", "lane": "gemini_pro",
            "lane_provider": "gemini", "author_provider": "claude",
            "signed": True, "findings": [], "crosscheck": True,
        })
        result = ka.generate_epic_acceptances(conn)
        assert result == []
        events = kb.list_events(conn, root)
        kinds = [(getattr(e, "kind", "") or "") for e in events]
        assert "conformance_gate_block" in kinds


def test_routine_epic_does_not_require_crosscheck(kanban_home):
    """Routine (non-high-risk) epic gets G3 on a single passing security verdict."""
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        ka.record_conformance_verdict(
            conn, root, axis="security", verdict="pass",
            lane="gemini_pro",
        )
        result = ka.generate_epic_acceptances(conn)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# B6: WI-QA4 fix-story auto-spawn
# ---------------------------------------------------------------------------

def test_security_fail_spawns_fix_story(kanban_home):
    """Security FAIL auto-spawns a conformance fix story."""
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        ka.record_conformance_verdict(
            conn, root, axis="security", verdict="fail",
            lane="gemini_pro", findings=["SQL injection in login endpoint"],
        )
        ka.generate_epic_acceptances(conn)
        # Check that a fix story was spawned
        events = kb.list_events(conn, root)
        spawn_events = [
            e for e in events
            if (getattr(e, "kind", "") or "") == "conformance_fix_story_created"
        ]
        assert len(spawn_events) == 1
        payload = getattr(spawn_events[0], "payload", None) or {}
        if isinstance(payload, str):
            payload = json.loads(payload)
        assert payload.get("reason") == "security_fail"


def test_security_fail_fix_story_bounded_to_wi_qa4_max(kanban_home, monkeypatch):
    """After WI_QA4_MAX_CONFORMANCE_RETRIES spawns, escalate-once and stop."""
    monkeypatch.setattr(ka, "WI_QA4_MAX_CONFORMANCE_RETRIES", 2)
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        # Simulate having already spawned 2 fix stories
        kb._append_event(conn, root, "conformance_fix_story_created",
                         {"fix_task_id": "t_fake1", "reason": "security_fail", "retry_number": 1})
        kb._append_event(conn, root, "conformance_fix_story_created",
                         {"fix_task_id": "t_fake2", "reason": "security_fail", "retry_number": 2})
        ka.record_conformance_verdict(
            conn, root, axis="security", verdict="fail",
            lane="gemini_pro",
        )
        ka.generate_epic_acceptances(conn)
        events = kb.list_events(conn, root)
        escalations = [
            e for e in events
            if (getattr(e, "kind", "") or "") == "conformance_escalated"
        ]
        # Exactly one escalation emitted
        assert len(escalations) == 1
        # No new fix story spawned beyond the cap
        spawns = [
            e for e in events
            if (getattr(e, "kind", "") or "") == "conformance_fix_story_created"
        ]
        assert len(spawns) == 2  # still 2; no new spawn


def test_escalate_once_not_repeated(kanban_home, monkeypatch):
    """Escalation is emitted at most once even across multiple ticks."""
    monkeypatch.setattr(ka, "WI_QA4_MAX_CONFORMANCE_RETRIES", 1)
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        kb._append_event(conn, root, "conformance_fix_story_created",
                         {"fix_task_id": "t_fake1", "reason": "security_fail", "retry_number": 1})
        # Simulate escalation already emitted
        kb._append_event(conn, root, "conformance_escalated",
                         {"reason": "security_fail", "retry_count": 1, "max_retries": 1})
        ka.record_conformance_verdict(
            conn, root, axis="security", verdict="fail",
            lane="gemini_pro",
        )
        # Second tick — should not emit another escalation
        ka.generate_epic_acceptances(conn)
        events = kb.list_events(conn, root)
        escalations = [
            e for e in events
            if (getattr(e, "kind", "") or "") == "conformance_escalated"
        ]
        assert len(escalations) == 1  # still just the original one
