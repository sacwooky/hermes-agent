"""Tests for ADD-ON C v2 Phase 3 — derived review-loop state (WI-C2) + G3-only (WI-C9).

The loop state is derived purely from the kanban event log, so it is resumable and
idempotent by construction. WI-C9: the only human surfaces on the per-artifact path are
a sticky block, a budget pause, and G3 — a normal in-flight artifact surfaces nothing.
LLM-free.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.review_loop import state as S


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _task(conn):
    return kb.create_task(conn, title="artifact", assignee="builder")


def _ev(conn, tid, kind, payload=None):
    with kb.write_txn(conn):
        kb._append_event(conn, tid, kind, payload or {})


# ---------------------------------------------------------------------------
# Stage derivation
# ---------------------------------------------------------------------------


def test_stage_build_before_completion(kanban_home):
    with kb.connect() as conn:
        tid = _task(conn)
        assert S.compute_loop_state(conn, tid).stage == S.BUILD


def test_stage_l0_after_build(kanban_home):
    with kb.connect() as conn:
        tid = _task(conn)
        _ev(conn, tid, "completed", {"summary": "built"})
        assert S.compute_loop_state(conn, tid).stage == S.L0


def test_stage_l0_passed(kanban_home):
    with kb.connect() as conn:
        tid = _task(conn)
        _ev(conn, tid, "completed")
        _ev(conn, tid, "l0_gate_passed", {})
        st = S.compute_loop_state(conn, tid)
        assert st.stage == S.L0 and st.l0_passed is True


def test_stage_queued_on_l0_fail(kanban_home):
    with kb.connect() as conn:
        tid = _task(conn)
        _ev(conn, tid, "completed")
        _ev(conn, tid, "l0_gate_failed", {"attempt": 1})
        st = S.compute_loop_state(conn, tid)
        assert st.stage == S.QUEUED and st.l0_failures == 1


def test_stage_blocked_on_l0_escalated(kanban_home):
    with kb.connect() as conn:
        tid = _task(conn)
        _ev(conn, tid, "completed")
        _ev(conn, tid, "l0_gate_failed")
        _ev(conn, tid, "l0_gate_escalated", {})
        st = S.compute_loop_state(conn, tid)
        assert st.stage == S.BLOCKED
        assert S.SURFACE_STICKY_BLOCK in st.surfaces


def test_stage_l2_on_verdict_pass(kanban_home):
    with kb.connect() as conn:
        tid = _task(conn)
        _ev(conn, tid, "completed")
        _ev(conn, tid, "l0_gate_passed")
        _ev(conn, tid, "verdict_recorded", {"verdict": "pass", "model_lane": "gemini"})
        st = S.compute_loop_state(conn, tid)
        assert st.stage == S.L2 and st.l2_verdict == "pass"


def test_stage_queued_on_review_block(kanban_home):
    with kb.connect() as conn:
        tid = _task(conn)
        _ev(conn, tid, "completed")
        _ev(conn, tid, "review_blocked", {})
        assert S.compute_loop_state(conn, tid).stage == S.QUEUED


def test_stage_queued_on_verdict_rejected(kanban_home):
    with kb.connect() as conn:
        tid = _task(conn)
        _ev(conn, tid, "completed")
        _ev(conn, tid, "verdict_rejected", {"reason": "signed-empty"})
        st = S.compute_loop_state(conn, tid)
        assert st.stage == S.QUEUED and st.l2_rejected is True


def test_stage_blocked_on_conformance(kanban_home):
    with kb.connect() as conn:
        tid = _task(conn)
        _ev(conn, tid, "completed")
        _ev(conn, tid, "conformance_gate_block", {"reason": "security_fail"})
        assert S.compute_loop_state(conn, tid).stage == S.BLOCKED


def test_stage_integrated(kanban_home):
    with kb.connect() as conn:
        tid = _task(conn)
        _ev(conn, tid, "completed")
        _ev(conn, tid, "verdict_recorded", {"verdict": "pass", "model_lane": "gemini"})
        _ev(conn, tid, "integrate_task_created", {"integrate_id": "t_x"})
        assert S.compute_loop_state(conn, tid).stage == S.INTEGRATED


def test_fusion_run_id_slot_captured(kanban_home):
    with kb.connect() as conn:
        tid = _task(conn)
        _ev(conn, tid, "completed")
        _ev(conn, tid, "verdict_recorded",
            {"verdict": "pass", "model_lane": "gemini", "fusion_run_id": "fr-123"})
        assert S.compute_loop_state(conn, tid).fusion_run_id == "fr-123"


def test_latest_verdict_wins_after_refix(kanban_home):
    with kb.connect() as conn:
        tid = _task(conn)
        _ev(conn, tid, "completed")
        _ev(conn, tid, "review_blocked", {})          # first review blocked
        _ev(conn, tid, "completed", {"summary": "fixed"})
        _ev(conn, tid, "verdict_recorded", {"verdict": "pass", "model_lane": "gemini"})
        assert S.compute_loop_state(conn, tid).stage == S.L2  # latest pass wins


# ---------------------------------------------------------------------------
# WI-C9: human surfaces
# ---------------------------------------------------------------------------


def test_no_human_surface_midloop(kanban_home):
    """A normal in-flight artifact (build → L0 pass → review pass) surfaces NO human."""
    with kb.connect() as conn:
        tid = _task(conn)
        _ev(conn, tid, "completed")
        _ev(conn, tid, "l0_gate_passed")
        _ev(conn, tid, "verdict_recorded", {"verdict": "pass", "model_lane": "gemini"})
        assert S.compute_loop_state(conn, tid).surfaces == []


def test_g3_is_a_human_surface(kanban_home):
    with kb.connect() as conn:
        tid = _task(conn)
        _ev(conn, tid, "completed")
        _ev(conn, tid, "acceptance_task_created", {"epic_id": tid})
        st = S.compute_loop_state(conn, tid)
        assert st.g3_pending is True and S.SURFACE_G3 in st.surfaces


def test_g3_cleared_after_accept(kanban_home):
    with kb.connect() as conn:
        tid = _task(conn)
        _ev(conn, tid, "completed")
        _ev(conn, tid, "acceptance_task_created", {"epic_id": tid})
        _ev(conn, tid, "accepted", {})
        st = S.compute_loop_state(conn, tid)
        assert st.g3_pending is False and S.SURFACE_G3 not in st.surfaces


def test_budget_pause_is_a_human_surface(kanban_home):
    with kb.connect() as conn:
        tid = _task(conn)
        _ev(conn, tid, "pause_for_approval", {"reason": "over cap"})
        assert S.SURFACE_BUDGET_PAUSE in S.compute_loop_state(conn, tid).surfaces


def test_resumable_deterministic(kanban_home):
    with kb.connect() as conn:
        tid = _task(conn)
        _ev(conn, tid, "completed")
        _ev(conn, tid, "l0_gate_passed")
        a = S.compute_loop_state(conn, tid)
        b = S.compute_loop_state(conn, tid)
        assert (a.stage, a.l0_passed, a.surfaces) == (b.stage, b.l0_passed, b.surfaces)


# ---------------------------------------------------------------------------
# WI-C9 packet annotation
# ---------------------------------------------------------------------------


def test_loop_capacity_full():
    cap = S.loop_capacity({"security": {"verdict": "pass"}})
    assert cap["capacity"] == "full" and cap["degraded"] is False


def test_loop_capacity_unknown_when_absent():
    cap = S.loop_capacity(None)
    assert cap["degraded"] is True and cap["capacity"] == "unknown"


def test_loop_capacity_degraded_on_fail():
    cap = S.loop_capacity({"security": {"verdict": "fail"}})
    assert cap["degraded"] is True and cap["capacity"] == "degraded"
