"""Tests for ADD-ON C v2 Phase 4 — L1 cheap screen (non-binding triage, WI-C4).

HTTP is mocked (LLM-free). Asserts: routine→no-escalate, risky→escalate, fail-OPEN to
escalate on any error, deterministic high-risk floor, and — critically — that L1 is
NON-BINDING: it records a signal but never blocks review and never writes a verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_autonomy as ka
from hermes_cli import kanban_db as kb
from hermes_cli.review_loop import l1_screen as L1
from hermes_cli.review_loop import state as S


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("NINEROUTER_KEY", "test-key")
    kb.init_db()
    return home


def _review_task(conn, tmp_path, *, title="build feature"):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    tid = kb.create_task(conn, title=title, assignee="builder",
                         workspace_kind="dir", workspace_path=str(ws), initial_status="running")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='review', claim_lock=NULL, current_run_id=NULL WHERE id=?", (tid,))
    return tid


def _kinds(conn, tid):
    return [(getattr(e, "kind", "") or "") for e in kb.list_events(conn, tid)]


def _payload(conn, tid, kind):
    for e in kb.list_events(conn, tid):
        if (getattr(e, "kind", "") or "") == kind:
            raw = getattr(e, "payload", None)
            return json.loads(raw) if isinstance(raw, str) else (raw or {})
    return None


# ---------------------------------------------------------------------------
# Executor (mocked HTTP)
# ---------------------------------------------------------------------------


def test_routine_clean_no_escalate(monkeypatch):
    monkeypatch.setenv("NINEROUTER_KEY", "k")
    monkeypatch.setattr(L1, "_post_chat", lambda *a, **k:
                        '{"risk":"routine","escalate":false,"findings_count":0,"summary":"ok"}')
    r = L1.run_l1_screen("trivial change")
    assert r.ok and r.escalate is False and r.risk == "routine"


def test_risky_escalates(monkeypatch):
    monkeypatch.setenv("NINEROUTER_KEY", "k")
    monkeypatch.setattr(L1, "_post_chat", lambda *a, **k:
                        '{"risk":"risky","escalate":true,"findings_count":3,"summary":"auth change"}')
    r = L1.run_l1_screen("auth change")
    assert r.escalate is True and r.findings_count == 3


def test_risk_risky_forces_escalate_even_if_flag_false(monkeypatch):
    monkeypatch.setenv("NINEROUTER_KEY", "k")
    monkeypatch.setattr(L1, "_post_chat", lambda *a, **k:
                        '{"risk":"risky","escalate":false,"findings_count":0,"summary":"x"}')
    assert L1.run_l1_screen("x").escalate is True  # risk==risky ⇒ escalate


def test_parse_failure_fails_open(monkeypatch):
    monkeypatch.setenv("NINEROUTER_KEY", "k")
    monkeypatch.setattr(L1, "_post_chat", lambda *a, **k: "not json at all")
    r = L1.run_l1_screen("x")
    assert r.ok is False and r.escalate is True  # fail-OPEN


def test_network_error_fails_open(monkeypatch):
    import urllib.error
    monkeypatch.setenv("NINEROUTER_KEY", "k")

    def _boom(*a, **k):
        raise urllib.error.URLError("conn refused")

    monkeypatch.setattr(L1, "_post_chat", _boom)
    r = L1.run_l1_screen("x")
    assert r.ok is False and r.escalate is True


def test_missing_key_fails_open(monkeypatch):
    monkeypatch.delenv("NINEROUTER_KEY", raising=False)
    r = L1.run_l1_screen("x")
    assert r.ok is False and r.escalate is True


def test_force_escalate_floor_overrides_routine(monkeypatch):
    monkeypatch.setenv("NINEROUTER_KEY", "k")
    monkeypatch.setattr(L1, "_post_chat", lambda *a, **k:
                        '{"risk":"routine","escalate":false,"findings_count":0,"summary":"ok"}')
    r = L1.run_l1_screen("x", force_escalate=True)
    assert r.escalate is True and r.risk == "risky"


def test_extract_json_embedded_in_prose():
    obj = L1._extract_json('Sure! {"risk":"routine","escalate":false} done')
    assert obj["risk"] == "routine"


# ---------------------------------------------------------------------------
# Policy: run_l1_screen_for_review_task (NON-BINDING)
# ---------------------------------------------------------------------------


def _mock_l1(monkeypatch, *, escalate, risk="routine", capture=None):
    def _fake(artifact, **kw):
        if capture is not None:
            capture.update(kw)
        return L1.L1Result(risk=risk, escalate=escalate, findings_count=0,
                          summary="m", model="ag/gemini-3-flash", ok=True)
    monkeypatch.setattr(L1, "run_l1_screen", _fake)


def test_record_emits_nonbinding_event(kanban_home, tmp_path, monkeypatch):
    _mock_l1(monkeypatch, escalate=False)
    with kb.connect() as conn:
        tid = _review_task(conn, tmp_path)
        ka.run_l1_screen_for_review_task(conn, tid, l1_cfg={})
        p = _payload(conn, tid, "l1_screen")
        assert p is not None and p["binding"] is False and p["escalate"] is False


def test_l1_never_blocks_review(kanban_home, tmp_path, monkeypatch):
    _mock_l1(monkeypatch, escalate=True, risk="risky")
    with kb.connect() as conn:
        tid = _review_task(conn, tmp_path)
        ret = ka.run_l1_screen_for_review_task(conn, tid, l1_cfg={})
        assert ret is None                       # returns nothing to block on
        assert kb.get_task(conn, tid).status == "review"  # not moved
        # no verdict written
        assert "verdict_recorded" not in _kinds(conn, tid)


def test_l1_dedup(kanban_home, tmp_path, monkeypatch):
    calls = {"n": 0}

    def _fake(artifact, **kw):
        calls["n"] += 1
        return L1.L1Result("routine", False, 0, "m", "ag/gemini-3-flash", True)

    monkeypatch.setattr(L1, "run_l1_screen", _fake)
    with kb.connect() as conn:
        tid = _review_task(conn, tmp_path)
        ka.run_l1_screen_for_review_task(conn, tid, l1_cfg={})
        ka.run_l1_screen_for_review_task(conn, tid, l1_cfg={})
        assert calls["n"] == 1  # deduped


def test_high_risk_title_forces_escalate_floor(kanban_home, tmp_path, monkeypatch):
    cap = {}
    _mock_l1(monkeypatch, escalate=False, capture=cap)
    with kb.connect() as conn:
        tid = _review_task(conn, tmp_path, title="auth token service")
        ka.run_l1_screen_for_review_task(conn, tid, l1_cfg={})
        assert cap.get("force_escalate") is True  # deterministic floor passed through


def test_loop_state_reads_l1_clean(kanban_home, tmp_path, monkeypatch):
    _mock_l1(monkeypatch, escalate=False)
    with kb.connect() as conn:
        tid = _review_task(conn, tmp_path)
        ka.run_l1_screen_for_review_task(conn, tid, l1_cfg={})
        assert S.compute_loop_state(conn, tid).l1_passed is True


def test_loop_state_reads_l1_escalated(kanban_home, tmp_path, monkeypatch):
    _mock_l1(monkeypatch, escalate=True, risk="risky")
    with kb.connect() as conn:
        tid = _review_task(conn, tmp_path)
        ka.run_l1_screen_for_review_task(conn, tid, l1_cfg={})
        assert S.compute_loop_state(conn, tid).l1_passed is False


# ---------------------------------------------------------------------------
# dispatch hook (gated, non-binding)
# ---------------------------------------------------------------------------


def _fake_spawn(*a, **k):
    return 4242


def test_dispatch_l1_disabled_noop(kanban_home, tmp_path, monkeypatch, all_assignees_spawnable):
    _mock_l1(monkeypatch, escalate=True)
    with kb.connect() as conn:
        tid = _review_task(conn, tmp_path)
        kb.dispatch_once(conn, spawn_fn=_fake_spawn, autonomy_cfg={})
        assert "l1_screen" not in _kinds(conn, tid)


def test_dispatch_l1_enabled_records_and_still_spawns(kanban_home, tmp_path, monkeypatch, all_assignees_spawnable):
    """L1 is non-binding: even on escalate, the review STILL spawns (L1 only records)."""
    _mock_l1(monkeypatch, escalate=True, risk="risky")
    cfg = {"l1_screen": {"enabled": True}}
    with kb.connect() as conn:
        tid = _review_task(conn, tmp_path)
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, autonomy_cfg=cfg)
        assert "l1_screen" in _kinds(conn, tid)
        assert any(s[0] == tid for s in res.spawned)  # review still spawned
