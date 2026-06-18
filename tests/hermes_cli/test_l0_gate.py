"""Tests for ADD-ON C v2 WI-C3 — the L0 deterministic gate.

Two layers, all LLM-free:
- Pure executor (``run_l0_gate``): real shell builtins (``true``/``false``/``sleep``)
  against a tmp dir, no DB.
- Policy + wiring (``run_l0_gate_for_review_task`` + the dispatch hook): the
  ``kanban_home`` fixture, asserting on emitted events / task status.

Verdict-path safety: none of these touch ``record_review_verdict`` /
``record_conformance_verdict``. The only state transition L0 makes is review→ready
(the same transition a rejected review uses) or →blocked on exhaust.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_autonomy as ka
from hermes_cli import kanban_db as kb
from hermes_cli.review_loop import l0_gate as l0


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _check(name, command, *, required=True, ctype="test"):
    return {"name": name, "command": command, "type": ctype, "required": required}


def _make_review_task(conn, tmp_path, *, title="build feature"):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    tid = kb.create_task(
        conn, title=title, assignee="builder",
        workspace_kind="dir", workspace_path=str(ws), initial_status="running",
    )
    # Move to 'review' the way a worker does (create_task only allows running/blocked).
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'review', claim_lock = NULL, "
            "current_run_id = NULL WHERE id = ?",
            (tid,),
        )
    return tid, ws


def _kinds(conn, tid):
    return [(getattr(e, "kind", "") or "") for e in kb.list_events(conn, tid)]


def _payload(conn, tid, kind):
    for e in kb.list_events(conn, tid):
        if (getattr(e, "kind", "") or "") == kind:
            raw = getattr(e, "payload", None)
            return json.loads(raw) if isinstance(raw, str) else (raw or {})
    return None


def _set_status(conn, tid, status):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, tid))


# ---------------------------------------------------------------------------
# 1. Pure executor (no DB)
# ---------------------------------------------------------------------------


def test_run_l0_gate_all_pass(tmp_path):
    res = l0.run_l0_gate(tmp_path, [_check("tests", "true"), _check("lint", "true")])
    assert res.passed is True
    assert res.failed_required == []
    assert all(c.exit_code == 0 and c.passed for c in res.checks)


def test_run_l0_gate_required_fail(tmp_path):
    res = l0.run_l0_gate(tmp_path, [_check("tests", "true"), _check("lint", "false")])
    assert res.passed is False
    assert res.failed_required == ["lint"]


def test_run_l0_gate_advisory_fail_does_not_fail_gate(tmp_path):
    res = l0.run_l0_gate(
        tmp_path,
        [_check("tests", "true"), _check("sast", "false", required=False)],
    )
    assert res.passed is True
    assert res.failed_required == []
    assert res.failed_advisory == ["sast"]


def test_run_l0_gate_timeout(tmp_path):
    res = l0.run_l0_gate(tmp_path, [_check("slow", "sleep 5")], timeout_s=1)
    c = res.checks[0]
    assert c.timed_out is True
    assert c.exit_code is None
    assert c.passed is False
    assert res.passed is False


def test_run_l0_gate_truncates_log(tmp_path):
    big = 'python3 -c "print(\'x\' * 100000)"'
    res = l0.run_l0_gate(tmp_path, [_check("noisy", big)], log_tail_bytes=2048)
    assert len(res.checks[0].truncated_log.encode("utf-8")) <= 2048 + 64  # marker slack


def test_run_l0_gate_empty_command_fails(tmp_path):
    res = l0.run_l0_gate(tmp_path, [_check("nada", "")])
    assert res.passed is False
    assert res.checks[0].passed is False


def test_run_l0_gate_no_checks_is_vacuous_pass(tmp_path):
    res = l0.run_l0_gate(tmp_path, [])
    assert res.passed is True
    assert res.checks == []


# ---------------------------------------------------------------------------
# 2. Policy: run_l0_gate_for_review_task
# ---------------------------------------------------------------------------


def test_pass_emits_attestation_and_proceeds(kanban_home, tmp_path):
    with kb.connect() as conn:
        tid, _ = _make_review_task(conn, tmp_path)
        settled = ka.run_l0_gate_for_review_task(
            conn, tid, l0_cfg={"checks": [_check("tests", "true")]},
        )
        assert settled is False  # review proceeds
        kinds = _kinds(conn, tid)
        assert "l0_attestation" in kinds
        assert "l0_gate_passed" in kinds
        att = _payload(conn, tid, "l0_attestation")
        assert att["attested_by"] == "l0_harness"
        assert att["passed"] is True


def test_fail_records_attestation_too(kanban_home, tmp_path):
    with kb.connect() as conn:
        tid, _ = _make_review_task(conn, tmp_path)
        ka.run_l0_gate_for_review_task(
            conn, tid, l0_cfg={"checks": [_check("tests", "false")]},
        )
        att = _payload(conn, tid, "l0_attestation")
        assert att is not None and att["passed"] is False


def test_fail_routes_to_ready(kanban_home, tmp_path):
    with kb.connect() as conn:
        tid, _ = _make_review_task(conn, tmp_path)
        settled = ka.run_l0_gate_for_review_task(
            conn, tid, l0_cfg={"checks": [_check("tests", "false")], "max_retries": 3},
        )
        assert settled is True  # no review token spent
        kinds = _kinds(conn, tid)
        assert "l0_gate_failed" in kinds
        assert "l0_gate_fix_requested" in kinds
        task = kb.get_task(conn, tid)
        assert task.status == "ready"


def test_retry_bounded_then_escalate_once(kanban_home, tmp_path):
    cfg = {"checks": [_check("tests", "false")], "max_retries": 2}
    with kb.connect() as conn:
        tid, _ = _make_review_task(conn, tmp_path)
        # attempt 1 (prior 0): 1 < 2 → routed to ready
        ka.run_l0_gate_for_review_task(conn, tid, l0_cfg=cfg)
        assert kb.get_task(conn, tid).status == "ready"
        _set_status(conn, tid, "review")
        # attempt 2 (prior 1): 2 < 2 is False → escalate once
        ka.run_l0_gate_for_review_task(conn, tid, l0_cfg=cfg)
        assert _kinds(conn, tid).count("l0_gate_escalated") == 1
        # further tick: already escalated → held, no second escalation, no re-run
        _set_status(conn, tid, "review")
        settled = ka.run_l0_gate_for_review_task(conn, tid, l0_cfg=cfg)
        assert settled is True
        assert _kinds(conn, tid).count("l0_gate_escalated") == 1


def test_on_exhaust_block_flips_blocked(kanban_home, tmp_path):
    cfg = {"checks": [_check("tests", "false")], "max_retries": 1, "on_exhaust": "block"}
    with kb.connect() as conn:
        tid, _ = _make_review_task(conn, tmp_path)
        ka.run_l0_gate_for_review_task(conn, tid, l0_cfg=cfg)  # prior 0 → 1<1 False → escalate
        assert "l0_gate_escalated" in _kinds(conn, tid)
        assert kb.get_task(conn, tid).status == "blocked"


def test_pass_dedup_no_rerun(kanban_home, tmp_path, monkeypatch):
    cfg = {"checks": [_check("tests", "true")]}
    with kb.connect() as conn:
        tid, _ = _make_review_task(conn, tmp_path)
        ka.run_l0_gate_for_review_task(conn, tid, l0_cfg=cfg)  # passes
        # second call must NOT re-run the executor
        calls = {"n": 0}
        real = l0.run_l0_gate

        def _spy(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr("hermes_cli.review_loop.l0_gate.run_l0_gate", _spy)
        settled = ka.run_l0_gate_for_review_task(conn, tid, l0_cfg=cfg)
        assert settled is False
        assert calls["n"] == 0  # deduped via l0_gate_passed


def test_no_checks_proceeds_silently(kanban_home, tmp_path):
    with kb.connect() as conn:
        tid, _ = _make_review_task(conn, tmp_path)
        settled = ka.run_l0_gate_for_review_task(conn, tid, l0_cfg={"checks": []})
        assert settled is False
        assert "l0_attestation" not in _kinds(conn, tid)


def test_builder_cannot_self_certify(kanban_home, tmp_path):
    """A failing check fails the gate regardless of any worker claim, and the
    attestation provenance is the harness — never the builder."""
    with kb.connect() as conn:
        tid, _ = _make_review_task(conn, tmp_path)
        # Even if a worker stamped a 'completed' claim, the gate ignores it and
        # runs the real check.
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "completed", {"summary": "L0 passes, trust me"})
        settled = ka.run_l0_gate_for_review_task(
            conn, tid, l0_cfg={"checks": [_check("tests", "false")]},
        )
        assert settled is True  # gate failed despite the worker claim
        assert _payload(conn, tid, "l0_attestation")["attested_by"] == "l0_harness"


def test_catchrate_metric_written_on_fail(kanban_home, tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    monkeypatch.setenv("HERMES_LEARNING_VAULT_ROOT", str(vault))
    with kb.connect() as conn:
        tid, _ = _make_review_task(conn, tmp_path)
        ka.run_l0_gate_for_review_task(
            conn, tid, l0_cfg={"checks": [_check("tests", "false")], "max_retries": 3},
        )
    stream = vault / "metrics" / "autonomy" / "l0_gate.jsonl"
    assert stream.exists()
    rows = [json.loads(ln) for ln in stream.read_text().splitlines() if ln.strip()]
    assert rows and rows[-1]["settled_at_l0"] is True and rows[-1]["passed"] is False


def test_catchrate_metric_noop_without_env(kanban_home, tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_LEARNING_VAULT_ROOT", raising=False)
    with kb.connect() as conn:
        tid, _ = _make_review_task(conn, tmp_path)
        # Must not raise even though the env is unset.
        ka.run_l0_gate_for_review_task(
            conn, tid, l0_cfg={"checks": [_check("tests", "false")]},
        )


# ---------------------------------------------------------------------------
# 3. Dispatch hook (gated, fail-open)
# ---------------------------------------------------------------------------


def _fake_spawn(*args, **kwargs):
    return 4242


def test_dispatch_hook_disabled_is_noop(kanban_home, tmp_path, all_assignees_spawnable):
    """With l0_gate absent/disabled, a review task dispatches as today; no l0 events."""
    with kb.connect() as conn:
        tid, _ = _make_review_task(conn, tmp_path)
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, autonomy_cfg={})
        assert "l0_attestation" not in _kinds(conn, tid)
        assert any(s[0] == tid for s in res.spawned)


def test_dispatch_hook_fail_blocks_review_spawn(kanban_home, tmp_path, all_assignees_spawnable):
    """With l0_gate enabled + a failing check, the review task is routed to the
    fix-loop and NOT spawned for review this tick."""
    spawned_ids = []

    def _spy_spawn(task, *a, **k):
        spawned_ids.append(task.id)
        return 4242

    cfg = {"l0_gate": {"enabled": True, "checks": [_check("tests", "false")], "max_retries": 3}}
    with kb.connect() as conn:
        tid, _ = _make_review_task(conn, tmp_path)
        res = kb.dispatch_once(conn, spawn_fn=_spy_spawn, autonomy_cfg=cfg)
        assert tid not in spawned_ids
        assert not any(s[0] == tid for s in res.spawned)
        assert "l0_gate_failed" in _kinds(conn, tid)
