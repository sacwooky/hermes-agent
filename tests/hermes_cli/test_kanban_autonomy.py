"""Tests for the three-gate autonomy module (kanban_autonomy).

Covers: verdict signing/verification + the provenance ladder, epic
acceptance generation/resolution (G3), rejection → fix story → re-arm,
decompose work_item_type stamping, and the unintegrated-work sweep.
LLM-free by design.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_autonomy as ka
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def verdict_key(tmp_path):
    key = tmp_path / "robin-verdict-key"
    key.write_bytes(b"test-shared-hmac-key-0123456789abcdef")
    return str(key)


@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "vault"
    (root / "runs").mkdir(parents=True)
    record = root / "runs" / "2026-06-12-900-test-review.md"
    record.write_text("# run record\nverdict evidence\n")
    return root


def _payload(task_id, vault_record="runs/2026-06-12-900-test-review.md", **kw):
    body = {
        "task_id": task_id,
        "verdict": kw.pop("verdict", "pass"),
        "run_record": vault_record,
        "model_lane": "claude-code",
        "findings": kw.pop("findings", []),
    }
    body.update(kw)
    return json.dumps(body, sort_keys=True)


# ---------------------------------------------------------------------------
# Verdict signing / verification
# ---------------------------------------------------------------------------


def test_sign_verify_roundtrip(verdict_key):
    payload = b'{"verdict": "pass"}'
    sig = ka.sign_verdict_payload(payload, key_path=verdict_key)
    assert ka.verify_verdict_signature(payload, sig, key_path=verdict_key)
    # Tampered payload fails.
    assert not ka.verify_verdict_signature(
        b'{"verdict": "PASS"}', sig, key_path=verdict_key
    )
    # Tampered signature fails.
    assert not ka.verify_verdict_signature(payload, "00" * 32, key_path=verdict_key)


def test_verify_missing_key_is_false(tmp_path):
    assert not ka.verify_verdict_signature(
        b"x", "aa", key_path=str(tmp_path / "missing")
    )


def _mk_review_task(conn, status="running"):
    tid = kb.create_task(conn, title="review me", assignee="builder")
    if status == "running":
        kb.claim_task(conn, tid)
    return tid


def test_record_verdict_refuses_local_file_channel(kanban_home, verdict_key, vault):
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        payload = _payload(tid)
        sig = ka.sign_verdict_payload(payload.encode(), key_path=verdict_key)
        out = ka.record_review_verdict(
            conn, tid, payload, sig,
            fetched_via="local-file",
            key_path=verdict_key, vault_root=str(vault),
        )
        assert out["ok"] is False
        assert "out-of-band" in out["reason"]
        events = [e.kind for e in kb.list_events(conn, tid)]
        assert "verdict_rejected" in events
        assert kb.get_task(conn, tid).status == "running"  # untouched


def test_record_verdict_rejects_bad_signature(kanban_home, verdict_key, vault):
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        payload = _payload(tid)
        out = ka.record_review_verdict(
            conn, tid, payload, "deadbeef" * 8,
            fetched_via="robin-api",
            key_path=verdict_key, vault_root=str(vault),
        )
        assert out["ok"] is False
        assert "HMAC" in out["reason"]


def test_record_verdict_rejects_missing_run_record(kanban_home, verdict_key, vault):
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        payload = _payload(tid, vault_record="runs/does-not-exist.md")
        sig = ka.sign_verdict_payload(payload.encode(), key_path=verdict_key)
        out = ka.record_review_verdict(
            conn, tid, payload, sig,
            fetched_via="robin-api",
            key_path=verdict_key, vault_root=str(vault),
        )
        assert out["ok"] is False
        assert "does not exist" in out["reason"]


def test_record_verdict_rejects_task_mismatch(kanban_home, verdict_key, vault):
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        payload = _payload("t_someoneelse")
        sig = ka.sign_verdict_payload(payload.encode(), key_path=verdict_key)
        out = ka.record_review_verdict(
            conn, tid, payload, sig,
            fetched_via="robin-api",
            key_path=verdict_key, vault_root=str(vault),
        )
        assert out["ok"] is False
        assert "does not match" in out["reason"]


def test_record_verdict_pass_completes_task(kanban_home, verdict_key, vault):
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        payload = _payload(tid, verdict="pass")
        sig = ka.sign_verdict_payload(payload.encode(), key_path=verdict_key)
        out = ka.record_review_verdict(
            conn, tid, payload, sig,
            fetched_via="robin-api",
            key_path=verdict_key, vault_root=str(vault),
        )
        assert out == {"ok": True, "verdict": "pass", "reason": None}
        task = kb.get_task(conn, tid)
        assert task.status == "done"
        events = [e.kind for e in kb.list_events(conn, tid)]
        assert "verdict_recorded" in events


def test_record_verdict_block_returns_to_ready_with_findings(
    kanban_home, verdict_key, vault
):
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        payload = _payload(tid, verdict="block", findings=["auth missing on /api/x"])
        sig = ka.sign_verdict_payload(payload.encode(), key_path=verdict_key)
        out = ka.record_review_verdict(
            conn, tid, payload, sig,
            fetched_via="robin-ssh",
            key_path=verdict_key, vault_root=str(vault),
        )
        assert out["ok"] is True and out["verdict"] == "block"
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        comments = kb.list_comments(conn, tid)
        assert any("auth missing on /api/x" in c.body for c in comments)


def test_record_verdict_pinned_hash_mismatch(kanban_home, verdict_key, vault):
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        payload = _payload(tid, run_record_sha256="ab" * 32)
        sig = ka.sign_verdict_payload(payload.encode(), key_path=verdict_key)
        out = ka.record_review_verdict(
            conn, tid, payload, sig,
            fetched_via="robin-api",
            key_path=verdict_key, vault_root=str(vault),
        )
        assert out["ok"] is False
        assert "sha256" in out["reason"]


# ---------------------------------------------------------------------------
# Decompose stamping + epic acceptance (G3)
# ---------------------------------------------------------------------------


def _decompose_epic(conn):
    root = kb.create_task(conn, title="Epic: member portal", triage=True)
    children = [
        {"title": "build dashboard", "assignee": "builder", "parents": []},
        {"title": "qa dashboard", "assignee": "qa", "parents": [0],
         "work_item_type": "qa"},
    ]
    child_ids = kb.decompose_triage_task(
        conn, root, root_assignee="orchestrator", children=children,
        author="orchestrator",
    )
    return root, child_ids


def test_decompose_stamps_work_item_types(kanban_home):
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
    assert kb.get_task_metadata(root)["work_item_type"] == "epic"
    assert kb.get_task_metadata(child_ids[0])["work_item_type"] == "story"
    assert kb.get_task_metadata(child_ids[1])["work_item_type"] == "qa"


def _finish_epic(conn, root, child_ids):
    for cid in child_ids:
        kb.complete_task(conn, cid, result="done",
                         metadata={"demo_url": "http://demo:9120/"})
    # Root promoted by recompute; complete it (orchestrator judge).
    kb.complete_task(conn, root, result="epic complete")


def test_epic_acceptance_generated_once(kanban_home):
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        created = ka.generate_epic_acceptances(conn)
        assert len(created) == 1
        acc = kb.get_task(conn, created[0])
        assert acc.status == "blocked"
        assert acc.assignee == "keith"
        assert kb.get_task_metadata(created[0])["work_item_type"] == "acceptance"
        # Approval row exists, pending.
        approvals = kb.list_task_approvals(created[0])
        assert approvals and approvals[0]["status"] == "pending"
        assert approvals[0]["approval_type"] == "epic_acceptance"
        # Demo URL harvested into the approval artifacts.
        assert "demo:9120" in json.dumps(approvals[0]["artifacts"])
        # Idempotent: second tick creates nothing.
        assert ka.generate_epic_acceptances(conn) == []
        # Sticky-blocked: recompute never promotes it.
        kb.recompute_ready(conn)
        assert kb.get_task(conn, created[0]).status == "blocked"


def test_acceptance_sweep_completes_on_accept(kanban_home):
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        (acc_id,) = ka.generate_epic_acceptances(conn)
        kb.record_task_acceptance(conn, acc_id, "keith", source="cli")
        out = ka.sweep_acceptance_tasks(conn)
        assert out["completed"] == [acc_id]
        assert kb.get_task(conn, acc_id).status == "done"
        # Approval row flipped approved for audit symmetry.
        approvals = kb.list_task_approvals(acc_id)
        assert approvals[0]["status"] == "approved"


def test_acceptance_rejection_spawns_fix_story_and_rearms(kanban_home):
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        (acc_id,) = ka.generate_epic_acceptances(conn)
        approval_id = kb.list_task_approvals(acc_id)[0]["approval_id"]
    kb.decide_task_approval(
        approval_id, decision="rejected", approver="keith",
        comment="nav is broken on mobile",
    )
    with kb.connect() as conn:
        out = ka.sweep_acceptance_tasks(conn)
        assert len(out["fix_stories"]) == 1
        fix_id = out["fix_stories"][0]
        fix = kb.get_task(conn, fix_id)
        assert fix.assignee == "builder"
        assert "nav is broken on mobile" in (fix.body or "")
        # Second sweep does not duplicate the fix story.
        out2 = ka.sweep_acceptance_tasks(conn)
        assert out2["fix_stories"] == []
        # Fix lands → acceptance re-armed with a fresh pending approval.
        kb.complete_task(conn, fix_id, result="fixed nav")
        out3 = ka.sweep_acceptance_tasks(conn)
        assert out3["rearmed"] == [acc_id]
        approvals = kb.list_task_approvals(acc_id)
        # (list_task_approvals orders by created_at only; same-second
        # creation makes index 0 ambiguous — assert on the set instead)
        assert any(a["status"] == "pending" for a in approvals)
        # Operator accepts the rework.
        kb.record_task_acceptance(conn, acc_id, "keith", source="cli")
        out4 = ka.sweep_acceptance_tasks(conn)
        assert out4["completed"] == [acc_id]


def test_autonomy_tick_respects_feature_gates(kanban_home):
    with kb.connect() as conn:
        root, child_ids = _decompose_epic(conn)
        _finish_epic(conn, root, child_ids)
        # Disabled config: nothing happens.
        out = ka.run_autonomy_tick(conn, cfg={})
        assert out == {}
        assert ka.generate_epic_acceptances.__name__  # module intact
        # Enabled: acceptance generated.
        out = ka.run_autonomy_tick(
            conn, cfg={"epic_acceptance": {"enabled": True}}
        )
        assert len(out["acceptances_created"]) == 1


# ---------------------------------------------------------------------------
# Unintegrated-work sweep
# ---------------------------------------------------------------------------


def _git(args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init", "-b", "integration"], repo)
    _git(["config", "user.email", "t@t"], repo)
    _git(["config", "user.name", "t"], repo)
    (repo / "a.txt").write_text("base\n")
    _git(["add", "."], repo)
    _git(["commit", "-m", "base"], repo)
    # Feature branch with an unmerged commit.
    _git(["checkout", "-b", "feat/t11"], repo)
    (repo / "b.txt").write_text("feature\n")
    _git(["add", "."], repo)
    _git(["commit", "-m", "feature work"], repo)
    _git(["checkout", "integration"], repo)
    return repo


def test_unintegrated_sweep_creates_integrate_task(kanban_home, git_repo):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="T11 edit fields", assignee="builder",
            workspace_kind="worktree", workspace_path=str(git_repo),
            branch_name="feat/t11",
        )
        kb.complete_task(conn, tid, result="done on worktree")
        created = ka.find_unintegrated_done_tasks(
            conn, integration_branch="integration"
        )
        assert len(created) == 1
        integ = kb.get_task(conn, created[0])
        assert integ.assignee == "integrator"
        assert "feat/t11" in (integ.body or "")
        assert kb.get_task_metadata(created[0])["work_item_type"] == "integrate"
        # Dedup on second sweep.
        assert ka.find_unintegrated_done_tasks(
            conn, integration_branch="integration"
        ) == []


def test_integrated_branch_is_skipped(kanban_home, git_repo):
    _git(["merge", "feat/t11"], git_repo)
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="merged story", assignee="builder",
            workspace_kind="worktree", workspace_path=str(git_repo),
            branch_name="feat/t11",
        )
        kb.complete_task(conn, tid, result="done")
        assert ka.find_unintegrated_done_tasks(
            conn, integration_branch="integration"
        ) == []
