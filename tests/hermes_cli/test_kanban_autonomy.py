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
        # One-step accept release: record_task_acceptance now completes the
        # epic-acceptance gate atomically in the same txn (run-547 fix), so
        # the gate is already done before any sweep runs.
        kb.record_task_acceptance(conn, acc_id, "keith", source="cli")
        assert kb.get_task(conn, acc_id).status == "done"
        # Approval row flipped approved for audit symmetry.
        approvals = kb.list_task_approvals(acc_id)
        assert approvals[0]["status"] == "approved"
        # The sweep is now a no-op for an already-released gate (the gate no
        # longer matches the blocked/scheduled filter).
        out = ka.sweep_acceptance_tasks(conn)
        assert out["completed"] == []


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
        # Operator accepts the rework — one-step release completes it atomically.
        kb.record_task_acceptance(conn, acc_id, "keith", source="cli")
        assert kb.get_task(conn, acc_id).status == "done"
        # Sweep is a no-op now that the gate is already released.
        out4 = ka.sweep_acceptance_tasks(conn)
        assert out4["completed"] == []


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
            conn, integration_branch="integration", require_review=False
        )
        assert len(created) == 1
        integ = kb.get_task(conn, created[0])
        assert integ.assignee == "integrator"
        assert "feat/t11" in (integ.body or "")
        assert kb.get_task_metadata(created[0])["work_item_type"] == "integrate"
        # Dedup on second sweep.
        assert ka.find_unintegrated_done_tasks(
            conn, integration_branch="integration", require_review=False
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


def test_record_verdict_fusion_run_id_and_confidence_flow(kanban_home, verdict_key, vault):
    """Phase 6: a fusion-shaped verdict (lanes_run, fusion_run_id, confidence) records a PASS
    (R8 satisfied by lanes_run, no model_lane) and the id + confidence reach LoopState."""
    from hermes_cli.review_loop import state as S
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        payload = _payload(
            tid, model_lane=None,
            lane="fusion-abc1234500000000", second_opinion_lane="cx/gpt-5.5-review",
            lanes_run=["ag/gemini-3.1-pro-low", "cx/gpt-5.4-review", "cx/gpt-5.5-review"],
            fusion_run_id="fusion-abc1234500000000",
            confidence={"capacity": "full", "judge_confidence": 0.9, "composite": 0.9,
                        "missing": [], "divergence": 0.0},
        )
        sig = ka.sign_verdict_payload(payload.encode(), key_path=verdict_key)
        out = ka.record_review_verdict(
            conn, tid, payload, sig, fetched_via="robin-ssh",
            key_path=verdict_key, vault_root=str(vault),
        )
        assert out["ok"] is True and out["verdict"] == "pass"  # R8 satisfied by lanes_run
        st = S.compute_loop_state(conn, tid)
        assert st.fusion_run_id == "fusion-abc1234500000000"
        assert st.confidence and st.confidence["composite"] == 0.9


def test_record_verdict_fusion_hollow_pass_still_rejected(kanban_home, verdict_key, vault):
    """A fusion PASS with NO lanes_run and NO model_lane is still a hollow pass → rejected (R8)."""
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        payload = _payload(tid, model_lane=None, lanes_run=[],
                           fusion_run_id="fusion-deadbeef00000000")
        sig = ka.sign_verdict_payload(payload.encode(), key_path=verdict_key)
        out = ka.record_review_verdict(
            conn, tid, payload, sig, fetched_via="robin-ssh",
            key_path=verdict_key, vault_root=str(vault),
        )
        assert out["ok"] is False and "hollow" in out["reason"]


# --- Stage 3: advisory design_quality axis (Robin run 540 BLOCK fixes) -------------

from hermes_cli.review_loop import ninerouter as _NR  # noqa: E402


def test_design_quality_crosscheck_is_rejected():
    """ENFORCED invariant: an advisory axis can never be recorded on the crosscheck
    path. The guard fires before any DB access, so conn is irrelevant here."""
    with pytest.raises(ValueError, match="advisory"):
        ka.record_conformance_verdict(
            None, "epic1", "design_quality", "pass", lane="l", crosscheck=True
        )


def test_advisory_design_review_skips_non_ui(monkeypatch):
    """No wireframe approval with a selected direction -> returns None, records nothing."""
    monkeypatch.setattr(kb, "list_task_approvals", lambda task_id: [])
    called = []
    monkeypatch.setattr(ka, "record_conformance_verdict",
                        lambda *a, **k: called.append((a, k)))
    out = ka.run_advisory_design_review(
        None, "t1", "e1", chat=lambda *a, **k: _NR.ChatResult("x", "m", True, None), model="m"
    )
    assert out is None
    assert called == []


def test_advisory_design_review_records_advisory(monkeypatch):
    """UI task with an approved direction -> sources artifact, runs review, records an
    ADVISORY design_quality verdict (crosscheck=False, 'concerns' -> 'fail')."""
    approval = {
        "approval_type": "wireframe",
        "artifacts": [{
            "selected_direction_id": "dir_a", "direction_set_id": "ds_1",
            "operator_rationale": "Linear-style", "design_token_summary": "Inter; #0b0b0f",
        }],
    }
    monkeypatch.setattr(kb, "list_task_approvals", lambda task_id: [approval])
    rec = {}
    monkeypatch.setattr(ka, "record_conformance_verdict",
                        lambda conn, epic, axis, verdict, **k: rec.update(
                            dict(axis=axis, verdict=verdict, **k)))

    def chat(model, messages, **kw):
        return _NR.ChatResult(
            json.dumps({"verdict": "concerns", "scores": {}, "findings": ["flat"],
                        "summary": "x"}), "m", True, None)

    out = ka.run_advisory_design_review(None, "t1", "e1", chat=chat, model="m")
    assert out["verdict"] == "concerns"
    assert rec["axis"] == "design_quality"
    assert rec["verdict"] == "fail"           # concerns -> advisory fail
    assert rec["crosscheck"] is False          # never on the blocking path


def _stub_l1(monkeypatch):
    from hermes_cli.review_loop import l1_screen as _l1
    monkeypatch.setattr(
        _l1, "run_l1_screen",
        lambda *a, **k: _l1.L1Result(risk="routine", escalate=False,
                                     findings_count=0, summary="ok", model="m", ok=True),
    )


def test_l1_hook_runs_advisory_design_review(monkeypatch, kanban_home):
    """The advisory design review fires automatically from the L1 review-phase pass for a
    UI task with an approved direction (wired, not deferred)."""
    _stub_l1(monkeypatch)
    monkeypatch.setattr(ka, "_latest_selected_direction",
                        lambda tid: {"selected_direction_id": "dir_a"})
    monkeypatch.setattr(ka, "_epic_for_story", lambda conn, sid: "epic1")
    captured = {}
    monkeypatch.setattr(ka, "run_advisory_design_review",
                        lambda conn, tid, eid, **kw: captured.update(task=tid, epic=eid))
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="Build the landing page", assignee="builder")
        ka.run_l1_screen_for_review_task(conn, tid, l1_cfg={"model": "m"})
    assert captured.get("task") == tid
    assert captured.get("epic") == "epic1"


def test_l1_hook_advisory_failure_is_isolated(monkeypatch, kanban_home):
    """A crash in the advisory design pass must NOT break the L1 review (fail-open)."""
    _stub_l1(monkeypatch)
    monkeypatch.setattr(ka, "_latest_selected_direction",
                        lambda tid: {"selected_direction_id": "dir_a"})
    monkeypatch.setattr(ka, "_epic_for_story", lambda conn, sid: "epic1")
    def _boom(*a, **k):
        raise RuntimeError("design review exploded")
    monkeypatch.setattr(ka, "run_advisory_design_review", _boom)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="Build the dashboard", assignee="builder")
        ka.run_l1_screen_for_review_task(conn, tid, l1_cfg={"model": "m"})  # must not raise
        kinds = [e.kind for e in kb.list_events(conn, tid)]
    assert "l1_screen" in kinds


def test_l1_hook_advisory_is_deduped_per_direction(monkeypatch, kanban_home):
    """Direction-scoped idempotency: the SAME approved direction is not re-reviewed."""
    _stub_l1(monkeypatch)
    monkeypatch.setattr(ka, "_latest_selected_direction",
                        lambda tid: {"selected_direction_id": "dir_a"})
    monkeypatch.setattr(ka, "_epic_for_story", lambda conn, sid: "epic1")
    calls = []
    monkeypatch.setattr(ka, "run_advisory_design_review",
                        lambda conn, tid, eid, **kw: calls.append(tid))
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="Build the pricing page", assignee="builder")
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "design_review_advisory",
                             {"selected_direction_id": "dir_a", "epic_id": "epic1"})
        ka.run_l1_screen_for_review_task(conn, tid, l1_cfg={"model": "m"})
    assert calls == []  # same direction -> not re-invoked


def test_l1_hook_advisory_reruns_on_new_direction(monkeypatch, kanban_home):
    """A NEW/revised approved direction (Stage-2 re-pick) DOES get a fresh advisory review,
    even though an older direction was already reviewed."""
    _stub_l1(monkeypatch)
    monkeypatch.setattr(ka, "_latest_selected_direction",
                        lambda tid: {"selected_direction_id": "dir_b"})  # operator re-picked
    monkeypatch.setattr(ka, "_epic_for_story", lambda conn, sid: "epic1")
    calls = []
    monkeypatch.setattr(ka, "run_advisory_design_review",
                        lambda conn, tid, eid, **kw: calls.append(tid))
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="Rebuild the hero", assignee="builder")
        with kb.write_txn(conn):  # prior review was for dir_a
            kb._append_event(conn, tid, "design_review_advisory",
                             {"selected_direction_id": "dir_a", "epic_id": "epic1"})
        ka.run_l1_screen_for_review_task(conn, tid, l1_cfg={"model": "m"})
    assert calls == [tid]  # new direction -> re-reviewed


# ---------------------------------------------------------------------------
# v18 Supreme Court Review Contract (structured-output enforcement)
# ---------------------------------------------------------------------------


def _sc_payload(task_id, review_type="code", verdict="Approved", **kw):
    """A conforming v18 Supreme Court structured verdict.

    Carries every required structured field + a non-empty rubric scorecard +
    a recorded review lane (so R8 is satisfied) + a valid vault run_record.
    Individual tests override / drop fields to exercise the fail-closed paths.
    """
    body = {
        "task_id": task_id,
        "review_type": review_type,
        "verdict": verdict,
        "confidence": 0.9,
        "scorecard": {"clarity": 9, "completeness": 8, "threshold_met": True},
        "blocking_issues": [],
        "advisory_issues": [],
        "missing_skill_findings": [],
        "required_repair_actions": [],
        "evidence_reviewed": ["diff.patch", "review-request.md"],
        "calibration_substrate_flags": [],
        "run_record": "runs/2026-06-12-900-test-review.md",
        "model_lane": "claude-code",
    }
    body.update(kw)
    # Allow tests to DELETE a key by passing it as None-with-intent via a
    # sentinel list of keys to drop.
    for k in kw.pop("_drop", []) if isinstance(kw.get("_drop"), list) else []:
        body.pop(k, None)
    return json.dumps(body, sort_keys=True)


def _sc_record(conn, tid, payload, verdict_key, vault, fetched_via="robin-ssh"):
    sig = ka.sign_verdict_payload(payload.encode(), key_path=verdict_key)
    return ka.record_review_verdict(
        conn, tid, payload, sig, fetched_via=fetched_via,
        key_path=verdict_key, vault_root=str(vault),
    )


def test_sc_conforming_code_verdict_completes(kanban_home, verdict_key, vault):
    """A fully-conforming SC code verdict (Approved) completes the task."""
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        out = _sc_record(conn, tid, _sc_payload(tid), verdict_key, vault)
        assert out == {"ok": True, "verdict": "pass", "reason": None}
        assert kb.get_task(conn, tid).status == "done"
        ev = [e for e in kb.list_events(conn, tid) if e.kind == "verdict_recorded"]
        assert ev
        p = ev[-1].payload
        if isinstance(p, str):
            p = json.loads(p)
        assert p["review_type"] == "code"


def test_sc_approved_with_minor_notes_passes(kanban_home, verdict_key, vault):
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        out = _sc_record(
            conn, tid,
            _sc_payload(tid, verdict="Approved-with-minor-notes"),
            verdict_key, vault,
        )
        assert out["ok"] is True and out["verdict"] == "pass"


def test_sc_rejected_for_revision_blocks(kanban_home, verdict_key, vault):
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        out = _sc_record(
            conn, tid,
            _sc_payload(
                tid, verdict="Rejected-for-revision",
                blocking_issues=["missing error state on the form"],
            ),
            verdict_key, vault,
        )
        assert out["ok"] is True and out["verdict"] == "block"
        assert kb.get_task(conn, tid).status == "ready"


def test_sc_rejected_wrong_skill_stack_blocks(kanban_home, verdict_key, vault):
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        out = _sc_record(
            conn, tid,
            _sc_payload(tid, review_type="wireframe",
                        verdict="Rejected-wrong-skill-stack"),
            verdict_key, vault,
        )
        assert out["ok"] is True and out["verdict"] == "block"


@pytest.mark.parametrize("missing_field", [
    "confidence", "scorecard", "blocking_issues", "advisory_issues",
    "missing_skill_findings", "required_repair_actions", "evidence_reviewed",
    "calibration_substrate_flags",
])
def test_sc_missing_required_field_is_rejected(
    kanban_home, verdict_key, vault, missing_field
):
    """Dropping ANY required structured field => hollow SC verdict => rejected."""
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        body = json.loads(_sc_payload(tid))
        body.pop(missing_field)
        payload = json.dumps(body, sort_keys=True)
        out = _sc_record(conn, tid, payload, verdict_key, vault)
        assert out["ok"] is False
        assert "hollow SC verdict" in out["reason"]
        assert kb.get_task(conn, tid).status == "running"  # untouched, fail-closed
        assert "verdict_rejected" in [e.kind for e in kb.list_events(conn, tid)]


def test_sc_empty_scorecard_is_rejected(kanban_home, verdict_key, vault):
    """A present-but-empty scorecard for a scorecard-bearing type is hollow."""
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        out = _sc_record(conn, tid, _sc_payload(tid, scorecard={}),
                         verdict_key, vault)
        assert out["ok"] is False and "scorecard" in out["reason"]
        assert kb.get_task(conn, tid).status == "running"


def test_sc_non_list_field_is_rejected(kanban_home, verdict_key, vault):
    """A required list field that is not a JSON list is rejected (shape)."""
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        out = _sc_record(conn, tid,
                         _sc_payload(tid, blocking_issues="oops not a list"),
                         verdict_key, vault)
        assert out["ok"] is False and "must be a JSON list" in out["reason"]


def test_sc_bad_verdict_vocabulary_is_rejected(kanban_home, verdict_key, vault):
    """A structured SC verdict with an off-vocabulary verdict is rejected."""
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        out = _sc_record(conn, tid, _sc_payload(tid, verdict="LGTM"),
                         verdict_key, vault)
        assert out["ok"] is False and "v18 vocabulary" in out["reason"]


def test_sc_hollow_pass_no_lane_still_rejected(kanban_home, verdict_key, vault):
    """An otherwise-conforming SC PASS with NO review lane is still hollow (R8)."""
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        body = json.loads(_sc_payload(tid))
        body.pop("model_lane")  # no lane / lanes_run -> R8 hollow
        payload = json.dumps(body, sort_keys=True)
        out = _sc_record(conn, tid, payload, verdict_key, vault)
        assert out["ok"] is False and "hollow" in out["reason"]


def test_sc_all_review_types_enforced(kanban_home, verdict_key, vault):
    """Every SC review type enforces the contract (missing field -> reject)."""
    for rt in ("wireframe", "prd", "code", "final-delivery", "general"):
        with kb.connect() as conn:
            tid = _mk_review_task(conn)
            body = json.loads(_sc_payload(tid, review_type=rt))
            body.pop("scorecard")
            payload = json.dumps(body, sort_keys=True)
            out = _sc_record(conn, tid, payload, verdict_key, vault)
            assert out["ok"] is False, f"{rt} should fail-closed"
            assert "hollow SC verdict" in out["reason"]


def test_legacy_payload_without_review_type_is_unaffected(
    kanban_home, verdict_key, vault
):
    """A legacy/fusion verdict (no review_type) bypasses SC enforcement and
    still completes via the existing lane-based path — backward compatibility."""
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        # _payload() has model_lane + no review_type and no SC structured fields.
        payload = _payload(tid, verdict="pass")
        out = _sc_record(conn, tid, payload, verdict_key, vault)
        assert out == {"ok": True, "verdict": "pass", "reason": None}
        assert kb.get_task(conn, tid).status == "done"


def test_sc_unrecognized_review_type_is_not_enforced(
    kanban_home, verdict_key, vault
):
    """An unknown review_type is treated as 'not an SC contract' -> no structured
    enforcement; it falls through to the legacy verdict path."""
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        # review_type the SC contract doesn't know about + legacy-shaped body.
        payload = _payload(tid, verdict="pass", review_type="security-audit")
        out = _sc_record(conn, tid, payload, verdict_key, vault)
        assert out["ok"] is True and out["verdict"] == "pass"


def _sc_record_expecting(conn, tid, payload, verdict_key, vault, expected, fetched_via="robin-ssh"):
    sig = ka.sign_verdict_payload(payload.encode(), key_path=verdict_key)
    return ka.record_review_verdict(
        conn, tid, payload, sig, fetched_via=fetched_via,
        key_path=verdict_key, vault_root=str(vault),
        expected_review_type=expected,
    )


def test_sc_gate_rejects_verdict_missing_review_type(kanban_home, verdict_key, vault):
    """557 fix: with expected_review_type set, a verdict carrying NO review_type
    cannot clear an SC gate (the hollow freeform-chat PASS class)."""
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        payload = _payload(tid, verdict="pass")  # legacy-shaped, no review_type
        out = _sc_record_expecting(conn, tid, payload, verdict_key, vault, "wireframe")
        assert out["ok"] is False
        assert "review_type" in (out.get("reason") or "")
        assert kb.get_task(conn, tid).status != "done"


def test_sc_gate_rejects_review_type_mismatch(kanban_home, verdict_key, vault):
    """A verdict whose review_type is not the gate's type is rejected."""
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        payload = _sc_payload(tid, review_type="prd")  # conforming, but wrong type
        out = _sc_record_expecting(conn, tid, payload, verdict_key, vault, "wireframe")
        assert out["ok"] is False
        reason = out.get("reason") or ""
        assert "prd" in reason and "wireframe" in reason


def test_sc_gate_accepts_matching_conforming_verdict(kanban_home, verdict_key, vault):
    """A conforming wireframe verdict at a wireframe gate still passes."""
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        payload = _sc_payload(tid, review_type="wireframe")
        out = _sc_record_expecting(conn, tid, payload, verdict_key, vault, "wireframe")
        assert out == {"ok": True, "verdict": "pass", "reason": None}
        assert kb.get_task(conn, tid).status == "done"


def test_enforce_supreme_court_contract_unit():
    """Direct unit coverage of the pure validator (no DB)."""
    good = {
        "verdict": "Approved", "confidence": 0.9,
        "scorecard": {"x": 1},
        "blocking_issues": [], "advisory_issues": [],
        "missing_skill_findings": [], "required_repair_actions": [],
        "evidence_reviewed": [], "calibration_substrate_flags": [],
    }
    assert ka.enforce_supreme_court_contract(good, "code") is None
    bad = dict(good)
    del bad["scorecard"]
    assert "missing required structured" in ka.enforce_supreme_court_contract(bad, "code")
    # normalization of the review_type token
    assert ka._normalize_review_type("Final-Delivery") == "final-delivery"
    assert ka._normalize_review_type("CODE") == "code"
    assert ka._normalize_review_type("nope") is None
    assert ka._normalize_review_type(None) is None


# ---------------------------------------------------------------------------
# Per-review-type SC contract: every recognized review type must REJECT a
# non-structured verdict and ACCEPT a valid structured one (task t_f0ce733c).
# The wireframe / prd / code / final-delivery types are the v18 gate types the
# Supreme Court contract was authored for; this block locks each one down so a
# regression in one type can't slip through the broader sweep tests above.
# ---------------------------------------------------------------------------

# Canonical gate review types named in the task. "final-delivery" is exercised
# via both its hyphen and underscore spellings to prove _normalize_review_type
# folds the alias before enforcement.
SC_GATE_REVIEW_TYPES = ["wireframe", "prd", "code", "final-delivery"]


def _legacy_typed_payload(task_id, review_type, verdict="pass", **kw):
    """A NON-structured (legacy-shaped) verdict that nevertheless declares an
    SC review_type. It has model_lane + run_record (so it would clear the R8
    lane check) but NONE of the SC_REQUIRED_FIELDS — exactly the hollow
    structured verdict the contract must fail-closed on."""
    return _payload(task_id, verdict=verdict, review_type=review_type, **kw)


@pytest.mark.parametrize("review_type", SC_GATE_REVIEW_TYPES)
def test_sc_nonstructured_verdict_rejected_per_type(
    kanban_home, verdict_key, vault, review_type
):
    """For each gate review type, a verdict declaring that review_type but
    lacking the structured SC fields is rejected fail-closed and leaves task
    state untouched (no completion, no block — requeue for re-review)."""
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        payload = _legacy_typed_payload(tid, review_type)
        out = _sc_record(conn, tid, payload, verdict_key, vault)
        assert out["ok"] is False, f"{review_type} non-structured must reject"
        assert "hollow SC verdict" in out["reason"]
        # fail-closed: the task is neither completed nor blocked.
        assert kb.get_task(conn, tid).status == "running"
        kinds = [e.kind for e in kb.list_events(conn, tid)]
        assert "verdict_rejected" in kinds
        assert "verdict_recorded" not in kinds


@pytest.mark.parametrize("review_type", SC_GATE_REVIEW_TYPES)
def test_sc_valid_structured_verdict_accepted_per_type(
    kanban_home, verdict_key, vault, review_type
):
    """For each gate review type, a fully-conforming structured Approved verdict
    is accepted: the task completes and the recorded review_type is normalized
    to the canonical token."""
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        out = _sc_record(
            conn, tid, _sc_payload(tid, review_type=review_type),
            verdict_key, vault,
        )
        assert out == {"ok": True, "verdict": "pass", "reason": None}, review_type
        assert kb.get_task(conn, tid).status == "done"
        ev = [e for e in kb.list_events(conn, tid) if e.kind == "verdict_recorded"]
        assert ev, f"{review_type} should emit verdict_recorded"
        p = ev[-1].payload
        if isinstance(p, str):
            p = json.loads(p)
        assert p["review_type"] == ka._normalize_review_type(review_type)


@pytest.mark.parametrize("review_type", SC_GATE_REVIEW_TYPES)
def test_sc_valid_structured_rejection_verdict_blocks_per_type(
    kanban_home, verdict_key, vault, review_type
):
    """For each gate review type, a conforming Rejected-for-revision verdict is
    honored as a block: the task returns to 'ready' so the builder lane
    respawns with the findings, rather than completing."""
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        out = _sc_record(
            conn, tid,
            _sc_payload(
                tid, review_type=review_type, verdict="Rejected-for-revision",
                blocking_issues=["gate-specific blocking finding"],
            ),
            verdict_key, vault,
        )
        assert out["ok"] is True and out["verdict"] == "block", review_type
        assert kb.get_task(conn, tid).status == "ready"


@pytest.mark.parametrize("review_type", SC_GATE_REVIEW_TYPES)
@pytest.mark.parametrize("missing_field", list(ka.SC_REQUIRED_FIELDS))
def test_sc_each_missing_field_rejected_per_type(
    kanban_home, verdict_key, vault, review_type, missing_field
):
    """Cross-product: dropping ANY single required structured field from ANY
    gate review type is rejected fail-closed. Locks the full field x type
    matrix so no (type, field) combination silently degrades."""
    with kb.connect() as conn:
        tid = _mk_review_task(conn)
        body = json.loads(_sc_payload(tid, review_type=review_type))
        # task_id and review_type are not in SC_REQUIRED_FIELDS, so every
        # parametrized field is a genuine structured field that must be present.
        body.pop(missing_field)
        payload = json.dumps(body, sort_keys=True)
        out = _sc_record(conn, tid, payload, verdict_key, vault)
        assert out["ok"] is False, f"{review_type}/{missing_field} must reject"
        assert kb.get_task(conn, tid).status == "running"


def test_sc_final_delivery_underscore_alias_enforced(
    kanban_home, verdict_key, vault
):
    """The 'final_delivery' underscore spelling normalizes to the same SC type
    and is enforced identically to 'final-delivery' (no enforcement bypass via
    the alternate spelling)."""
    with kb.connect() as conn:
        # underscore spelling, non-structured -> rejected
        tid = _mk_review_task(conn)
        out = _sc_record(
            conn, tid, _legacy_typed_payload(tid, "final_delivery"),
            verdict_key, vault,
        )
        assert out["ok"] is False and "hollow SC verdict" in out["reason"]
    with kb.connect() as conn:
        # underscore spelling, structured -> accepted, normalized token recorded
        tid2 = _mk_review_task(conn)
        out2 = _sc_record(
            conn, tid2, _sc_payload(tid2, review_type="final_delivery"),
            verdict_key, vault,
        )
        assert out2 == {"ok": True, "verdict": "pass", "reason": None}
        ev = [e for e in kb.list_events(conn, tid2)
              if e.kind == "verdict_recorded"]
        p = ev[-1].payload
        if isinstance(p, str):
            p = json.loads(p)
        assert p["review_type"] == "final-delivery"

