"""Regression tests for Slice 4/5 incident (runs/2026-06-05-003).

These tests reconstruct the incident pattern where work was decomposed
and dispatched without operator approval artifacts on disk, and assert
that the gate-hardening measures (F1, F2, F3) catch it.

The F1 gate (PacketBeforeDispatchError in decompose_triage_task) prevents
decomposition of triage tasks that reference approval packets which don't
exist on disk.

The F2 gate (_has_gate_hold / R3_GATE_PHRASES) prevents auto-promotion of
tasks blocked with dependency/approval hold phrases.

The F3 packet-audit (conductor_vault/packet_audit) detects executed cards
with missing or unresolvable packet references.

See: APPROVAL-conductor-gate-hardening-2026-06-06.md  F4
"""

from __future__ import annotations

import json
import os
import sqlite3
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import kanban_db as kb

# F3 packet-audit module (installed as package, not in repo tree)
try:
    from conductor_vault import packet_audit as pa
except ImportError:
    pa = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper to skip F3 tests if packet_audit not available
# ---------------------------------------------------------------------------

_requires_pa = pytest.mark.skipif(
    pa is None,
    reason="conductor_vault.packet_audit not installed",
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACE", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create_task_with_body(conn, title, assignee, body, *,
                           status="todo", workspace_path=None):
    """Create a task with a specific body, optionally setting workspace_path.

    Note: create_task only accepts initial_status in {running, blocked}.
    For other statuses we create as 'running' then update.
    """
    init_status = status if status in kb.VALID_INITIAL_STATUSES else "running"
    tid = kb.create_task(conn, title=title, assignee=assignee,
                         body=body, initial_status=init_status)
    if status != init_status:
        conn.execute(
            "UPDATE tasks SET status = ? WHERE id = ?",
            (status, tid),
        )
    if workspace_path:
        conn.execute(
            "UPDATE tasks SET workspace_path = ? WHERE id = ?",
            (workspace_path, tid),
        )
    return tid


# ---------------------------------------------------------------------------
# T4.1 - Incident state fixtures
# ---------------------------------------------------------------------------

@_requires_pa
class TestIncidentStateFixtures:
    """Reconstruct the runs/2026-06-05-003 incident pattern.

    The Slice 4/5 incident: a triage card referenced an approval packet
    (APPROVAL-*.md) in its body, but the file did not exist on disk.
    The orchestrator decomposed it into child tasks that were dispatched
    and executed without any operator approval gate catching it.
    """

    def test_triage_card_references_missing_packet(self, kanban_home):
        """A triage card with a packet reference but no on-file packet
        reconstructs the root cause of the incident."""
        ws = kanban_home / "workspace"
        ws.mkdir()
        with kb.connect() as conn:
            tid = _create_task_with_body(
                conn,
                title="Triage: conductor gate hardening",
                assignee="orchestrator",
                body="Approval packet: docs/prds/APPROVAL-conductor-gate-hardening-2026-06-06.md",
                status="triage",
                workspace_path=str(ws),
            )
            row = conn.execute(
                "SELECT body, workspace_path FROM tasks WHERE id = ?",
                (tid,),
            ).fetchone()
            assert "APPROVAL-" in row["body"]
            # No packet file exists
            assert not (ws / "docs" / "prds" /
                        "APPROVAL-conductor-gate-hardening-2026-06-06.md").exists()

    def test_decompose_fans_into_children(self, kanban_home):
        """Decompose a triage task into 3 children (task/story/review)
        as the orchestrator did in the incident."""
        with kb.connect() as conn:
            tid = _create_task_with_body(
                conn,
                title="Triage: conductor gate hardening",
                assignee="orchestrator",
                body="Approval packet: docs/prds/APPROVAL-test.md",
                status="triage",
                workspace_path=str(kanban_home / "workspace"),
            )
            # Create the packet file so decompose succeeds
            prd_dir = kanban_home / "workspace" / "docs" / "prds"
            prd_dir.mkdir(parents=True)
            (prd_dir / "APPROVAL-test.md").write_text("# Approval\nApproved.")

            children = kb.decompose_triage_task(
                conn, tid,
                root_assignee="orchestrator",
                children=[
                    {"title": "Implement gate", "assignee": "builder"},
                    {"title": "Write tests", "assignee": "qa"},
                    {"title": "Review changes", "assignee": "reviewer"},
                ],
            )
            assert children is not None
            assert len(children) == 3

    def test_r3_blocked_task_with_packet_reference(self, kanban_home):
        """An R3-blocked task that references an approval packet —
        the F2 gate should hold it."""
        with kb.connect() as conn:
            tid = _create_task_with_body(
                conn,
                title="Deploy to production",
                assignee="ops",
                body="Approval packet: docs/prds/APPROVAL-prod-deploy.md",
                status="running",
            )
            kb.block_task(
                conn, tid,
                reason="Waiting on dependencies: need APPROVAL-prod-deploy.md",
            )
            assert kb._has_gate_hold(conn, tid) is True

    def test_completed_task_without_packet(self, kanban_home):
        """A completed task that never had a packet reference —
        this is the incident aftermath: work was done without approval trail."""
        with kb.connect() as conn:
            tid = _create_task_with_body(
                conn,
                title="Hotfix: emergency patch",
                assignee="builder",
                body="Apply critical security fix immediately.",
                status="done",
            )
            row = conn.execute(
                "SELECT body FROM tasks WHERE id = ?", (tid,),
            ).fetchone()
            refs = pa._extract_packet_refs(row["body"])
            assert len(refs) == 0  # No packet reference

    def test_grandfathered_task_with_valid_packet(self, kanban_home):
        """A grandfathered task with a valid packet is a control case —
        it should pass audit without findings."""
        ws = kanban_home / "vault"
        ws.mkdir()
        prd_dir = ws / "docs" / "prds"
        prd_dir.mkdir(parents=True)
        packet = "APPROVAL-grandfathered-test.md"
        (prd_dir / packet).write_text("# Retroactive approval\nGrandfathered.")

        # Create kanban DB with an executed card referencing the packet
        kanban_db_path = kanban_home / "kanban.db"
        monkeypatch_value = str(kanban_db_path)
        conn = sqlite3.connect(str(kanban_db_path))
        conn.row_factory = sqlite3.Row
        # Minimal schema for packet_audit
        conn.executescript(textwrap.dedent("""\
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                body TEXT DEFAULT '',
                status TEXT DEFAULT 'todo',
                created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
            );
            CREATE TABLE IF NOT EXISTS task_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                profile TEXT,
                status TEXT,
                started_at REAL,
                ended_at REAL,
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            );
            CREATE TABLE IF NOT EXISTS task_comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                author TEXT,
                body TEXT DEFAULT '',
                created_at REAL DEFAULT (strftime('%s','now')),
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            );
        """))
        conn.execute(
            "INSERT INTO tasks (id, title, body, status, created_at) VALUES (?, ?, ?, ?, strftime('%s','now'))",
            ("t_grandfathered", "Grandfathered feature", f"Approval: docs/prds/{packet}", "done"),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at) VALUES (?, ?, ?, strftime('%s','now'), strftime('%s','now'))",
            ("t_grandfathered", "builder", "done"),
        )
        conn.commit()
        conn.close()

        result = pa.run_packet_audit(kanban_db_path, ws)
        # The packet is on file, so no findings
        assert not result.has_findings()


# ---------------------------------------------------------------------------
# T4.2 - F1 gate (packet-before-decomposition)
# ---------------------------------------------------------------------------

class TestF1Gate:
    """Tests for the F1 gate: PacketBeforeDispatchError.

    When a triage task's body references an APPROVAL-*.md packet but the
    file does not exist on disk, decompose_triage_task MUST raise
    PacketBeforeDispatchError.
    """

    def test_decompose_without_packet_raises_error(self, kanban_home):
        """Decomposing a triage task whose body references a missing packet
        must raise PacketBeforeDispatchError."""
        ws = kanban_home / "workspace"
        ws.mkdir()
        with kb.connect() as conn:
            tid = _create_task_with_body(
                conn,
                title="Triage: needs approval",
                assignee="orchestrator",
                body="Approval packet: docs/prds/APPROVAL-missing.md",
                status="triage",
                workspace_path=str(ws),
            )
            # Packet file does NOT exist — this must raise
            with pytest.raises(kb.PacketBeforeDispatchError, match="Decomposition blocked"):
                kb.decompose_triage_task(
                    conn, tid,
                    root_assignee="orchestrator",
                    children=[
                        {"title": "Child 1", "assignee": "builder"},
                    ],
                )

    def test_decompose_with_packet_succeeds(self, kanban_home):
        """Decomposing a triage task whose packet IS on file must succeed."""
        ws = kanban_home / "workspace"
        ws.mkdir()
        prd_dir = ws / "docs" / "prds"
        prd_dir.mkdir(parents=True)
        (prd_dir / "APPROVAL-exists.md").write_text("# Approved\nYes.")

        with kb.connect() as conn:
            tid = _create_task_with_body(
                conn,
                title="Triage: has approval",
                assignee="orchestrator",
                body="Approval packet: docs/prds/APPROVAL-exists.md",
                status="triage",
                workspace_path=str(ws),
            )
            children = kb.decompose_triage_task(
                conn, tid,
                root_assignee="orchestrator",
                children=[
                    {"title": "Child 1", "assignee": "builder"},
                ],
            )
            assert children is not None
            assert len(children) == 1

    def test_decompose_no_packet_reference_skips_gate(self, kanban_home):
        """Triaging a task with NO packet reference in its body should
        skip the gate entirely (housekeeping tasks unaffected)."""
        ws = kanban_home / "workspace"
        ws.mkdir()
        with kb.connect() as conn:
            tid = _create_task_with_body(
                conn,
                title="Triage: housekeeping",
                assignee="orchestrator",
                body="Organize the backlog and triage old tickets.",
                status="triage",
                workspace_path=str(ws),
            )
            # No packet reference — must succeed without needing any file
            children = kb.decompose_triage_task(
                conn, tid,
                root_assignee="orchestrator",
                children=[
                    {"title": "Clean old tickets", "assignee": "builder"},
                ],
            )
            assert children is not None
            assert len(children) == 1

    def test_decompose_accepts_grandfathered_packets(self, kanban_home):
        """Decomposing with a grandfathered packet name on disk succeeds."""
        ws = kanban_home / "workspace"
        ws.mkdir()
        prd_dir = ws / "docs" / "prds"
        prd_dir.mkdir(parents=True)
        (prd_dir / "APPROVAL-conductor-infinite-brain-n8n-slice1-2026-06-05.md").write_text(
            "# Retroactive\nGrandfathered.",
        )

        with kb.connect() as conn:
            tid = _create_task_with_body(
                conn,
                title="Triage: grandfathered work",
                assignee="orchestrator",
                body="Approval packet: docs/prds/APPROVAL-conductor-infinite-brain-n8n-slice1-2026-06-05.md",
                status="triage",
                workspace_path=str(ws),
            )
            children = kb.decompose_triage_task(
                conn, tid,
                root_assignee="orchestrator",
                children=[
                    {"title": "Child 1", "assignee": "builder"},
                ],
            )
            assert children is not None
            assert len(children) == 1


# ---------------------------------------------------------------------------
# T4.3 - F2 gate (R3 gate hold)
# ---------------------------------------------------------------------------

class TestF2Gate:
    """Tests for the F2 gate: R3_GATE_PHRASES and _has_gate_hold.

    Tasks blocked with an R3-gate phrase must not be auto-promoted
    until the hold is satisfied (unblock signal in comments).
    """

    def test_dispatch_blocked_without_packet_refuses(self, kanban_home):
        """A task blocked with an R3-gate phrase has _has_gate_hold=True."""
        with kb.connect() as conn:
            tid = _create_task_with_body(
                conn,
                title="Deploy to prod",
                assignee="ops",
                body="Approval packet: docs/prds/APPROVAL-prod-deploy.md",
                status="running",
            )
            kb.block_task(
                conn, tid,
                reason="Waiting on dependencies: need APPROVAL-prod-deploy.md",
            )
            assert kb._has_gate_hold(conn, tid) is True

    def test_dispatch_blocked_with_packet_allows(self, kanban_home):
        """A task blocked with an R3-gate phrase that is satisfied
        (UNBLOCK: comment) should have _gate_hold_satisfied=True."""
        with kb.connect() as conn:
            tid = _create_task_with_body(
                conn,
                title="Deploy to prod",
                assignee="ops",
                body="Approval packet: docs/prds/APPROVAL-prod-deploy.md",
                status="running",
            )
            kb.block_task(
                conn, tid,
                reason="Waiting on dependencies: need APPROVAL-prod-deploy.md",
            )
            kb.add_comment(conn, tid, author="operator", body="UNBLOCK: approval packet: APPROVAL-prod-deploy.md signed")
            assert kb._gate_hold_satisfied(conn, tid) is True

    def test_r3_gate_detects_all_phrases(self, kanban_home):
        """F2 must detect all R3_GATE_PHRASES."""
        phrases = kb.R3_GATE_PHRASES
        assert len(phrases) >= 3  # At least waiting, needs approval, requires approval

        for phrase in phrases:
            with kb.connect() as conn:
                tid = kb.create_task(conn, title=f"Test {phrase[:20]}", assignee="a")
                # Task is created in "running" status (no parents → ready, but initial is running)
                # block_task transitions running -> blocked
                kb.block_task(conn, tid, reason=f"Blocked: {phrase}")
                assert kb._has_gate_hold(conn, tid) is True, (
                    f"Phrase '{phrase}' not detected by _has_gate_hold"
                )

    def test_r3_gate_no_false_positive_on_soft_blocks(self, kanban_home):
        """F2 must NOT false-positive on soft block reasons like
        'need clarification' or 'review-required'."""
        soft_reasons = [
            "review-required: needs code review",
            "need clarification on scope",
            "waiting for feedback from user",
            "blocked: dependency not resolved yet",
            "stuck on API rate limit",
        ]
        for reason in soft_reasons:
            with kb.connect() as conn:
                tid = kb.create_task(
                    conn, title=f"Soft block: {reason[:20]}", assignee="a",
                )
                # create_task starts tasks in running; block_task transitions to blocked
                kb.block_task(conn, tid, reason=reason)
                assert kb._has_gate_hold(conn, tid) is False, (
                    f"Soft reason '{reason}' should not trigger _has_gate_hold"
                )

    def test_ready_task_with_missing_packet_ref_has_hold(self, kanban_home):
        """A ready task that references a missing packet and has an
        R3-block event should have gate hold."""
        with kb.connect() as conn:
            tid = _create_task_with_body(
                conn,
                title="Feature: gate check",
                assignee="builder",
                body="Approval packet: docs/prds/APPROVAL-feature-gate.md",
                status="running",
            )
            # Manually set to blocked with R3 phrase, then try ready promotion
            kb.block_task(
                conn, tid,
                reason="Requires approval packet before dispatch",
            )
            assert kb._has_gate_hold(conn, tid) is True


# ---------------------------------------------------------------------------
# T4.4 - F3 packet-audit
# ---------------------------------------------------------------------------

@_requires_pa
class TestF3PacketAudit:
    """Tests for the F3 packet-audit: conductor_vault/packet_audit.

    The audit walks kanban cards and flags executed tasks with missing
    or unresolvable packet references.
    """

    @pytest.fixture
    def audit_db(self, tmp_path):
        """Create a minimal kanban DB for packet audit tests."""
        db_path = tmp_path / "audit.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.executescript(textwrap.dedent("""\
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                body TEXT DEFAULT '',
                status TEXT DEFAULT 'todo',
                created_at REAL NOT NULL DEFAULT (strftime('%s','now'))
            );
            CREATE TABLE IF NOT EXISTS task_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                profile TEXT,
                status TEXT,
                started_at REAL,
                ended_at REAL,
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            );
            CREATE TABLE IF NOT EXISTS task_comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                author TEXT,
                body TEXT DEFAULT '',
                created_at REAL DEFAULT (strftime('%s','now')),
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            );
        """))
        conn.commit()
        return db_path

    def test_audit_flags_missing_packet(self, tmp_path, audit_db):
        """Audit flags completed tasks without on-file packets."""
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "docs" / "prds").mkdir(parents=True)
        # NO approval packets on disk

        conn = sqlite3.connect(str(audit_db))
        conn.row_factory = sqlite3.Row
        conn.execute(
            "INSERT INTO tasks (id, title, body, status) VALUES (?, ?, ?, ?)",
            ("t1", "Completed without packet",
             "Approval packet: docs/prds/APPROVAL-missing.md", "done"),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at) VALUES (?, ?, ?, strftime('%s','now'), strftime('%s','now'))",
            ("t1", "builder", "done"),
        )
        conn.commit()
        conn.close()

        result = pa.run_packet_audit(audit_db, vault)
        assert result.has_findings()
        assert any(f.task_id == "t1" for f in result.findings)

    def test_audit_passes_with_on_file_packet(self, tmp_path, audit_db):
        """Audit passes tasks whose referenced packet IS on file."""
        vault = tmp_path / "vault"
        vault.mkdir()
        prd_dir = vault / "docs" / "prds"
        prd_dir.mkdir(parents=True)
        (prd_dir / "APPROVAL-valid.md").write_text("# Approved\nYes.")

        conn = sqlite3.connect(str(audit_db))
        conn.row_factory = sqlite3.Row
        conn.execute(
            "INSERT INTO tasks (id, title, body, status) VALUES (?, ?, ?, ?)",
            ("t1", "Has valid packet",
             "Approval packet: docs/prds/APPROVAL-valid.md", "done"),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at) VALUES (?, ?, ?, strftime('%s','now'), strftime('%s','now'))",
            ("t1", "builder", "done"),
        )
        conn.commit()
        conn.close()

        result = pa.run_packet_audit(audit_db, vault)
        assert not result.has_findings()

    def test_audit_handles_mixed_batch(self, tmp_path, audit_db):
        """Audit correctly handles a batch with both valid and missing packets."""
        vault = tmp_path / "vault"
        vault.mkdir()
        prd_dir = vault / "docs" / "prds"
        prd_dir.mkdir(parents=True)
        (prd_dir / "APPROVAL-valid.md").write_text("# Approved\nYes.")

        conn = sqlite3.connect(str(audit_db))
        conn.row_factory = sqlite3.Row
        # Task 1: has valid packet
        conn.execute(
            "INSERT INTO tasks (id, title, body, status) VALUES (?, ?, ?, ?)",
            ("t1", "Valid", "Approval packet: docs/prds/APPROVAL-valid.md", "done"),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at) VALUES (?, ?, ?, strftime('%s','now'), strftime('%s','now'))",
            ("t1", "builder", "done"),
        )
        # Task 2: missing packet
        conn.execute(
            "INSERT INTO tasks (id, title, body, status) VALUES (?, ?, ?, ?)",
            ("t2", "Missing", "Approval packet: docs/prds/APPROVAL-missing.md", "done"),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at) VALUES (?, ?, ?, strftime('%s','now'), strftime('%s','now'))",
            ("t2", "builder", "done"),
        )
        conn.commit()
        conn.close()

        result = pa.run_packet_audit(audit_db, vault)
        assert result.has_findings()
        # Only t2 should have a finding
        missing_ids = {f.task_id for f in result.findings}
        assert "t2" in missing_ids
        assert "t1" not in missing_ids

    def test_audit_skips_non_executed_tasks(self, tmp_path, audit_db):
        """Audit skips tasks that were never executed (no run record)."""
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "docs" / "prds").mkdir(parents=True)
        # NO approval packets on disk

        conn = sqlite3.connect(str(audit_db))
        conn.row_factory = sqlite3.Row
        # Task with no run record — should be skipped
        conn.execute(
            "INSERT INTO tasks (id, title, body, status) VALUES (?, ?, ?, ?)",
            ("t1", "Never executed",
             "Approval packet: docs/prds/APPROVAL-missing.md", "todo"),
        )
        conn.commit()
        conn.close()

        result = pa.run_packet_audit(audit_db, vault)
        # No executed tasks means no findings
        assert result.total_executed == 0
        assert not result.has_findings()

    def test_audit_flags_housekeeping_with_no_reference(self, tmp_path, audit_db):
        """Audit flags executed tasks without any packet reference as
        'missing_packet' — housekeeping tasks still need a reference
        once they've been executed ( dispatched to a worker)."""
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "docs" / "prds").mkdir(parents=True)

        conn = sqlite3.connect(str(audit_db))
        conn.row_factory = sqlite3.Row
        conn.execute(
            "INSERT INTO tasks (id, title, body, status) VALUES (?, ?, ?, ?)",
            ("t1", "Housekeeping", "Clean up old branches.", "done"),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at) VALUES (?, ?, ?, strftime('%s','now'), strftime('%s','now'))",
            ("t1", "builder", "done"),
        )
        conn.commit()
        conn.close()

        result = pa.run_packet_audit(audit_db, vault)
        # Housekeeping task has no packet reference — it gets a finding
        # (F3 flags executed cards without references as "missing_packet")
        assert result.has_findings()
        # The finding type should be "missing_packet" (no reference at all)
        assert result.findings[0].finding_type == "missing_packet"
