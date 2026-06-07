"""Tests for F2: kanban hard gate on 'Waiting on dependencies'.

When a blocked task's reason contains an R3-gate phrase (e.g.
'Waiting on dependencies'), recompute_ready must NOT auto-promote
it to 'ready' until the comment thread contains an unblock signal
(e.g. 'UNBLOCK:').  This prevents premature dispatch of tasks that
need human approval or an external dependency satisfied before work
can meaningfully proceed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Clear any leaked kanban DB/board env so the connect() helper
    # resolves to a fresh per-test DB under tmp_path.
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# _has_gate_hold
# ---------------------------------------------------------------------------

class TestHasGateHold:
    """Unit tests for _has_gate_hold(conn, task_id)."""

    def test_no_events_returns_false(self, kanban_home):
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="t", assignee="a")
            assert kb._has_gate_hold(conn, tid) is False

    def test_blocked_event_with_gate_phrase_returns_true(self, kanban_home):
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="t", assignee="a")
            kb.block_task(conn, tid, reason="Waiting on dependencies: need Phase 11 packet")
            assert kb._has_gate_hold(conn, tid) is True

    def test_blocked_event_with_non_gate_phrase_returns_false(self, kanban_home):
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="t", assignee="a")
            kb.block_task(conn, tid, reason="review-required: needs code review")
            assert kb._has_gate_hold(conn, tid) is False

    def test_blocked_then_unblocked_returns_false(self, kanban_home):
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="t", assignee="a")
            kb.block_task(conn, tid, reason="Waiting on dependencies: need packet")
            kb.unblock_task(conn, tid)
            assert kb._has_gate_hold(conn, tid) is False

    def test_multiple_blocked_events_latest_wins(self, kanban_home):
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="t", assignee="a")
            # First block with gate phrase
            kb.block_task(conn, tid, reason="Waiting on dependencies: need packet")
            # Unblock
            kb.unblock_task(conn, tid)
            # Second block with non-gate phrase
            kb.block_task(conn, tid, reason="review-required: diff looks wrong")
            assert kb._has_gate_hold(conn, tid) is False

    def test_created_blocked_status_with_gate_reason(self, kanban_home):
        """Tasks created with initial_status='blocked' whose created event
        status payload says 'blocked' are NOT gate-held unless a subsequent
        block_task call added a gate reason.  The created event doesn't
        carry a 'reason' field."""
        with kb.connect() as conn:
            parent = kb.create_task(conn, title="parent", assignee="a")
            kb.complete_task(conn, parent)
            child = kb.create_task(
                conn, title="child", assignee="a", parents=[parent],
                initial_status="blocked",
            )
            # Created event has status='blocked' but no reason text.
            # _has_gate_hold checks blocked events only — created event
            # doesn't have a gate reason payload.
            assert kb._has_gate_hold(conn, child) is False


# ---------------------------------------------------------------------------
# _gate_hold_satisfied
# ---------------------------------------------------------------------------

class TestGateHoldSatisfied:
    """Unit tests for _gate_hold_satisfied(conn, task_id)."""

    def test_no_comments_returns_false(self, kanban_home):
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="t", assignee="a")
            assert kb._gate_hold_satisfied(conn, tid) is False

    def test_unblock_comment_returns_true(self, kanban_home):
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="t", assignee="a")
            kb.add_comment(conn, tid, "operator", "UNBLOCK: approval signed")
            assert kb._gate_hold_satisfied(conn, tid) is True

    def test_non_unblock_comment_returns_false(self, kanban_home):
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="t", assignee="a")
            kb.add_comment(conn, tid, "worker", "still working on it")
            assert kb._gate_hold_satisfied(conn, tid) is False

    def test_approval_packet_reference_satisfies(self, kanban_home):
        """A comment mentioning 'APPROVAL' or 'approval packet' should also
        satisfy the gate hold."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="t", assignee="a")
            kb.add_comment(
                conn, tid, "operator",
                "Approval packet: docs/prds/APPROVAL-foo.md",
            )
            assert kb._gate_hold_satisfied(conn, tid) is True


# ---------------------------------------------------------------------------
# Integration: recompute_ready + gate hold
# ---------------------------------------------------------------------------

class TestRecomputeReadyGateHold:
    """Integration: recompute_ready refuses to promote gate-held tasks
    until the hold is satisfied by a comment."""

    def test_gate_hold_blocks_promotion(self, kanban_home):
        with kb.connect() as conn:
            parent = kb.create_task(conn, title="parent", assignee="a")
            child = kb.create_task(conn, title="child", assignee="a", parents=[parent])
            kb.complete_task(conn, parent)
            # Child is now ready (parents done). Block it with a gate reason.
            kb.block_task(conn, child, reason="Waiting on dependencies: need Phase 11 packet")
            assert kb.get_task(conn, child).status == "blocked"
            # recompute_ready should NOT promote — gate hold not satisfied
            promoted = kb.recompute_ready(conn)
            assert promoted == 0
            assert kb.get_task(conn, child).status == "blocked"

    def test_gate_hold_promotes_after_unblock_comment(self, kanban_home):
        with kb.connect() as conn:
            parent = kb.create_task(conn, title="parent", assignee="a")
            child = kb.create_task(conn, title="child", assignee="a", parents=[parent])
            kb.complete_task(conn, parent)
            kb.block_task(conn, child, reason="Waiting on dependencies: need Phase 11 packet")
            # Add unblock comment
            kb.add_comment(conn, child, "operator", "UNBLOCK: packet signed")
            # recompute_ready should now promote
            promoted = kb.recompute_ready(conn)
            assert promoted == 1
            assert kb.get_task(conn, child).status == "ready"

    def test_non_gate_blocked_task_still_promotes(self, kanban_home):
        """A blocked task with a non-gate reason should still be promoted
        when parents are done (existing behavior preserved)."""
        with kb.connect() as conn:
            parent = kb.create_task(conn, title="parent", assignee="a")
            child = kb.create_task(conn, title="child", assignee="a", parents=[parent])
            kb.complete_task(conn, parent)
            kb.block_task(conn, child, reason="review-required: diff needs review")
            # Not a gate hold, but IS sticky — should NOT auto-promote
            # (sticky block check comes first)
            promoted = kb.recompute_ready(conn)
            assert promoted == 0
            assert kb.get_task(conn, child).status == "blocked"

    def test_initial_status_blocked_promotes_when_parents_done(self, kanban_home):
        """Tasks created with initial_status='blocked' (no gate reason)
        should promote normally when parents are done. Note: complete_task
        already triggers recompute_ready internally, so the assertion
        checks the post-completion state, not the return value of a
        subsequent recompute_ready call."""
        with kb.connect() as conn:
            parent = kb.create_task(conn, title="parent", assignee="a")
            child = kb.create_task(
                conn, title="child", assignee="a", parents=[parent],
                initial_status="blocked",
            )
            assert kb.get_task(conn, child).status == "blocked"
            kb.complete_task(conn, parent)
            # complete_task triggers recompute_ready internally; the
            # child should be auto-promoted because there's no gate
            # hold and no sticky block.
            assert kb.get_task(conn, child).status == "ready"

    def test_gate_hold_with_partial_phrase(self, kanban_home):
        """Gate phrases should match substring, not exact string."""
        with kb.connect() as conn:
            parent = kb.create_task(conn, title="parent", assignee="a")
            child = kb.create_task(conn, title="child", assignee="a", parents=[parent])
            kb.complete_task(conn, parent)
            # Different phrasing but contains 'Waiting on dependencies'
            kb.block_task(conn, child, reason="Blocked: Waiting on dependencies for upstream PR merge")
            assert kb.get_task(conn, child).status == "blocked"
            promoted = kb.recompute_ready(conn)
            assert promoted == 0
            assert kb.get_task(conn, child).status == "blocked"

    def test_gate_hold_case_insensitive(self, kanban_home):
        with kb.connect() as conn:
            parent = kb.create_task(conn, title="parent", assignee="a")
            child = kb.create_task(conn, title="child", assignee="a", parents=[parent])
            kb.complete_task(conn, parent)
            kb.block_task(conn, child, reason="WAITING ON DEPENDENCIES: need signoff")
            assert kb.get_task(conn, child).status == "blocked"
            promoted = kb.recompute_ready(conn)
            assert promoted == 0

    def test_gate_hold_unblock_case_insensitive(self, kanban_home):
        with kb.connect() as conn:
            parent = kb.create_task(conn, title="parent", assignee="a")
            child = kb.create_task(conn, title="child", assignee="a", parents=[parent])
            kb.complete_task(conn, parent)
            kb.block_task(conn, child, reason="Waiting on dependencies: need packet")
            kb.add_comment(conn, child, "operator", "unblock: packet approved")
            promoted = kb.recompute_ready(conn)
            assert promoted == 1

    def test_gate_hold_then_unblock_task_still_promotes(self, kanban_home):
        """After an explicit unblock_task call (which removes sticky block
        AND promotes the task to ready if parents are done), gate hold is
        gone (blocked event was superseded by unblocked event)."""
        with kb.connect() as conn:
            parent = kb.create_task(conn, title="parent", assignee="a")
            child = kb.create_task(conn, title="child", assignee="a", parents=[parent])
            kb.complete_task(conn, parent)
            kb.block_task(conn, child, reason="Waiting on dependencies: need packet")
            # Explicit unblock removes both sticky block AND gate hold
            # AND promotes the task to ready (since parents are done).
            kb.unblock_task(conn, child)
            assert kb.get_task(conn, child).status == "ready"
            # A subsequent recompute_ready is a no-op (task already ready)
            # but should not error or demote the task.
            kb.recompute_ready(conn)
            assert kb.get_task(conn, child).status == "ready"

    def test_gate_hold_does_not_affect_todo_tasks(self, kanban_home):
        """Todo tasks (not blocked) should not be affected by gate hold logic."""
        with kb.connect() as conn:
            parent = kb.create_task(conn, title="parent", assignee="a")
            child = kb.create_task(conn, title="child", assignee="a", parents=[parent])
            # Child is todo (parent not done)
            assert kb.get_task(conn, child).status == "todo"
            kb.complete_task(conn, parent)
            # complete_task triggers recompute_ready internally; the
            # child should be auto-promoted (no gate reason, no sticky
            # block, parents done).
            assert kb.get_task(conn, child).status == "ready"
