"""Slice 2 wiring proof — cli._run_kanban_conductor_q against a REAL board.

Deterministic: a fake session agent (no LLM) whose run_conversation returns a
'DONE:' message, driving real kanban_db lifecycle (recompute_ready, list_tasks,
claim_task, complete_task, add_comment, build_worker_context). Proves the
conductor entrypoint actually moves cards to done on a real DB.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_conductor import drive_board_from_cli


def _judge(verdict, reason="ok"):
    # Fake aux judge so tests never hit a real model.
    return lambda goal, resp: (verdict, reason, False, None)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


class _FakeAgent:
    def __init__(self, session_id, response):
        self.session_id = session_id
        self._response = response
        self.calls = 0

    def run_conversation(self, *, user_message, conversation_history):
        self.calls += 1
        return {"final_response": self._response}


class _FakeCli:
    def __init__(self, response="DONE: implemented and verified"):
        self.session_id = "sess-conductor"
        self.conversation_history = []
        self.agent = _FakeAgent(self.session_id, response)


def test_conductor_entrypoint_drives_real_cards_to_done(kanban_home, monkeypatch):
    with kb.connect() as c:
        kb.create_task(c, title="Card A", body="do A", assignee="builder")
        kb.create_task(c, title="Card B", body="do B", assignee="builder")

    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setenv("HERMES_PROFILE", "conductor")
    monkeypatch.setenv("HERMES_KANBAN_CONDUCTOR", "1")

    fake = _FakeCli(response="DONE: built it, tests pass")
    drive_board_from_cli(fake, board="default", author="conductor", judge=_judge("done"))

    with kb.connect() as c:
        rows = kb.list_tasks(c, include_archived=False)
        by_title = {t.title: t.status for t in rows}
    assert by_title.get("Card A") == "done"
    assert by_title.get("Card B") == "done"
    assert fake.agent.calls >= 2  # at least one build turn per card


def test_conductor_entrypoint_blocks_when_agent_cannot_finish(kanban_home, monkeypatch):
    with kb.connect() as c:
        kb.create_task(c, title="Hard card", body="unfinishable", assignee="builder")

    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setenv("HERMES_PROFILE", "conductor")
    monkeypatch.setenv("HERMES_KANBAN_CONDUCTOR", "1")

    # Agent never emits DONE:; the aux judge is unreachable here (no model), so
    # the loop coerces to continue and blocks the card on budget rather than
    # falsely completing it.
    fake = _FakeCli(response="still working, no signal yet")
    drive_board_from_cli(fake, board="default", author="conductor", judge=_judge("continue"), max_turns_per_card=2, max_total_turns=2)

    with kb.connect() as c:
        rows = kb.list_tasks(c, include_archived=False)
        status = {t.title: t.status for t in rows}.get("Hard card")
    assert status == "blocked"


def test_conductor_on_empty_board_is_a_clean_noop(kanban_home):
    fake = _FakeCli()
    res = drive_board_from_cli(fake, board="default", author="conductor")
    assert fake.agent.calls == 0  # no cards → agent never invoked
    assert res["completed"] == [] and res["blocked"] == []
