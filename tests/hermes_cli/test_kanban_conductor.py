"""Slice 1 — continuous-conductor routing + one-conductor-per-board guard.

Covers the additive, off-by-default conductor path: the pure config-precedence
helpers (``conductor_enabled_for`` / ``conductor_profile_for``) and
``ensure_conductor`` (spawn-once, no-double-spawn while alive, respawn when
dead). The dispatcher's existing ``dispatch_once`` path is untouched when a
board is not opted in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


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


@pytest.fixture
def conn(kanban_home):
    with kb.connect() as c:
        yield c


# --------------------------------------------------------------------------
# Config precedence (pure functions)
# --------------------------------------------------------------------------

def test_conductor_disabled_by_default():
    assert kb.conductor_enabled_for({}, "any") is False
    assert kb.conductor_enabled_for(None, "any") is False
    assert kb.conductor_enabled_for({"conductor": {}}, "any") is False


def test_conductor_global_default_enables_all_boards():
    cfg = {"conductor": {"default_enabled": True}}
    assert kb.conductor_enabled_for(cfg, "board-a") is True
    assert kb.conductor_enabled_for(cfg, "board-b") is True


def test_per_board_override_beats_global():
    # Global on, one board opts OUT.
    cfg = {"conductor": {"default_enabled": True},
           "boards": {"legacy": {"conductor": {"enabled": False}}}}
    assert kb.conductor_enabled_for(cfg, "legacy") is False
    assert kb.conductor_enabled_for(cfg, "other") is True

    # Global off, one board opts IN.
    cfg2 = {"boards": {"pilot": {"conductor": {"enabled": True}}}}
    assert kb.conductor_enabled_for(cfg2, "pilot") is True
    assert kb.conductor_enabled_for(cfg2, "other") is False


def test_conductor_profile_precedence():
    # per-board profile wins
    cfg = {"conductor": {"profile": "glob"},
           "orchestrator_profile": "orch",
           "boards": {"b": {"conductor": {"profile": "boardp"}}}}
    assert kb.conductor_profile_for(cfg, "b", fallback="fb") == "boardp"
    # global profile next
    assert kb.conductor_profile_for(cfg, "other", fallback="fb") == "glob"
    # orchestrator_profile next
    cfg2 = {"orchestrator_profile": "orch"}
    assert kb.conductor_profile_for(cfg2, "x", fallback="fb") == "orch"
    # fallback last
    assert kb.conductor_profile_for({}, "x", fallback="fb") == "fb"


# --------------------------------------------------------------------------
# ensure_conductor — spawn-once / no-double-spawn / respawn
# --------------------------------------------------------------------------

def _fake_spawn_factory(pid):
    calls = []

    def _spawn(board, workspace, *, profile):
        calls.append((board, workspace, profile))
        return pid

    return _spawn, calls


def test_ensure_conductor_spawns_when_none_alive(conn, monkeypatch):
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    spawn, calls = _fake_spawn_factory(4242)

    res = kb.ensure_conductor(conn, board="default", profile="orch", spawn_fn=spawn)

    assert len(calls) == 1, "should spawn exactly one conductor"
    assert calls[0][2] == "orch"
    assert res.spawned == [(kb.CONDUCTOR_PID_MARKER, "orch", calls[0][1])]
    db_path = kb.kanban_db_path(board="default")
    assert kb._read_conductor_pid(db_path) == 4242


def test_ensure_conductor_no_double_spawn_while_alive(conn, monkeypatch):
    # First tick spawns and records pid 4242.
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    spawn, calls = _fake_spawn_factory(4242)
    kb.ensure_conductor(conn, board="default", profile="orch", spawn_fn=spawn)
    assert len(calls) == 1

    # Now that conductor is "alive" — a second tick must NOT spawn again.
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: pid == 4242)
    res2 = kb.ensure_conductor(conn, board="default", profile="orch", spawn_fn=spawn)
    assert len(calls) == 1, "must not spawn a second conductor while the first is alive"
    assert res2.spawned == []


def test_ensure_conductor_respawns_when_dead(conn, monkeypatch):
    # Seed a stale pid on disk.
    db_path = kb.kanban_db_path(board="default")
    kb._write_conductor_pid(db_path, 9999)
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)  # 9999 is dead
    spawn, calls = _fake_spawn_factory(5555)

    res = kb.ensure_conductor(conn, board="default", profile="orch", spawn_fn=spawn)

    assert len(calls) == 1, "a dead conductor pid must be respawned"
    assert res.spawned == [(kb.CONDUCTOR_PID_MARKER, "orch", calls[0][1])]
    assert kb._read_conductor_pid(db_path) == 5555


def test_conductor_pid_roundtrip(tmp_path):
    db_path = tmp_path / "kanban.db"
    assert kb._read_conductor_pid(db_path) is None
    kb._write_conductor_pid(db_path, 1234)
    assert kb._read_conductor_pid(db_path) == 1234
    kb._write_conductor_pid(db_path, None)  # clear
    assert kb._read_conductor_pid(db_path) is None
