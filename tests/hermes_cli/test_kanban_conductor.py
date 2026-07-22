"""Slice 1 — continuous-conductor routing + one-conductor-per-board guard.

Covers the additive, off-by-default conductor path: the pure config-precedence
helpers, and ensure_conductor's LIFETIME-LOCK guard (spawn when the board's
.conductor.lock is free, no-op while a conductor holds it, respawn once
released, refuse a symlinked lock path). The dispatcher's existing dispatch_once
path is untouched when a board is not opted in.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb

fcntl = pytest.importorskip("fcntl")  # the guard is POSIX flock-based


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
    cfg = {"conductor": {"default_enabled": True},
           "boards": {"legacy": {"conductor": {"enabled": False}}}}
    assert kb.conductor_enabled_for(cfg, "legacy") is False
    assert kb.conductor_enabled_for(cfg, "other") is True

    cfg2 = {"boards": {"pilot": {"conductor": {"enabled": True}}}}
    assert kb.conductor_enabled_for(cfg2, "pilot") is True
    assert kb.conductor_enabled_for(cfg2, "other") is False


def test_conductor_profile_precedence():
    cfg = {"conductor": {"profile": "glob"},
           "orchestrator_profile": "orch",
           "boards": {"b": {"conductor": {"profile": "boardp"}}}}
    assert kb.conductor_profile_for(cfg, "b", fallback="fb") == "boardp"
    assert kb.conductor_profile_for(cfg, "other", fallback="fb") == "glob"
    cfg2 = {"orchestrator_profile": "orch"}
    assert kb.conductor_profile_for(cfg2, "x", fallback="fb") == "orch"
    assert kb.conductor_profile_for({}, "x", fallback="fb") == "fb"


# --------------------------------------------------------------------------
# ensure_conductor — lifetime-lock guard
# --------------------------------------------------------------------------

def _spy_spawn(pid=4242):
    calls = []

    def _spawn(board, workspace, *, profile, inherit_fd=None):
        # A real conductor would hold inherit_fd for life; the spy just records
        # the call and returns a pid (its "conductor" does not survive, so the
        # parent closing its fd copy releases the lock — correct for a spy).
        calls.append({"board": board, "workspace": workspace, "profile": profile,
                      "inherit_fd": inherit_fd})
        return pid

    return _spawn, calls


def test_ensure_conductor_spawns_when_lock_free(conn):
    spawn, calls = _spy_spawn()
    res = kb.ensure_conductor(conn, board="default", profile="orch", spawn_fn=spawn)
    assert len(calls) == 1, "should spawn when no conductor holds the lock"
    assert calls[0]["profile"] == "orch"
    assert calls[0]["inherit_fd"] is not None, "must pass the held lock fd to inherit"
    assert res.spawned == [(kb.CONDUCTOR_PID_MARKER, "orch", calls[0]["workspace"])]


def test_ensure_conductor_no_spawn_while_lock_held(conn):
    # Simulate a live conductor by holding the board's conductor lock ourselves.
    db_path = kb.kanban_db_path(board="default")
    lock_path = kb._conductor_lock_path(db_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    held = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        spawn, calls = _spy_spawn()
        res = kb.ensure_conductor(conn, board="default", profile="orch", spawn_fn=spawn)
        assert calls == [], "must NOT spawn while a conductor holds the lock"
        assert res.spawned == []
    finally:
        fcntl.flock(held, fcntl.LOCK_UN)
        os.close(held)


def test_ensure_conductor_respawns_after_lock_released(conn):
    db_path = kb.kanban_db_path(board="default")
    lock_path = kb._conductor_lock_path(db_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    held = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spawn, calls = _spy_spawn()
    assert kb.ensure_conductor(conn, board="default", profile="orch", spawn_fn=spawn).spawned == []
    fcntl.flock(held, fcntl.LOCK_UN)
    os.close(held)  # conductor "exited" → lock free
    res = kb.ensure_conductor(conn, board="default", profile="orch", spawn_fn=spawn)
    assert len(calls) == 1, "must respawn once the lock is free again"
    assert res.spawned


def test_ensure_conductor_refuses_symlinked_lock_path(conn, tmp_path):
    db_path = kb.kanban_db_path(board="default")
    lock_path = kb._conductor_lock_path(db_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    victim = tmp_path / "victim"
    victim.write_text("keep", encoding="utf-8")
    lock_path.symlink_to(victim)
    spawn, calls = _spy_spawn()
    res = kb.ensure_conductor(conn, board="default", profile="orch", spawn_fn=spawn)
    assert calls == [], "must refuse a symlinked lock path (O_NOFOLLOW)"
    assert res.spawned == []
    assert victim.read_text(encoding="utf-8") == "keep"


# --------------------------------------------------------------------------
# _spawn_conductor — no board management, log symlink-hardened
# --------------------------------------------------------------------------

def test_spawn_conductor_has_no_board_management(kanban_home, tmp_path, monkeypatch):
    """No kanban-worker skill, no HERMES_KANBAN_TASK → the single-card kanban
    lifecycle tools never attach and the agent cannot manage the board."""
    import subprocess

    captured = {}

    class _FakeProc:
        pid = 4321

    def _fake_popen(cmd, **kw):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kw.get("env") or {})
        captured["pass_fds"] = kw.get("pass_fds")
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    ws = tmp_path / "ws"
    ws.mkdir()

    pid = kb._spawn_conductor("default", str(ws), profile="default")

    assert pid == 4321
    assert "kanban-worker" not in captured["cmd"]
    assert "-Q" in captured["cmd"]
    assert captured["env"].get("HERMES_KANBAN_CONDUCTOR") == "1"
    assert "HERMES_KANBAN_TASK" not in captured["env"]
    assert captured["pass_fds"] == ()  # no inherit fd → nothing inherited


def test_spawn_conductor_passes_inherit_fd(kanban_home, tmp_path, monkeypatch):
    import subprocess

    captured = {}

    class _FakeProc:
        pid = 88

    monkeypatch.setattr(
        subprocess, "Popen",
        lambda cmd, **kw: (captured.update(pass_fds=kw.get("pass_fds")) or _FakeProc()),
    )
    ws = tmp_path / "ws"
    ws.mkdir()
    kb._spawn_conductor("default", str(ws), profile="default", inherit_fd=7)
    assert captured["pass_fds"] == (7,)  # the lock fd is inherited by the child


def test_spawn_conductor_log_refuses_symlink(kanban_home, tmp_path, monkeypatch):
    """A symlink at __conductor__.log must not redirect conductor stdout."""
    import subprocess

    class _FakeProc:
        pid = 777

    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: _FakeProc())
    victim = tmp_path / "log_victim.txt"
    victim.write_text("keep-me", encoding="utf-8")
    log_dir = kb.worker_logs_dir(board="default")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{kb.CONDUCTOR_PID_MARKER}.log"
    log_path.symlink_to(victim)
    ws = tmp_path / "ws"
    ws.mkdir()

    pid = kb._spawn_conductor("default", str(ws), profile="default")

    assert pid == 777
    assert log_path.is_symlink()
    assert victim.read_text(encoding="utf-8") == "keep-me"


def test_spawn_conductor_log_rotation_skips_symlink(kanban_home, tmp_path, monkeypatch):
    """Rotation must not run on a symlinked log path (stat/rename surface). With a
    1-byte rotation threshold the target far exceeds it, so rotation WOULD trigger
    if the path were not symlink-guarded."""
    import subprocess

    class _FakeProc:
        pid = 5

    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: _FakeProc())
    monkeypatch.setattr(kb, "worker_log_rotation_config", lambda: (1, 1))  # rotate at 1 byte
    victim = tmp_path / "big_victim.txt"
    victim.write_text("x" * 4096, encoding="utf-8")  # well over the threshold
    log_dir = kb.worker_logs_dir(board="default")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{kb.CONDUCTOR_PID_MARKER}.log"
    log_path.symlink_to(victim)
    ws = tmp_path / "ws"
    ws.mkdir()

    kb._spawn_conductor("default", str(ws), profile="default")

    assert log_path.is_symlink(), "symlinked log path must NOT be rotated (renamed) away"
    assert victim.read_text(encoding="utf-8") == "x" * 4096, "rotation must not touch the target"


# --------------------------------------------------------------------------
# Pilot finding: conductor must work in the board's default_workdir, not $HOME
# --------------------------------------------------------------------------

def test_conductor_workspace_prefers_board_default_workdir(kanban_home, tmp_path, monkeypatch):
    wd = tmp_path / "project"
    wd.mkdir()
    monkeypatch.setattr(kb, "read_board_metadata", lambda slug: {"default_workdir": str(wd)})
    assert kb._conductor_workspace("default") == str(wd)


def test_conductor_workspace_falls_back_and_creates_dir(kanban_home, monkeypatch):
    monkeypatch.setattr(kb, "read_board_metadata", lambda slug: {})
    ws = kb._conductor_workspace("default")
    assert ws == str(kb.workspaces_root(board="default"))
    assert os.path.isdir(ws)  # created so cwd never falls back to $HOME


def test_ensure_conductor_spawns_in_board_workspace(conn, tmp_path, monkeypatch):
    wd = tmp_path / "repo"
    wd.mkdir()
    monkeypatch.setattr(kb, "read_board_metadata", lambda slug: {"default_workdir": str(wd)})
    spawn, calls = _spy_spawn()
    kb.ensure_conductor(conn, board="default", profile="orch", spawn_fn=spawn)
    assert calls and calls[0]["workspace"] == str(wd), "conductor must run in the board default_workdir"
