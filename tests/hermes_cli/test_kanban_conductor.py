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
    kb.create_task(conn, title='w', assignee='builder')  # workable card so the conductor spawns
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
    kb.create_task(conn, title='w', assignee='builder')  # workable card so the conductor spawns
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
        captured["cwd"] = kw.get("cwd")
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
    assert captured["cwd"] is not None  # conductor cwd pinned to the workspace (fd/realpath)
    assert 7 not in (captured["pass_fds"] or ())  # no lock fd here (inherit_fd was None)


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
    assert 7 in captured["pass_fds"]  # the lock fd is inherited by the child (ws dir fd may also be present)


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
    kb.create_task(conn, title='w', assignee='builder')  # workable card so the conductor spawns
    wd = tmp_path / "repo"
    wd.mkdir()
    monkeypatch.setattr(kb, "read_board_metadata", lambda slug: {"default_workdir": str(wd)})
    spawn, calls = _spy_spawn()
    kb.ensure_conductor(conn, board="default", profile="orch", spawn_fn=spawn)
    assert calls and calls[0]["workspace"] == str(wd), "conductor must run in the board default_workdir"


# --------------------------------------------------------------------------
# Security (Robin BLOCK): validate board default_workdir; never mkdir arbitrary
# --------------------------------------------------------------------------

def test_is_safe_conductor_workdir_accepts_real_dir(tmp_path):
    d = tmp_path / "repo"
    d.mkdir()
    assert kb._is_safe_conductor_workdir(str(d)) is True


def test_is_safe_conductor_workdir_rejects_dangerous_values(tmp_path):
    assert kb._is_safe_conductor_workdir("/etc") is False           # sensitive
    assert kb._is_safe_conductor_workdir("/") is False              # root
    assert kb._is_safe_conductor_workdir("relative/path") is False  # not absolute
    assert kb._is_safe_conductor_workdir(str(tmp_path / "nope")) is False  # nonexistent
    assert kb._is_safe_conductor_workdir("/tmp/x; rm -rf /") is False      # metachars
    assert kb._is_safe_conductor_workdir("/usr/lib") is False       # under sensitive root
    # symlink to a real dir is refused
    real = tmp_path / "real"; real.mkdir()
    link = tmp_path / "link"; link.symlink_to(real)
    assert kb._is_safe_conductor_workdir(str(link)) is False


def test_conductor_workspace_rejects_unsafe_default_and_never_mkdirs_it(kanban_home, monkeypatch):
    monkeypatch.setattr(kb, "read_board_metadata", lambda slug: {"default_workdir": "/etc/hermes-attack"})
    ws = kb._conductor_workspace("default")
    assert ws == str(kb.workspaces_root(board="default"))  # fell back, did not use /etc
    assert not os.path.exists("/etc/hermes-attack")         # never created an arbitrary dir


def test_conductor_workspace_uses_safe_default_workdir(kanban_home, tmp_path, monkeypatch):
    d = tmp_path / "project"; d.mkdir()
    monkeypatch.setattr(kb, "read_board_metadata", lambda slug: {"default_workdir": str(d)})
    assert kb._conductor_workspace("default") == str(d)


def test_is_safe_conductor_workdir_rejects_traversal_and_symlinked_parent(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    # `..` traversal component, even though the resolved dir exists
    assert kb._is_safe_conductor_workdir(str(tmp_path / "sub" / ".." / "sub")) is False
    # symlinked PARENT component (the path itself is not a symlink)
    realp = tmp_path / "realparent"
    realp.mkdir()
    (realp / "proj").mkdir()
    linkp = tmp_path / "linkparent"
    linkp.symlink_to(realp)
    assert kb._is_safe_conductor_workdir(str(linkp / "proj")) is False
    # a fully-canonical real dir still passes
    assert kb._is_safe_conductor_workdir(str(realp / "proj")) is True


def test_ensure_conductor_no_spawn_when_board_has_no_workable_cards(conn):
    """The workable-work gate: an opted-in but empty board must NOT spawn a
    conductor (avoids the empty-board spawn/idle/respawn storm)."""
    spawn, calls = _spy_spawn()
    res = kb.ensure_conductor(conn, board="default", profile="orch", spawn_fn=spawn)
    assert calls == [], "must not spawn on a board with no ready+assigned cards"
    assert res.spawned == []


# --------------------------------------------------------------------------
# Follow-up: shared symlink-hardened spawn-log open (worker + conductor)
# --------------------------------------------------------------------------

def test_open_spawn_log_safely_refuses_symlink(tmp_path):
    victim = tmp_path / "victim"
    victim.write_text("keep", encoding="utf-8")
    logp = tmp_path / "spawn.log"
    logp.symlink_to(victim)
    f = kb._open_spawn_log_safely(logp, 1_000_000, 1)
    try:
        f.write(b"conductor output that must NOT reach the victim\n")
        f.flush()
    finally:
        f.close()
    assert victim.read_text(encoding="utf-8") == "keep"  # target untouched (wrote to devnull)
    assert logp.is_symlink()  # symlink not replaced/rotated


def test_open_spawn_log_safely_opens_regular_file(tmp_path):
    logp = tmp_path / "spawn.log"
    f = kb._open_spawn_log_safely(logp, 1_000_000, 1)
    try:
        f.write(b"hello\n")
        f.flush()
    finally:
        f.close()
    assert logp.is_file() and not logp.is_symlink()
    assert b"hello" in logp.read_bytes()


def test_open_spawn_log_safely_refuses_hardlink(tmp_path):
    """O_NOFOLLOW catches symlinks but not HARD links — a hard-linked log path
    (nlink>1) must be refused so appends can't land in a victim's inode."""
    victim = tmp_path / "victim"
    victim.write_text("keep", encoding="utf-8")
    logp = tmp_path / "spawn.log"
    os.link(victim, logp)  # hard link, not a symlink
    f = kb._open_spawn_log_safely(logp, 1_000_000, 1)
    try:
        f.write(b"must not reach the victim inode\n")
        f.flush()
    finally:
        f.close()
    assert victim.read_text(encoding="utf-8") == "keep"  # refused → wrote to devnull


# --------------------------------------------------------------------------
# K4: the G3 epic-acceptance sweep on conductor boards
# --------------------------------------------------------------------------

AGENT_EA = {"epic_acceptance": {"enabled": True, "approver": "agent",
                                "reviewer_assignee": "product", "fix_assignee": "builder"}}


def _worker_spy():
    calls = []

    def _spawn(task, workspace, board=None):
        calls.append({"id": task.id, "assignee": task.assignee, "board": board})
        return 4343

    return _spawn, calls


def _finished_epic(conn):
    root = kb.create_task(conn, title="Epic: conductor board", triage=True)
    child_ids = kb.decompose_triage_task(
        conn, root, root_assignee="orchestrator", author="orchestrator",
        children=[{"title": "build it", "assignee": "builder", "parents": []}],
    )
    for cid in child_ids:
        kb.complete_task(conn, cid, result="done")
    kb.complete_task(conn, root, result="epic complete")
    return root


def _review_id(conn):
    row = conn.execute("SELECT task_id FROM task_metadata "
                       "WHERE work_item_type = 'acceptance_review'").fetchone()
    return row["task_id"] if row else None


def test_conductor_board_sweep_opens_the_gate_and_spawns_the_review_as_its_reviewer(
        conn, all_assignees_spawnable):
    _finished_epic(conn)
    spawn, calls = _worker_spy()
    out = kb.conductor_board_sweeps(conn, board="default", autonomy_cfg=AGENT_EA, spawn_fn=spawn)
    assert len(out["autonomy"]["acceptances_created"]) == 1
    rid = _review_id(conn)
    assert rid, "agent mode opens one review"
    assert calls == [{"id": rid, "assignee": "product", "board": "default"}]
    run = kb.latest_run(conn, rid)
    assert run.profile == "product" and kb.get_task(conn, rid).status == "running"


def test_conductor_board_sweep_is_off_without_epic_acceptance(conn, all_assignees_spawnable):
    _finished_epic(conn)
    spawn, calls = _worker_spy()
    assert kb.conductor_board_sweeps(conn, board="default", autonomy_cfg={}, spawn_fn=spawn) == {}
    assert calls == [] and _review_id(conn) is None


def test_conductor_never_takes_an_acceptance_review(conn, all_assignees_spawnable):
    _finished_epic(conn)
    kb.conductor_board_sweeps(conn, board="default", autonomy_cfg=AGENT_EA,
                              spawn_fn=lambda task, ws, board=None: None)
    rid = _review_id(conn)
    assert kb.conductor_skips(conn, rid)
    # Put the review back to ready: it alone must not wake a conductor.
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready', claim_lock=NULL WHERE id = ?", (rid,))
    spawn, calls = _spy_spawn()
    assert kb.ensure_conductor(conn, board="default", profile="orch", spawn_fn=spawn).spawned == []
    assert calls == []


def test_skipped_cards_leave_other_ready_cards_to_the_conductor(conn, all_assignees_spawnable):
    other = kb.create_task(conn, title="normal card", assignee="builder")
    spawn, calls = _worker_spy()
    assert kb.dispatch_conductor_skipped(conn, board="default", spawn_fn=spawn).spawned == []
    assert calls == [] and kb.get_task(conn, other).status == "ready"
    assert not kb.conductor_skips(conn, other)


def test_skipped_card_spawn_honours_the_autonomy_pause(conn, all_assignees_spawnable, monkeypatch):
    _finished_epic(conn)
    kb.conductor_board_sweeps(conn, board="default", autonomy_cfg=AGENT_EA,
                              spawn_fn=lambda task, ws, board=None: None)
    rid = _review_id(conn)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready', claim_lock=NULL WHERE id = ?", (rid,))
    from hermes_cli.review_loop import pause
    monkeypatch.setattr(pause, "is_paused", lambda board=None: (True, "test pause"))
    spawn, calls = _worker_spy()
    res = kb.dispatch_conductor_skipped(conn, board="default", spawn_fn=spawn)
    assert res.paused and calls == [] and kb.get_task(conn, rid).status == "ready"


def test_skipped_card_with_a_non_profile_assignee_is_not_spawned(conn, monkeypatch):
    _finished_epic(conn)
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: False)
    spawn, calls = _worker_spy()
    out = kb.conductor_board_sweeps(conn, board="default", autonomy_cfg=AGENT_EA, spawn_fn=spawn)
    assert calls == [] and out["spawned_reviews"] == []
