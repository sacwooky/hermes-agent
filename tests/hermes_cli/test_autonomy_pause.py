"""Tests for ADD-ON C v2 Phase 2 — WI-8 global autonomy pause.

Covers the pause primitive (trip/clear/is_paused, manual vs auto source, outage
auto-pause/auto-resume) and the dispatch_once integration (paused → no new spawns,
housekeeping still runs, auto-resume on clear). LLM-free.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.review_loop import pause as P


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Redirect the pause sentinels into a tmp state dir."""
    d = tmp_path / "state"
    d.mkdir()
    monkeypatch.setattr(P, "_STATE_DIR", str(d))
    monkeypatch.setattr(P, "GLOBAL_PAUSE_SENTINEL", str(d / "autonomy.paused"))
    return d


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Primitive
# ---------------------------------------------------------------------------


def test_default_not_paused(state_dir):
    assert P.is_paused() == (False, "")


def test_trip_and_clear_manual(state_dir):
    P.trip_pause("operator stop", source=P.SOURCE_MANUAL)
    paused, reason = P.is_paused()
    assert paused is True and "operator stop" in reason
    assert P.clear_pause() is True
    assert P.is_paused()[0] is False


def test_plaintext_sentinel_is_paused(state_dir):
    # Operator `echo "stop" > sentinel` (not JSON) still reads as paused.
    Path(P.GLOBAL_PAUSE_SENTINEL).write_text("stop now")
    paused, reason = P.is_paused()
    assert paused is True and "stop now" in reason


def test_unreadable_sentinel_fails_safe(state_dir):
    # A sentinel path that can't be read as a file (here: it's a directory)
    # must NOT wedge the dispatcher shut — _read_sentinel swallows and returns None.
    Path(P.GLOBAL_PAUSE_SENTINEL).mkdir()
    paused, _ = P.is_paused()
    assert paused is False


def test_auto_resume_does_not_clear_manual(state_dir):
    P.trip_pause("operator stop", source=P.SOURCE_MANUAL)
    # auto-resume path must not clear a manual pause
    assert P.clear_pause(only_source=P.SOURCE_AUTO) is False
    assert P.is_paused()[0] is True


def test_outage_autopause_trips_and_resumes(state_dir, monkeypatch):
    calls = {"open": 5}
    monkeypatch.setattr(P, "count_open_breakers", lambda *_a, **_k: calls["open"])
    # 5 >= threshold 3 → auto-trip
    paused, _ = P.maybe_autopause_on_outage(threshold=3)
    assert paused is True
    rec = json.loads(Path(P.GLOBAL_PAUSE_SENTINEL).read_text())
    assert rec["source"] == P.SOURCE_AUTO
    # breakers recover → auto-resume clears it
    calls["open"] = 0
    paused, _ = P.maybe_autopause_on_outage(threshold=3)
    assert paused is False
    assert not Path(P.GLOBAL_PAUSE_SENTINEL).exists()


def test_outage_autopause_never_overrides_manual(state_dir, monkeypatch):
    P.trip_pause("operator stop", source=P.SOURCE_MANUAL)
    monkeypatch.setattr(P, "count_open_breakers", lambda *_a, **_k: 0)
    # recovery must NOT clear the manual pause
    P.maybe_autopause_on_outage(threshold=3)
    assert P.is_paused()[0] is True
    rec = json.loads(Path(P.GLOBAL_PAUSE_SENTINEL).read_text())
    assert rec["source"] == P.SOURCE_MANUAL


def test_threshold_zero_is_noop(state_dir, monkeypatch):
    monkeypatch.setattr(P, "count_open_breakers", lambda *_a, **_k: 99)
    paused, _ = P.maybe_autopause_on_outage(threshold=0)
    assert paused is False  # disabled


# ---------------------------------------------------------------------------
# dispatch_once integration
# ---------------------------------------------------------------------------


def _fake_spawn(*a, **k):
    return 4242


def test_dispatch_not_paused_spawns(state_dir, kanban_home, all_assignees_spawnable):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="builder")
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn)
        assert res.paused is False
        assert any(s[0] == tid for s in res.spawned)


def test_dispatch_paused_skips_spawn(state_dir, kanban_home, all_assignees_spawnable):
    P.trip_pause("operator stop")
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="builder")
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn)
        assert res.paused is True
        assert "operator stop" in res.pause_reason
        assert not res.spawned  # no new work spawned
        # task stays ready (held), not lost
        assert kb.get_task(conn, tid).status == "ready"


def test_dispatch_auto_resumes_after_clear(state_dir, kanban_home, all_assignees_spawnable):
    P.trip_pause("operator stop")
    with kb.connect() as conn:
        kb.create_task(conn, title="t", assignee="builder")
        assert kb.dispatch_once(conn, spawn_fn=_fake_spawn).paused is True
        P.clear_pause()
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn)
        assert res.paused is False
        assert res.spawned  # resumes spawning
