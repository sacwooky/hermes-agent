"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).

They also cover the 2026-06-22 restart-proof dispatcher-stall fix: the
background watcher tasks must be retained against GC and the dispatcher must be
supervised so it cannot silently die after its "embedded in gateway" log.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from gateway.kanban_watchers import GatewayKanbanWatchersMixin

KANBAN_METHODS = [
    "_kanban_notifier_watcher",
    "_kanban_dispatcher_watcher",
    "_kanban_advance",
    "_kanban_unsub",
    "_kanban_rewind",
    "_deliver_kanban_artifacts",
]


def test_mixin_defines_kanban_methods():
    for m in KANBAN_METHODS:
        assert hasattr(GatewayKanbanWatchersMixin, m), f"mixin missing {m}"


def test_gateway_runner_inherits_mixin():
    # Import here so a heavy gateway import only happens if the first test passed.
    from gateway.run import GatewayRunner

    assert issubclass(GatewayRunner, GatewayKanbanWatchersMixin)
    # Each kanban method resolves to the mixin's implementation via the MRO.
    for m in KANBAN_METHODS:
        owner = next(c for c in GatewayRunner.__mro__ if m in c.__dict__)
        assert owner is GatewayKanbanWatchersMixin, (
            f"{m} resolved to {owner.__name__}, expected the mixin"
        )


def test_watcher_loops_are_coroutines():
    # The two long-running watchers are async loops.
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_notifier_watcher)
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._kanban_dispatcher_watcher)


# --- 2026-06-22 restart-proof dispatcher-stall fix -------------------------
#
# Root cause: gateway/run.py spawned the dispatcher with a bare
# `asyncio.create_task(...)` whose handle was discarded. The event loop keeps
# only a *weak* reference to a task, so it was garbage-collected at the first
# await — the dispatcher logged "embedded in gateway" once and then went
# silent forever. Fix = retain the handle in `_background_tasks` AND wrap the
# loop in a supervisor that respawns it if it ever exits while running.


class _FakeHost(GatewayKanbanWatchersMixin):
    """Minimal host exposing only the state the helpers under test touch."""

    def __init__(self) -> None:
        self._background_tasks: set = set()
        self._running = True


def test_track_background_task_retains_and_discards():
    host = _FakeHost()

    async def _scenario():
        async def _noop():
            return None

        task = asyncio.create_task(_noop())
        ret = host._track_background_task(task)
        # Returns the same task and registers it for retention.
        assert ret is task
        assert task in host._background_tasks
        # Once the task finishes, the done-callback drops the strong ref so
        # the retention set doesn't leak.
        await task
        await asyncio.sleep(0)  # let the done-callback run
        assert task not in host._background_tasks

    asyncio.run(_scenario())


def test_track_background_task_creates_set_if_missing():
    host = GatewayKanbanWatchersMixin.__new__(GatewayKanbanWatchersMixin)

    async def _scenario():
        async def _noop():
            return None

        task = asyncio.create_task(_noop())
        host._track_background_task(task)
        assert hasattr(host, "_background_tasks")
        assert task in host._background_tasks
        await task

    asyncio.run(_scenario())


def test_supervisor_restarts_silently_exiting_watcher():
    """A watcher that *returns* while the gateway is running is a silent
    stall; the supervisor must respawn it."""
    host = _FakeHost()
    calls = {"n": 0}

    async def _scenario():
        async def flaky_watcher():
            calls["n"] += 1
            if calls["n"] >= 3:
                # Stop the gateway so the supervisor exits and the test ends.
                host._running = False
            return None  # silent exit each time

        # restart_delay=0 so the test doesn't actually wait.
        await host._supervised_watcher(
            flaky_watcher, name="test dispatcher", restart_delay=0.0
        )

    asyncio.run(asyncio.wait_for(_scenario(), timeout=5))
    # Respawned until _running flipped False: at least the 3 observed calls.
    assert calls["n"] >= 3


def test_supervisor_restarts_on_exception():
    """A watcher that raises (not CancelledError) is respawned."""
    host = _FakeHost()
    calls = {"n": 0}

    async def _scenario():
        async def boom_watcher():
            calls["n"] += 1
            if calls["n"] >= 2:
                host._running = False
                return None
            raise RuntimeError("watcher crashed")

        await host._supervised_watcher(
            boom_watcher, name="test dispatcher", restart_delay=0.0
        )

    asyncio.run(asyncio.wait_for(_scenario(), timeout=5))
    assert calls["n"] >= 2


def test_supervisor_does_not_restart_when_not_running():
    """If the gateway is already shutting down, the supervisor exits without
    running the watcher (and without restarting)."""
    host = _FakeHost()
    host._running = False
    calls = {"n": 0}

    async def _scenario():
        async def watcher():
            calls["n"] += 1

        await host._supervised_watcher(
            watcher, name="test dispatcher", restart_delay=0.0
        )

    asyncio.run(asyncio.wait_for(_scenario(), timeout=5))
    assert calls["n"] == 0


def test_supervisor_propagates_cancellation_without_restart():
    """Gateway stop() cancels _background_tasks; the supervisor must let the
    CancelledError propagate and must NOT respawn."""
    host = _FakeHost()
    calls = {"n": 0}
    started = asyncio.Event()

    async def _scenario():
        async def long_watcher():
            calls["n"] += 1
            started.set()
            await asyncio.sleep(100)  # block until cancelled

        sup = asyncio.create_task(
            host._supervised_watcher(
                long_watcher, name="test dispatcher", restart_delay=0.0
            )
        )
        await started.wait()
        sup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await sup
        # Watcher ran exactly once and was not respawned after cancellation.
        assert calls["n"] == 1

    asyncio.run(asyncio.wait_for(_scenario(), timeout=5))


def test_supervised_watcher_is_coroutine():
    assert inspect.iscoroutinefunction(GatewayKanbanWatchersMixin._supervised_watcher)


def test_dispatcher_loop_emits_heartbeat():
    """The dispatcher loop body contains a per-tick heartbeat log so a wedged
    dispatcher is observable (regression guard for the silent-stall bug)."""
    src = inspect.getsource(GatewayKanbanWatchersMixin._kanban_dispatcher_watcher)
    assert "heartbeat" in src, "dispatcher loop lost its heartbeat log"


def test_singleton_dispatcher_lock_is_exclusive(tmp_path):
    """Only one holder of the dispatcher lock at a time — the backstop that
    stops concurrent dispatchers double reclaiming and corrupting shared
    kanban SQLite index pages under wal_autocheckpoint=0. (Upstream port.)"""
    from gateway.kanban_watchers import _acquire_singleton_lock, _release_singleton_lock

    lock = tmp_path / "kanban" / ".dispatcher.lock"

    h1, st1 = _acquire_singleton_lock(lock)
    assert st1 == "held" and h1 is not None

    # A second acquire while the first is held must be refused, not granted.
    h2, st2 = _acquire_singleton_lock(lock)
    assert st2 == "contended" and h2 is None

    # Releasing the first lets a fresh acquire succeed (lock is reusable).
    _release_singleton_lock(h1)
    h3, st3 = _acquire_singleton_lock(lock)
    assert st3 == "held" and h3 is not None
    _release_singleton_lock(h3)
