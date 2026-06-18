"""Global autonomy pause / kill-switch (WI-8, ADD-ON C v2 Phase 2 / WI-C8).

A simple, fail-safe, sentinel-file pause for the dispatcher. When tripped, the
dispatcher does its housekeeping (reclaim crashed/stale workers, run autonomy
sweeps) but **spawns no new work** — in-flight tasks finish, the queue holds, and
the board auto-resumes the moment the sentinel is cleared.

Two trip sources, distinguished so they don't fight:
- **manual** — an operator (or a higher-level control) trips it; only an operator
  clears it. Survives breaker recovery.
- **auto** — :func:`maybe_autopause_on_outage` trips it when many provider
  credentials are simultaneously OPEN (a fleet-wide outage), and **auto-clears**
  it when they recover. Never overrides a manual pause.

Default state is *not paused* (no sentinel). Everything here is best-effort and
fail-safe: a read error is treated as "not paused" so a corrupt sentinel can never
wedge the dispatcher shut.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

_log = logging.getLogger(__name__)

#: Global pause sentinel. A per-board sentinel may sit beside it as
#: ``autonomy.paused.<board>``.
_STATE_DIR = os.path.expanduser("~/.hermes/state")
GLOBAL_PAUSE_SENTINEL = os.path.join(_STATE_DIR, "autonomy.paused")

SOURCE_MANUAL = "manual"
SOURCE_AUTO = "auto"


def _sentinel_path(board: Optional[str]) -> str:
    if board:
        return os.path.join(_STATE_DIR, f"autonomy.paused.{board}")
    return GLOBAL_PAUSE_SENTINEL


def _read_sentinel(path: str) -> Optional[dict]:
    try:
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            txt = fh.read().strip()
        if not txt:
            return {"source": SOURCE_MANUAL, "reason": ""}
        try:
            return json.loads(txt)
        except Exception:
            # Plain-text reason (operator `echo "..." > sentinel`).
            return {"source": SOURCE_MANUAL, "reason": txt}
    except Exception:
        # Fail-safe: unreadable sentinel must not wedge the dispatcher.
        _log.debug("autonomy pause: sentinel read failed at %s", path, exc_info=True)
        return None


def is_paused(board: Optional[str] = None) -> Tuple[bool, str]:
    """Return ``(paused, reason)``. Checks the per-board sentinel then the global one."""
    for path in ([_sentinel_path(board)] if board else []) + [GLOBAL_PAUSE_SENTINEL]:
        rec = _read_sentinel(path)
        if rec is not None:
            return True, str(rec.get("reason") or rec.get("source") or "paused")
    return False, ""


def trip_pause(reason: str, *, board: Optional[str] = None, source: str = SOURCE_MANUAL) -> None:
    """Write the pause sentinel. Best-effort."""
    try:
        Path(_STATE_DIR).mkdir(parents=True, exist_ok=True)
        path = _sentinel_path(board)
        payload = {"source": source, "reason": reason, "tripped_at": int(time.time())}
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True))
        _log.warning("autonomy pause TRIPPED (source=%s board=%s): %s", source, board or "*", reason)
    except Exception:
        _log.debug("autonomy pause: trip failed", exc_info=True)


def clear_pause(*, board: Optional[str] = None, only_source: Optional[str] = None) -> bool:
    """Remove the pause sentinel. If ``only_source`` is set, clear only when the
    sentinel's source matches (so auto-resume never clears a manual pause).

    :returns: True if a sentinel was removed.
    """
    try:
        path = _sentinel_path(board)
        rec = _read_sentinel(path)
        if rec is None:
            return False
        if only_source is not None and rec.get("source") != only_source:
            return False
        os.remove(path)
        _log.info("autonomy pause CLEARED (board=%s)", board or "*")
        return True
    except FileNotFoundError:
        return False
    except Exception:
        _log.debug("autonomy pause: clear failed", exc_info=True)
        return False


def count_open_breakers(path: Optional[str] = None) -> int:
    """Coarse count of provider credentials whose breaker is currently OPEN.

    Reads the provider-health state file directly (fail-safe: 0 on any error).
    Used as the fleet-outage signal for :func:`maybe_autopause_on_outage`.
    """
    try:
        from hermes_cli import provider_health as ph
        state_path = path or ph._DEFAULT_STATE_PATH
        if not os.path.exists(state_path):
            return 0
        with open(state_path, encoding="utf-8") as fh:
            data = json.load(fh)
        records = data.get("records") or {}
        now_ms = int(time.monotonic() * 1000)  # only relative ordering matters
        open_count = 0
        for key, rec in records.items():
            try:
                cls, _, ident = key.partition("::")
                eff = ph.get_state(ident, failure_class=cls or ph.AUTH, path=state_path, now_ms=now_ms)
                if eff == ph.OPEN:
                    open_count += 1
            except Exception:
                if (rec or {}).get("state") == "OPEN":
                    open_count += 1
        return open_count
    except Exception:
        _log.debug("autonomy pause: open-breaker count failed", exc_info=True)
        return 0


def maybe_autopause_on_outage(
    *,
    threshold: int,
    board: Optional[str] = None,
    state_path: Optional[str] = None,
) -> Tuple[bool, str]:
    """Trip an AUTO pause when ``>= threshold`` credentials are OPEN; auto-clear the
    AUTO pause when they recover. Never trips or clears a MANUAL pause.

    :returns: ``(paused_now, reason)`` reflecting the post-call state.
    """
    if threshold <= 0:
        return is_paused(board)

    open_now = count_open_breakers(state_path)
    path = _sentinel_path(board) if board else GLOBAL_PAUSE_SENTINEL
    rec = _read_sentinel(path)
    currently_auto = bool(rec and rec.get("source") == SOURCE_AUTO)

    if open_now >= threshold:
        if rec is None:  # not paused → trip auto
            trip_pause(
                f"outage auto-pause: {open_now} provider credentials OPEN (>= {threshold})",
                board=board, source=SOURCE_AUTO,
            )
        return True, f"outage: {open_now} OPEN"
    # Recovered: clear ONLY if we were the ones who auto-tripped.
    if currently_auto:
        clear_pause(board=board, only_source=SOURCE_AUTO)
        _log.info("autonomy pause AUTO-RESUMED (open breakers %d < %d)", open_now, threshold)
    return is_paused(board)
