"""L0 attestation — the out-of-band evidence writer (WI-C3, ADD-ON C v2).

This is the ONLY L0 module that touches the kanban DB. It records the
:class:`~hermes_cli.review_loop.l0_gate.L0Result` as a single ``l0_attestation``
event stamped ``attested_by: "l0_harness"`` — provenance that makes explicit the
gate was run by the dispatcher/harness, **not** asserted by the completing worker.

Both PASS and FAIL are written (a failing build is evidence too). The writer wraps its
own transaction and never raises into the dispatch loop.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli.review_loop.l0_gate import L0Result

_log = logging.getLogger(__name__)

#: Event kind for the out-of-band L0 evidence record.
L0_ATTESTATION_EVENT = "l0_attestation"
#: Schema tag on the payload so downstream readers can version-gate.
L0_ATTESTATION_SCHEMA = "l0_attestation/v1"


def _result_payload(result: L0Result) -> dict:
    return {
        "schema": L0_ATTESTATION_SCHEMA,
        "attested_by": "l0_harness",  # provenance: harness, NOT the builder
        "passed": result.passed,
        "failed_required": result.failed_required,
        "failed_advisory": result.failed_advisory,
        "duration_s": result.duration_s,
        "workspace": result.workspace,
        "checks": [
            {
                "name": c.name,
                "type": c.type,
                "command": c.command,
                "exit_code": c.exit_code,
                "passed": c.passed,
                "required": c.required,
                "timed_out": c.timed_out,
                "duration_s": c.duration_s,
                "log_tail": c.truncated_log,
            }
            for c in result.checks
        ],
    }


def record_l0_attestation(
    conn: sqlite3.Connection,
    task_id: str,
    result: L0Result,
    *,
    run_id: Optional[int] = None,
    board: Optional[str] = None,
) -> bool:
    """Write the ``l0_attestation`` event for ``task_id``. Best-effort, never raises.

    :returns: True if the event was written, False if it was skipped on error.
    """
    try:
        payload = _result_payload(result)
        if board:
            payload["board"] = board
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, L0_ATTESTATION_EVENT, payload, run_id=run_id)
        return True
    except Exception:
        _log.debug("l0_attestation: emit skipped for %s", task_id, exc_info=True)
        return False
