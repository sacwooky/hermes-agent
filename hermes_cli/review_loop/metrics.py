"""L0 catch-rate metric (WI-C3, ADD-ON C v2).

Best-effort, DEFAULT-OFF, fail-safe — a direct sibling of
``kanban_autonomy._emit_epic_run_metric``. When ``HERMES_LEARNING_VAULT_ROOT`` is set,
append one JSONL row per L0 decision to ``metrics/autonomy/l0_gate.jsonl`` (the same
``metrics/autonomy/`` dir the conductor-vault instrumentation report already globs).

``settled_at_l0=True`` means the artifact FAILed L0 and was kept OUT of model review
(the scaling win); ``False`` means L0 passed and the artifact proceeds to review. The
per-project catch-rate is then ``count(settled_at_l0=True) / count(all rows)``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

#: Stream file (under HERMES_LEARNING_VAULT_ROOT/metrics/autonomy/).
L0_METRIC_FILE = "l0_gate.jsonl"


def emit_l0_catchrate(
    task_id: str,
    *,
    settled_at_l0: bool,
    passed: bool,
    board: Optional[str] = None,
    failed_required: Optional[list] = None,
    duration_s: Optional[float] = None,
) -> None:
    """Append one L0 catch-rate row. Never raises; no-op when the env is unset."""
    try:
        root = os.environ.get("HERMES_LEARNING_VAULT_ROOT")
        if not root:
            return
        d = Path(root) / "metrics" / "autonomy"
        d.mkdir(parents=True, exist_ok=True)
        row = {
            "task_id": task_id,
            "board": board or "",
            "settled_at_l0": bool(settled_at_l0),
            "passed": bool(passed),
            "failed_required": list(failed_required or []),
            "recorded_at": int(time.time()),
            "source": "l0_gate",
        }
        if isinstance(duration_s, (int, float)):
            row["duration_s"] = round(float(duration_s), 3)
        with open(d / L0_METRIC_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    except Exception:
        _log.debug("l0 catch-rate metric emit skipped for %s", task_id, exc_info=True)
