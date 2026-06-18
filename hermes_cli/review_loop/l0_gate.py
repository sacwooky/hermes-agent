"""L0 deterministic gate — the pure executor (WI-C3, ADD-ON C v2).

Runs a configured list of deterministic checks (tests / lint / type-check / build /
SAST) in the task's workspace as subprocesses, with a per-check wall-clock timeout and
tail-truncated output. **No LLM, no DB, no network of its own.** This module is
deliberately import-pure (it imports only the stdlib) so it can be unit-tested by running
real shell builtins (``true`` / ``false`` / ``sleep``) against a temp dir with zero setup.

A *required* check that fails (or times out) fails the gate. An *advisory* check that
fails is recorded but does not fail the gate. The gate's truth is the returned
:class:`L0Result`; the harness — not the builder — records it out-of-band
(see :mod:`hermes_cli.review_loop.attestation`).
"""

from __future__ import annotations

import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

#: Default per-check wall-clock cap (seconds).
DEFAULT_TIMEOUT_S = 600
#: Default tail size for captured combined stdout+stderr (bytes).
DEFAULT_LOG_TAIL_BYTES = 8192


@dataclass(frozen=True)
class L0CheckResult:
    """Outcome of a single deterministic check."""

    name: str
    type: str
    command: str
    exit_code: Optional[int]  # None == timed out / never completed
    passed: bool
    required: bool
    timed_out: bool
    duration_s: float
    truncated_log: str


@dataclass(frozen=True)
class L0Result:
    """Aggregate outcome of an L0 gate run over one workspace."""

    passed: bool  # all REQUIRED checks passed (advisory FAIL never fails the gate)
    checks: list[L0CheckResult]
    workspace: str
    started_at: int
    duration_s: float

    @property
    def failed_required(self) -> list[str]:
        """Names of required checks that did not pass."""
        return [c.name for c in self.checks if c.required and not c.passed]

    @property
    def failed_advisory(self) -> list[str]:
        """Names of advisory checks that did not pass (informational only)."""
        return [c.name for c in self.checks if not c.required and not c.passed]


def _tail_truncate(text: str, limit: int) -> str:
    """Return at most ``limit`` bytes from the END of ``text`` (UTF-8 safe).

    The tail is kept because the actionable error (traceback, failing assertion,
    SAST finding) is almost always at the end of a check's output.
    """
    if limit <= 0 or not text:
        return ""
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= limit:
        return text
    tail = raw[-limit:]
    truncated = tail.decode("utf-8", errors="replace")
    return "…[truncated]…\n" + truncated


def _run_one_check(
    check: dict,
    workspace: Path,
    *,
    timeout_s: int,
    log_tail_bytes: int,
    env: Optional[dict] = None,
) -> L0CheckResult:
    name = str(check.get("name") or check.get("type") or "check")
    ctype = str(check.get("type") or "")
    command = str(check.get("command") or "")
    required = bool(check.get("required", True))

    if not command.strip():
        return L0CheckResult(
            name=name, type=ctype, command=command, exit_code=None,
            passed=False, required=required, timed_out=False, duration_s=0.0,
            truncated_log="(no command configured)",
        )

    t0 = time.monotonic()
    try:
        proc = subprocess.run(  # noqa: S603 — commands come from operator config, not workers
            shlex.split(command),
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_s)),
            env=env,
        )
        duration = time.monotonic() - t0
        combined = (proc.stdout or "") + (proc.stderr or "")
        return L0CheckResult(
            name=name, type=ctype, command=command, exit_code=proc.returncode,
            passed=(proc.returncode == 0), required=required, timed_out=False,
            duration_s=round(duration, 3),
            truncated_log=_tail_truncate(combined, log_tail_bytes),
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - t0
        partial = ""
        for stream in (getattr(exc, "stdout", None), getattr(exc, "stderr", None)):
            if stream:
                partial += stream.decode("utf-8", "replace") if isinstance(stream, bytes) else stream
        return L0CheckResult(
            name=name, type=ctype, command=command, exit_code=None,
            passed=False, required=required, timed_out=True,
            duration_s=round(duration, 3),
            truncated_log=_tail_truncate(
                (partial + f"\n(timed out after {timeout_s}s)").strip(), log_tail_bytes
            ),
        )
    except (OSError, ValueError) as exc:
        duration = time.monotonic() - t0
        return L0CheckResult(
            name=name, type=ctype, command=command, exit_code=None,
            passed=False, required=required, timed_out=False,
            duration_s=round(duration, 3),
            truncated_log=_tail_truncate(f"(failed to launch: {exc})", log_tail_bytes),
        )


def run_l0_gate(
    workspace,
    checks_cfg: list,
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    log_tail_bytes: int = DEFAULT_LOG_TAIL_BYTES,
    env: Optional[dict] = None,
) -> L0Result:
    """Run every configured check in ``workspace`` and aggregate the outcome.

    :param workspace: directory to run the checks in (the task's resolved workspace).
    :param checks_cfg: list of ``{name, command, type, required}`` dicts.
    :param timeout_s: per-check wall-clock cap.
    :param log_tail_bytes: tail size for each check's captured output.
    :param env: optional environment for the subprocesses (defaults to inherited).
    :returns: :class:`L0Result`. ``passed`` is True iff all *required* checks passed.

    Never raises on a check failure — failures are captured in the result. (It can
    raise only on a programming error such as a non-iterable ``checks_cfg``.)
    """
    ws = Path(workspace)
    started = int(time.time())
    t0 = time.monotonic()
    results: list[L0CheckResult] = []
    for check in checks_cfg or []:
        results.append(
            _run_one_check(
                check, ws, timeout_s=timeout_s, log_tail_bytes=log_tail_bytes, env=env
            )
        )
    passed = all(c.passed for c in results if c.required)
    return L0Result(
        passed=passed,
        checks=results,
        workspace=str(ws),
        started_at=started,
        duration_s=round(time.monotonic() - t0, 3),
    )
