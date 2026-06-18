"""ADD-ON C v2 — automated review loop.

Phase 1 ships the **L0 deterministic gate** (WI-C3): run tests / lint / type-check /
build / SAST in a sandbox *before any model review fires*. No LLM, no token cost. The
builder does not self-certify — the harness captures results and attests them
out-of-band as a ``l0_attestation`` kanban event. A required-check FAIL routes the task
back to the fix-retry loop before a single review token is spent.

Public surface (Phase 1):
- :func:`run_l0_gate` / :class:`L0Result` / :class:`L0CheckResult` — pure executor (no DB).
- :func:`record_l0_attestation` — out-of-band evidence writer (the only DB-touching piece).
- :func:`emit_l0_catchrate` — best-effort, default-off catch-rate metric.
"""

from hermes_cli.review_loop.l0_gate import (
    L0CheckResult,
    L0Result,
    run_l0_gate,
)

__all__ = [
    "L0CheckResult",
    "L0Result",
    "run_l0_gate",
    "record_l0_attestation",
    "emit_l0_catchrate",
]


def __getattr__(name: str):  # lazy re-export to keep l0_gate import-pure (no DB)
    if name == "record_l0_attestation":
        from hermes_cli.review_loop.attestation import record_l0_attestation

        return record_l0_attestation
    if name == "emit_l0_catchrate":
        from hermes_cli.review_loop.metrics import emit_l0_catchrate

        return emit_l0_catchrate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
