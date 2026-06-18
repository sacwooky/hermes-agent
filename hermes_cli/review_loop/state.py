"""Per-artifact review-loop state (WI-C2 + WI-C9, ADD-ON C v2 Phase 3).

The automated loop is **build → L0 → L1 → L2 → integrate**, with the human touching
it only at G3. This module gives that loop an explicit, observable, *resumable* state
**derived from the kanban event log** — not stored mutable state. Because it is derived,
a crash/restart re-derives the exact same position from the durable events: the loop is
idempotent and resumable by construction.

L1 (cheap screen, Phase 4) and L2 (Robin Fusion, Phase 6) are not built yet; their slots
exist here so later phases populate them without reshaping the model. Until L2 lands, the
existing signed Robin review verdict (`verdict_recorded`) stands in for the L2 result and
the `fusion_run_id` slot stays ``None``.

WI-C9: the only human-surfacing conditions on the per-artifact path are a **sticky block**
(retries exhausted), a **budget pause**, and **G3** (epic acceptance). :func:`human_surfaces`
enumerates exactly those — anything else means a human was inserted mid-loop, which the
tests forbid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from hermes_cli import kanban_db as kb

# Loop stages (terminal-ish position of the artifact in the loop).
BUILD = "build"
L0 = "l0"
L1 = "l1"
L2 = "l2"
INTEGRATED = "integrated"
BLOCKED = "blocked"      # sticky: needs operator
QUEUED = "queued"        # routed back to the builder (fix-retry in flight)

# Human-surface reasons (WI-C9).
SURFACE_STICKY_BLOCK = "sticky_block"
SURFACE_BUDGET_PAUSE = "budget_pause"
SURFACE_G3 = "g3_acceptance"


@dataclass
class LoopState:
    task_id: str
    stage: str
    l0_passed: bool = False
    l0_failures: int = 0
    l0_escalated: bool = False
    l1_passed: Optional[bool] = None      # None until Phase 4 (L1 screen)
    l2_verdict: Optional[str] = None      # "pass"/"block" (today: Robin review verdict)
    l2_rejected: bool = False             # verdict_rejected (e.g. signed-empty)
    fusion_run_id: Optional[str] = None   # set by Phase 6 (Robin Fusion) L2
    integrated: bool = False
    g3_pending: bool = False
    budget_paused: bool = False
    surfaces: list = field(default_factory=list)


def _events(conn, task_id):
    try:
        return kb.list_events(conn, task_id)
    except Exception:
        return []


def _payload(ev) -> dict:
    raw = getattr(ev, "payload", None)
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return raw or {}


def compute_loop_state(conn, task_id: str) -> LoopState:
    """Derive the artifact's loop position from its event log (pure read)."""
    evs = _events(conn, task_id)
    kinds = [(getattr(e, "kind", "") or "") for e in evs]

    built = "completed" in kinds
    l0_passed = "l0_gate_passed" in kinds
    l0_failures = sum(1 for k in kinds if k == "l0_gate_failed")
    l0_escalated = "l0_gate_escalated" in kinds

    # Latest review verdict wins (re-reviews after a fix).
    l2_verdict: Optional[str] = None
    l2_rejected = False
    fusion_run_id: Optional[str] = None
    for e in evs:
        k = getattr(e, "kind", "") or ""
        if k == "verdict_recorded":
            p = _payload(e)
            l2_verdict = (p.get("verdict") or "pass").lower()
            fusion_run_id = p.get("fusion_run_id") or fusion_run_id
            l2_rejected = False
        elif k == "review_blocked":
            l2_verdict = "block"
        elif k == "verdict_rejected":
            l2_rejected = True

    conformance_blocked = ("conformance_gate_block" in kinds) or ("conformance_escalated" in kinds)
    integrated = "integrate_task_created" in kinds
    g3_pending = ("acceptance_task_created" in kinds) and ("accepted" not in kinds)
    budget_paused = "pause_for_approval" in kinds
    sticky = l0_escalated or conformance_blocked or ("gave_up" in kinds)

    # Stage precedence: terminal/blocking states first, then progress.
    if integrated:
        stage = INTEGRATED
    elif sticky:
        stage = BLOCKED
    elif l2_verdict == "pass":
        stage = L2
    elif l2_rejected or l2_verdict == "block":
        stage = QUEUED
    elif l0_escalated:
        stage = BLOCKED
    elif l0_failures and not l0_passed:
        stage = QUEUED
    elif l0_passed:
        stage = L0
    elif built:
        stage = L0  # built, awaiting/at L0 gate
    else:
        stage = BUILD

    st = LoopState(
        task_id=task_id, stage=stage,
        l0_passed=l0_passed, l0_failures=l0_failures, l0_escalated=l0_escalated,
        l2_verdict=l2_verdict, l2_rejected=l2_rejected, fusion_run_id=fusion_run_id,
        integrated=integrated, g3_pending=g3_pending, budget_paused=budget_paused,
    )
    st.surfaces = human_surfaces(st)
    return st


def human_surfaces(state: LoopState) -> list:
    """WI-C9: the ONLY human-surfacing conditions on the per-artifact path.

    Returns a (possibly empty) list of surface reasons. A normal artifact that is
    building / passing L0 / under review / integrating returns ``[]`` — no human is
    inserted mid-loop. The human appears only on a sticky block, a budget pause, or G3.
    """
    out = []
    if state.stage == BLOCKED:
        out.append(SURFACE_STICKY_BLOCK)
    if state.budget_paused:
        out.append(SURFACE_BUDGET_PAUSE)
    if state.g3_pending:
        out.append(SURFACE_G3)
    return out


def loop_capacity(conformance: Optional[dict]) -> dict:
    """WI-C9 packet annotation: coarse capacity/confidence for the G3 packet.

    Phase 3 derives a minimal signal from conformance-verdict presence; Phase 6
    (Robin Fusion) replaces this with the real ``min(capacity, judge_confidence)``
    confidence object. ``full`` = a security conformance verdict is present and not a
    fail; ``degraded`` = missing/penalised — surfaced to the operator at G3.
    """
    sec = (conformance or {}).get("security") if isinstance(conformance, dict) else None
    if sec and (sec.get("verdict") in ("pass", "skip")):
        return {"capacity": "full", "confidence": "high", "degraded": False, "source": "phase3-conformance-derived"}
    if not conformance:
        return {"capacity": "unknown", "confidence": "low", "degraded": True, "source": "phase3-conformance-derived"}
    return {"capacity": "degraded", "confidence": "low", "degraded": True, "source": "phase3-conformance-derived"}
