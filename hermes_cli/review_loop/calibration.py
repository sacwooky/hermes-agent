"""Calibration + regression meta-loop (WI-C10, ADD-ON C v2 Phase 8).

Keeps the Fusion judge honest over time — **offline/background, never in the live verdict path**.
Three jobs, all seeded by real outcomes:
1. **Per-juror scorecards** — emit role events (the conductor_vault `{role, kind, review_lane}`
   schema) per Fusion run so the existing instrumentation report computes per-lane precision /
   fused-escape-rate / availability. A chronically noisy or absent seat becomes visible/prunable.
2. **Regression cases** — every *escape* (the loop passed L2 but the human rejected at G3) becomes a
   permanent regression case, and the escape is attributed to the reviewer lanes that passed it.
3. **Pinned judge contract** — the judge slug + rubric version are pinned; a change must clear the
   regression suite (≥ the agreement bar) and ride a REVIEWED amendment.

Pure + fail-safe: event builders are pure; the emit helper appends to the same
`metrics/autonomy/role-events.jsonl` stream `conductor_vault.instrumentation` already reads, gated on
`HERMES_LEARNING_VAULT_ROOT`, and never raises into a caller.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

ROLE_EVENTS_FILE = "role-events.jsonl"
REGRESSION_FILE = "fusion-regression-cases.jsonl"

#: WI-C10 pinned judge contract. A change rides a REVIEWED amendment + must clear the
#: regression suite at >= AGREEMENT_BAR before it ships.
JUDGE_CONTRACT = {"slug": "cx/gpt-5.5-review", "rubric_version": "fusion-v1"}
AGREEMENT_BAR = 0.80


def _reviewer_lanes(fusion_result) -> list:
    """The lanes that participated in a binding review (jury seats that responded + judge)."""
    lanes = []
    for j in getattr(fusion_result, "jury", []) or []:
        if j.get("ok") and j.get("lane"):
            lanes.append(j["lane"])
    jl = (getattr(fusion_result, "judge", {}) or {}).get("lane")
    if jl:
        lanes.append(jl)
    return lanes


def fusion_review_events(fusion_result) -> list:
    """One reviewer `review` role event per participating lane (per-juror scorecard denominator)."""
    return [{"role": "reviewer", "kind": "review", "review_lane": lane}
            for lane in _reviewer_lanes(fusion_result)]


def g3_escape_events(reviewer_lanes: list) -> list:
    """One reviewer `review_escape` event per lane that passed an artifact the human later rejected
    at G3 (the per-lane escape-rate numerator)."""
    return [{"role": "reviewer", "kind": "review_escape", "review_lane": lane}
            for lane in (reviewer_lanes or [])]


def emit_role_events(events: list, *, vault_root: Optional[str] = None) -> bool:
    """Append role events to `metrics/autonomy/role-events.jsonl`. Fail-safe; no-op when the env
    is unset. Returns True if written."""
    try:
        root = vault_root or os.environ.get("HERMES_LEARNING_VAULT_ROOT")
        if not root or not events:
            return False
        d = Path(root) / "metrics" / "autonomy"
        d.mkdir(parents=True, exist_ok=True)
        now = int(time.time())
        with open(d / ROLE_EVENTS_FILE, "a", encoding="utf-8") as fh:
            for e in events:
                fh.write(json.dumps({**e, "recorded_at": now}, sort_keys=True, ensure_ascii=False) + "\n")
        return True
    except Exception:
        _log.debug("calibration: role-event emit skipped", exc_info=True)
        return False


def register_regression_case(case: dict, *, vault_root: Optional[str] = None) -> bool:
    """Append a G3-escape regression case. Fail-safe; no-op when the env is unset."""
    try:
        root = vault_root or os.environ.get("HERMES_LEARNING_VAULT_ROOT")
        if not root:
            return False
        d = Path(root) / "metrics" / "autonomy"
        d.mkdir(parents=True, exist_ok=True)
        rec = {**case, "kind": "fusion_regression_case", "recorded_at": int(time.time())}
        with open(d / REGRESSION_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
        return True
    except Exception:
        _log.debug("calibration: regression-case append skipped", exc_info=True)
        return False


def record_g3_escape(fusion_result, *, task_id: str, reject_reason: str,
                     vault_root: Optional[str] = None) -> None:
    """Background hook (offline): an artifact the loop PASSed at L2 was rejected by the human at G3.
    Attributes the escape to the passing reviewer lanes + files a permanent regression case."""
    lanes = _reviewer_lanes(fusion_result)
    emit_role_events(g3_escape_events(lanes), vault_root=vault_root)
    register_regression_case({
        "task_id": task_id,
        "fusion_run_id": getattr(fusion_result, "fusion_run_id", None),
        "reject_reason": reject_reason,
        "escaped_lanes": lanes,
    }, vault_root=vault_root)


def assert_judge_contract(slug: str, rubric_version: str, *, pinned: Optional[dict] = None) -> None:
    """Raise if the judge slug / rubric version drifts from the pinned WI-C10 contract.
    A legitimate change updates JUDGE_CONTRACT via a REVIEWED amendment after the regression suite passes."""
    p = pinned or JUDGE_CONTRACT
    if slug != p["slug"] or rubric_version != p["rubric_version"]:
        raise ValueError(
            f"judge contract drift: ({slug!r}, {rubric_version!r}) != pinned "
            f"({p['slug']!r}, {p['rubric_version']!r}) — a change requires a REVIEWED amendment "
            f"+ a passing regression suite")


def agreement_rate(labeled: list) -> float:
    """Fraction of labeled regression cases where the judge verdict matches the human label."""
    pairs = [(c.get("judge_verdict"), c.get("human_label")) for c in labeled or []]
    pairs = [(j, h) for j, h in pairs if j is not None and h is not None]
    if not pairs:
        return 0.0
    return round(sum(1 for j, h in pairs if j == h) / len(pairs), 4)


def meets_agreement_bar(labeled: list, *, bar: float = AGREEMENT_BAR) -> bool:
    """WI-C10 gate: a judge model/rubric change may ship only at >= the agreement bar."""
    return agreement_rate(labeled) >= bar


def per_juror_scorecard(role_events: list) -> dict:
    """Per-lane precision proxy (escape rate) + volume from emitted role events.
    `escape_rate = review_escape / review` per lane; lower is better. (conductor_vault's
    `compute_role_scorecards` produces the canonical fleet report; this is a local convenience.)"""
    by_lane: dict = {}
    for e in role_events or []:
        if e.get("role") != "reviewer":
            continue
        lane = e.get("review_lane")
        if not lane:
            continue
        d = by_lane.setdefault(lane, {"review": 0, "review_escape": 0})
        k = e.get("kind")
        if k in d:
            d[k] += 1
    out = {}
    for lane, c in by_lane.items():
        reviews = c["review"]
        out[lane] = {
            "reviews": reviews,
            "escapes": c["review_escape"],
            "escape_rate": round(c["review_escape"] / reviews, 4) if reviews else None,
        }
    return out
