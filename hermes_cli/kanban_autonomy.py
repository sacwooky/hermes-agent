"""Three-gate autonomy primitives for the kanban dispatcher.

Implements the autonomous-execution layer approved in
``decision-autonomous-three-gate-orchestration-v1`` (vault run
2026-06-12-351): between PRD approval (G1) and epic acceptance (G3),
boards run unattended — build → Robin review → QA → integrate → demo —
with the dispatcher tick driving three sweeps from this module:

1. **Epic acceptance (G3)** — when an epic root completes, generate
   exactly one non-spawnable acceptance task + a pending
   ``epic_acceptance`` approval carrying the deliverable evidence
   (demo URL, integration SHA, verdict/QA links). Operator acceptance
   (``hermes kanban accept <id>`` or an approved approval row)
   completes it; rejection spawns a fix story and re-requests
   acceptance once the fix lands.

2. **Verdict integrity** — :func:`record_review_verdict` is the only
   sanctioned path for honoring a Robin review verdict. It requires an
   HMAC-SHA256 signature with the shared Robin verdict key, an
   out-of-band fetch channel, and an existing vault run record. A
   verdict that fails any check leaves the task in place and emits
   ``verdict_rejected`` (closes the FluxCreds R6 fabrication class).

3. **Unintegrated-work sweep** — done tasks whose branch never landed
   on the project's integration branch get an ``integrate`` task
   auto-created for the integrator lane (closes the Hermes Console
   T11 "reviewer-passed work sat in a worktree for 11 days" class).

Every sweep is feature-gated by config passed from the gateway
dispatcher (``kanban.autonomy`` + per-board overrides) so boards opt
in one at a time (pilot-first rollout).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from hermes_cli import kanban_db as kb

_log = logging.getLogger(__name__)

# E4-S4: startup checks — headroom learn OFF, no :8797 chain
try:
    from hermes_cli.headroom_guard import check_headroom_learn_off, check_no_8797_base_url
    for _w in check_headroom_learn_off() + check_no_8797_base_url():
        _log.warning("headroom_guard startup: %s", _w)
except Exception:
    pass

DEFAULT_VERDICT_KEY_PATH = "~/.hermes/credentials/robin-verdict-key"
DEFAULT_VAULT_ROOT = "/srv/fluxlabs/vault/conductor-vault"
VALID_VERDICTS = frozenset({"pass", "pass-with-notes", "block"})

# --- v18 Supreme Court Review Contract (structured output) ------------------
# Source of truth: conductor-vault
# wiki/projects/project-intake-discovery-process.md ("Supreme Court Review
# Contract"). For each review TYPE the reviewer MUST emit structured output
# (not freeform); a verdict missing any required field, or missing the rubric
# scorecard for its type, is treated as a HOLLOW verdict → verdict_rejected +
# requeue (fail-closed). Enforcement is OPT-IN, keyed on the payload declaring
# a ``review_type`` — legacy/fusion verdicts (no review_type) keep the existing
# lane-based R8 hollow check unchanged.
SC_REVIEW_TYPES = frozenset(
    {"wireframe", "prd", "code", "final-delivery", "final_delivery", "final", "general"}
)
# Review types that require a rubric scorecard (templates/rubrics/<type>-rubric.md).
SC_SCORECARD_TYPES = frozenset(
    {"wireframe", "prd", "code", "final-delivery", "final_delivery", "final", "general"}
)
# Every structured field the contract requires, regardless of review type.
SC_REQUIRED_FIELDS = (
    "verdict",
    "confidence",
    "scorecard",
    "blocking_issues",
    "advisory_issues",
    "missing_skill_findings",
    "required_repair_actions",
    "evidence_reviewed",
    "calibration_substrate_flags",
)
# Fields that must be JSON lists when present (the contract specifies [] shape).
SC_LIST_FIELDS = (
    "blocking_issues",
    "advisory_issues",
    "missing_skill_findings",
    "required_repair_actions",
    "evidence_reviewed",
    "calibration_substrate_flags",
)
# v18 verdict vocabulary → lifecycle outcome. Approved / Approved-with-minor-notes
# pass; Rejected-for-revision / Rejected-wrong-skill-stack block.
SC_PASS_VERDICTS = frozenset({"approved", "approved_with_minor_notes"})
SC_BLOCK_VERDICTS = frozenset(
    {"rejected_for_revision", "rejected_wrong_skill_stack"}
)

# Provenance channels a verdict may legitimately arrive through. A file
# that "appeared" in a local worktree is NOT one of them — that is the
# exact fabrication vector this module exists to close.
VALID_FETCH_CHANNELS = frozenset({"robin-api", "robin-ssh"})
# Cap integrate-task creation per tick so a legacy backlog drains
# gradually instead of flooding the integrator lane in one tick.
MAX_INTEGRATE_TASKS_PER_TICK = 3


# ---------------------------------------------------------------------------
# Verdict signing / verification
# ---------------------------------------------------------------------------


def _resolve_key_path(key_path: Optional[str] = None) -> Path:
    return Path(key_path or DEFAULT_VERDICT_KEY_PATH).expanduser()


def _load_verdict_key(key_path: Optional[str] = None) -> bytes:
    """Read the shared HMAC key. Raises ``FileNotFoundError`` if absent.

    The key bytes never enter any LLM context: signing and verification
    happen in-process (CLI helper / dispatcher), and callers only ever
    see booleans and hex digests.
    """
    p = _resolve_key_path(key_path)
    data = p.read_bytes().strip()
    if not data:
        raise ValueError(f"verdict key at {p} is empty")
    # Match robin:send-review.sh's key handling: a hex-string key file is
    # decoded to its raw bytes (the key was generated as `xxd -p` hex); a
    # non-hex key is used as raw bytes. Without this the HMAC key differs from
    # Robin's signer and EVERY signature mis-verifies (verdict-record bug,
    # 2026-06-13).
    try:
        return bytes.fromhex(data.decode("ascii").strip())
    except (ValueError, UnicodeDecodeError):
        return data


def sign_verdict_payload(payload: bytes, *, key_path: Optional[str] = None) -> str:
    """Return the hex HMAC-SHA256 signature of *payload*."""
    key = _load_verdict_key(key_path)
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def verify_verdict_signature(
    payload: bytes,
    signature: str,
    *,
    key_path: Optional[str] = None,
) -> bool:
    """Constant-time check of *signature* against *payload*."""
    try:
        expected = sign_verdict_payload(payload, key_path=key_path)
    except (FileNotFoundError, ValueError):
        return False
    return hmac.compare_digest(expected, (signature or "").strip().lower())


def _normalize_review_type(review_type: Any) -> Optional[str]:
    """Canonicalize a payload ``review_type`` to a known SC type, or None.

    Accepts case / hyphen / underscore variants (``Final-Delivery`` →
    ``final-delivery``). Returns the canonical token when recognized, else
    ``None`` (caller treats unrecognized/absent as "not an SC contract").
    """
    if not isinstance(review_type, str):
        return None
    t = review_type.strip().lower().replace(" ", "-").replace("_", "-")
    # Fold the underscore aliases we accept in SC_REVIEW_TYPES.
    aliases = {"final-delivery": "final-delivery", "final_delivery": "final-delivery"}
    t = aliases.get(t, t)
    canon = t.replace("-", "_")
    if t in SC_REVIEW_TYPES or canon in SC_REVIEW_TYPES:
        return t
    return None


def enforce_supreme_court_contract(
    payload: dict, review_type: str
) -> Optional[str]:
    """Validate a v18 Supreme Court structured verdict. Fail-closed.

    Returns a rejection reason string if the payload violates the contract,
    or ``None`` if it conforms. The contract (vault
    ``project-intake-discovery-process.md`` → "Supreme Court Review Contract"):
    every verdict MUST carry the structured fields in :data:`SC_REQUIRED_FIELDS`,
    the rubric ``scorecard`` for its review type must be a non-empty object, the
    list-shaped fields must be JSON lists, and ``verdict`` must be drawn from the
    v18 vocabulary (Approved / Approved-with-minor-notes / Rejected-for-revision
    / Rejected-wrong-skill-stack).

    A verdict missing any required field, or missing the scorecard for its
    review type, is hollow → reject + requeue (same fail-closed posture as the
    R8 hollow-PASS check).
    """
    # 1. Every required structured field must be PRESENT (key exists). Absence
    #    is the hollow-verdict signal the contract guards against.
    missing = [f for f in SC_REQUIRED_FIELDS if f not in payload]
    if missing:
        return (
            f"hollow SC verdict ({review_type}): missing required structured "
            f"field(s) {missing} — Supreme Court Review Contract requires "
            f"{list(SC_REQUIRED_FIELDS)} (fail-closed, requeue for re-review)"
        )

    # 2. Scorecard must be a non-empty object for types that carry a rubric.
    scorecard = payload.get("scorecard")
    if review_type in SC_SCORECARD_TYPES or review_type.replace("-", "_") in SC_SCORECARD_TYPES:
        if not isinstance(scorecard, dict) or not scorecard:
            return (
                f"hollow SC verdict ({review_type}): rubric scorecard is "
                f"missing or empty — every {review_type} review must carry the "
                f"templates/rubrics/{review_type}-rubric.md scorecard "
                f"(fail-closed, requeue)"
            )
    elif scorecard is not None and not isinstance(scorecard, dict):
        return f"SC verdict ({review_type}): scorecard must be a JSON object"

    # 3. List-shaped fields must actually be lists (shape contract).
    for f in SC_LIST_FIELDS:
        if not isinstance(payload.get(f), list):
            return (
                f"SC verdict ({review_type}): field {f!r} must be a JSON list "
                f"(got {type(payload.get(f)).__name__})"
            )

    # 4. verdict must be in the v18 vocabulary.
    raw = str(payload.get("verdict", "")).strip().lower().replace("-", "_").replace(" ", "_")
    if raw not in SC_PASS_VERDICTS and raw not in SC_BLOCK_VERDICTS:
        return (
            f"SC verdict ({review_type}): verdict={payload.get('verdict')!r} is "
            f"not in the v18 vocabulary "
            f"{sorted(SC_PASS_VERDICTS | SC_BLOCK_VERDICTS)}"
        )
    return None


def record_review_verdict(
    conn: sqlite3.Connection,
    task_id: str,
    payload_json: str,
    signature: str,
    *,
    fetched_via: str,
    key_path: Optional[str] = None,
    vault_root: Optional[str] = None,
    board: Optional[str] = None,
) -> dict:
    """Honor a signed Robin review verdict — the ONLY path out of review.

    Verification ladder (all must pass, in order):
      1. ``fetched_via`` is an out-of-band channel (``robin-api`` /
         ``robin-ssh``) — local-worktree files are categorically refused.
      2. HMAC-SHA256 signature over the exact payload bytes verifies
         against the shared Robin verdict key.
      3. Payload parses, names this task, and carries a valid verdict.
      4. The referenced vault run record exists; when the payload pins
         ``run_record_sha256`` the file hash must match.

    On a ``pass`` / ``pass-with-notes`` verdict the task completes (its
    summary carries the verdict + run-record pointer). On ``block`` the
    findings land as a comment and the task returns to ``ready`` so the
    builder lane respawns with the findings in context.

    Any failure emits ``verdict_rejected`` with the reason and leaves
    task state untouched — escalation surfaces it to the operator.

    Returns ``{"ok": bool, "verdict": str|None, "reason": str|None}``.
    """

    def _reject(reason: str) -> dict:
        with kb.write_txn(conn):
            kb._append_event(
                conn,
                task_id,
                "verdict_rejected",
                {
                    "reason": reason,
                    "fetched_via": fetched_via,
                    "signature_prefix": (signature or "")[:16],
                },
            )
        _log.warning("verdict rejected for %s: %s", task_id, reason)
        return {"ok": False, "verdict": None, "reason": reason}

    if fetched_via not in VALID_FETCH_CHANNELS:
        return _reject(
            f"fetched_via={fetched_via!r} is not an out-of-band channel "
            f"(allowed: {sorted(VALID_FETCH_CHANNELS)}); verdicts must be "
            "pulled from Robin's host, never trusted from local files"
        )

    try:
        payload = json.loads(payload_json)
    except (json.JSONDecodeError, TypeError):
        return _reject("payload is not valid JSON")
    if not isinstance(payload, dict):
        return _reject("payload is not a JSON object")

    # Signature is HMAC over the CANONICAL payload — matching the deployed
    # robin:send-review.sh signer: json.dumps(payload, sort_keys=True,
    # separators=(",",":"), ensure_ascii=False). Verifying over canonical form
    # (not raw bytes) lets the courier pass the payload however it serialized
    # it without a byte-exact round-trip. Fall back to raw-bytes verification
    # for the legacy/test path.
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if not (
        verify_verdict_signature(canon.encode("utf-8"), signature, key_path=key_path)
        or verify_verdict_signature(payload_json.encode("utf-8"), signature, key_path=key_path)
    ):
        return _reject("HMAC signature verification failed")

    if payload.get("task_id") != task_id:
        return _reject(
            f"payload.task_id={payload.get('task_id')!r} does not match {task_id!r}"
        )

    # v18 Supreme Court Review Contract (fail-closed, opt-in via review_type).
    # When the payload declares a recognized review_type, it MUST satisfy the
    # full structured-output schema (required fields + rubric scorecard + v18
    # verdict vocabulary). A non-conforming SC verdict is hollow → reject +
    # requeue, the same posture R8 uses for a lane-less PASS. Legacy/fusion
    # verdicts (no review_type) skip this and rely on the existing R8 check.
    sc_review_type = _normalize_review_type(payload.get("review_type"))
    if sc_review_type is not None:
        violation = enforce_supreme_court_contract(payload, sc_review_type)
        if violation:
            return _reject(violation)

    # Normalize Robin's verdict vocabulary to the two lifecycle outcomes.
    # send-review.sh emits BUILD_READY / PASS / PASS_WITH_NOTES /
    # CHANGES_REQUESTED / BLOCK / REVISE (any case); the v18 SC contract emits
    # Approved / Approved-with-minor-notes / Rejected-for-revision /
    # Rejected-wrong-skill-stack. Map all of them to pass|block.
    raw_verdict = str(payload.get("verdict", "")).strip().lower().replace("-", "_")
    PASS_SET = {
        "pass", "build_ready", "pass_with_notes", "approve", "approved", "ok",
        # v18 SC vocabulary
        "approved_with_minor_notes",
    }
    BLOCK_SET = {"block", "blocked", "changes_requested", "revise", "reject",
                 "rejected", "do_not_build", "fail",
                 # v18 SC vocabulary
                 "rejected_for_revision", "rejected_wrong_skill_stack"}
    if raw_verdict in PASS_SET:
        verdict = "pass"
    elif raw_verdict in BLOCK_SET:
        verdict = "block"
    else:
        return _reject(f"unrecognized verdict={raw_verdict!r} from Robin")

    # run_record is OPTIONAL: send-review.sh's signed verdict is itself the
    # provenance (HMAC + out-of-band fetch from Robin's host). When a payload
    # DOES pin a vault run_record, validate it; otherwise the signature stands.
    run_record = str(payload.get("run_record") or "").strip()
    if run_record:
        vault = Path(vault_root or DEFAULT_VAULT_ROOT)
        record_path = (vault / run_record) if not os.path.isabs(run_record) else Path(run_record)
        try:
            record_path = record_path.resolve()
            vault_resolved = vault.resolve()
        except OSError:
            return _reject(f"run_record path {run_record!r} cannot be resolved")
        if vault_resolved not in record_path.parents and record_path != vault_resolved:
            return _reject(f"run_record {run_record!r} is outside the vault root")
        if not record_path.is_file():
            return _reject(f"run_record {run_record!r} does not exist in the vault")
        pinned_hash = str(payload.get("run_record_sha256", "")).strip().lower()
        if pinned_hash:
            actual = hashlib.sha256(record_path.read_bytes()).hexdigest()
            if not hmac.compare_digest(actual, pinned_hash):
                return _reject("run_record_sha256 does not match the vault file")

    findings = payload.get("findings") or payload.get("risks") or []
    model_lane = payload.get("model_lane") or payload.get("lane")
    lanes_run = payload.get("lanes_run")
    fusion_run_id = payload.get("fusion_run_id")  # Phase 6: L2 Robin Fusion run id
    fusion_confidence = payload.get("confidence")  # Phase 6: confidence × risk object

    # R8 — reject HOLLOW pass verdicts. A real review records the lane(s) that ran;
    # a lane that errored/quota-failed and emitted an empty PASS would not. BLOCK
    # verdicts are honored regardless (a block is still actionable). Validity =
    # recorded lane id + signature (project-intake-discovery-process Review handoff).
    # Placed BEFORE the verdict_recorded event so a hollow pass never emits the
    # event the integration sweeps trust as proof of a real review.
    if verdict == "pass" and not (model_lane or (isinstance(lanes_run, list) and lanes_run)):
        return _reject(
            "hollow PASS verdict: no review lane recorded (model_lane / lane / lanes_run) "
            "— the review lane likely failed (quota/timeout); left in review for requeue"
        )

    with kb.write_txn(conn):
        kb._append_event(
            conn,
            task_id,
            "verdict_recorded",
            {
                "verdict": verdict,
                "model_lane": model_lane,
                "findings_count": len(findings) if isinstance(findings, list) else None,
                "run_record": run_record,
                "fetched_via": fetched_via,
                "signature_prefix": signature[:16],
                "signature": signature,  # full HMAC -> exact dedup (verdict courier)
                # Phase 6: carry the Fusion run id + confidence so LoopState.fusion_run_id
                # populates and the G3 packet can surface real confidence×risk.
                "fusion_run_id": fusion_run_id,
                "confidence": fusion_confidence,
                # v18: record the SC review type when this was a structured
                # Supreme Court verdict (None for legacy/fusion verdicts).
                "review_type": sc_review_type,
            },
        )

    if verdict == "pass":
        summary = (
            f"Robin verdict: {raw_verdict.upper()}"
            + (f" (lane: {model_lane})" if model_lane else "")
            + (f". Run record: {run_record}" if run_record else " (signed verdict).")
        )
        try:
            kb.complete_task(
                conn,
                task_id,
                result=summary,
                summary=summary,
                metadata={
                    "verdict": raw_verdict,
                    "model_lane": model_lane,
                    "run_record": run_record or None,
                    "findings": findings,
                    "commit": payload.get("commit"),
                },
                board=board,
            )
        except kb.AcceptanceRequiredError:
            # Manually-gated board (e.g. fleet-key): a PASS review must NOT
            # auto-complete the task — the operator still owns the final
            # `accept`. Rather than propagate the error (which loses the
            # signed verdict), park the task in the board's standard
            # acceptance-required blocked state with the verdict on record,
            # so it surfaces in the morning report's ACCEPT-READY list.
            with kb.write_txn(conn):
                now = int(time.time())
                conn.execute(
                    "INSERT INTO task_comments (task_id, author, body, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        task_id,
                        "robin-review",
                        f"REVIEW PASS ({model_lane or 'robin'}) — signed verdict. "
                        f"Board requires operator acceptance; awaiting "
                        f"`hermes kanban accept {task_id}`.\n{summary}",
                        now,
                    ),
                )
                conn.execute(
                    "UPDATE tasks SET status = 'blocked', claim_lock = NULL, "
                    "claim_expires = NULL, worker_pid = NULL, current_run_id = NULL "
                    "WHERE id = ?",
                    (task_id,),
                )
                kb._append_event(
                    conn,
                    task_id,
                    "blocked",
                    {
                        "reason": (
                            f"acceptance-required: Robin review PASSED "
                            f"({model_lane or 'robin'}); operator must accept"
                        ),
                        "verdict": "pass",
                    },
                )
            return {
                "ok": True,
                "verdict": verdict,
                "reason": "awaiting_operator_acceptance",
            }
        except kb.QaGateError as qa_err:
            # v18 QA gate refused this auto-completion (e.g. a roll-up whose
            # QA children aren't all done). Don't crash the autonomy loop or
            # lose the signed verdict — park the task blocked with the gate
            # detail on record so the operator/flow-manager can resolve it.
            detail = "; ".join(qa_err.violations) or "QA evidence missing"
            with kb.write_txn(conn):
                now = int(time.time())
                conn.execute(
                    "INSERT INTO task_comments (task_id, author, body, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        task_id,
                        "robin-review",
                        f"REVIEW PASS ({model_lane or 'robin'}) — signed verdict, "
                        f"but v18 QA gate blocked auto-completion: {detail}.\n"
                        f"{summary}",
                        now,
                    ),
                )
                conn.execute(
                    "UPDATE tasks SET status = 'blocked', claim_lock = NULL, "
                    "claim_expires = NULL, worker_pid = NULL, current_run_id = NULL "
                    "WHERE id = ?",
                    (task_id,),
                )
                kb._append_event(
                    conn,
                    task_id,
                    "blocked",
                    {
                        "reason": f"qa-gate: {detail}",
                        "verdict": "pass",
                    },
                )
            return {
                "ok": True,
                "verdict": verdict,
                "reason": "qa_gate_blocked",
            }
        return {"ok": True, "verdict": verdict, "reason": None}

    # verdict == "block": findings go back to the builder lane.
    findings_text = "\n".join(
        f"- {f}" if isinstance(f, str) else f"- {json.dumps(f, ensure_ascii=False)}"
        for f in (findings if isinstance(findings, list) else [findings])
    ) or "(no structured findings in verdict payload)"
    with kb.write_txn(conn):
        now = int(time.time())
        conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                task_id,
                "robin-review",
                f"REVIEW BLOCK ({model_lane or 'robin'}):\n{findings_text}\n"
                f"Run record: {run_record}",
                now,
            ),
        )
        conn.execute(
            "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
            "claim_expires = NULL, worker_pid = NULL, current_run_id = NULL "
            "WHERE id = ? AND status IN ('running', 'review', 'done')",
            (task_id,),
        )
        kb._append_event(
            conn,
            task_id,
            "review_blocked",
            {"run_record": run_record, "model_lane": model_lane},
        )
    return {"ok": True, "verdict": "block", "reason": None}


# ---------------------------------------------------------------------------
# Epic acceptance (G3)
# ---------------------------------------------------------------------------


def _has_event(
    conn: sqlite3.Connection, task_id: str, kind: str
) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT id, payload, created_at FROM task_events "
        "WHERE task_id = ? AND kind = ? ORDER BY id DESC LIMIT 1",
        (task_id, kind),
    ).fetchone()


def _stamp_work_item_type(
    conn: sqlite3.Connection, task_id: str, work_item_type: str
) -> None:
    """Idempotent task_metadata upsert usable inside an open write_txn."""
    now = int(time.time())
    conn.execute(
        "INSERT INTO task_metadata "
        "(task_id, work_item_type, tags_json, updated_at, updated_by) "
        "VALUES (?, ?, '[]', ?, 'autonomy') "
        "ON CONFLICT(task_id) DO UPDATE SET "
        "work_item_type = excluded.work_item_type, "
        "updated_at = excluded.updated_at",
        (task_id, work_item_type, now),
    )


def _harvest_epic_artifacts(conn: sqlite3.Connection, epic_id: str) -> list[dict]:
    """Collect deliverable evidence from the epic's story graph.

    Stories are the *parents* of the epic root in ``task_links``
    (decompose links the root under every child so it waits for the
    graph). Best-effort: pull demo URLs / commits / verdict run records
    out of each story's latest completed-run metadata.
    """
    artifacts: list[dict] = []
    story_rows = conn.execute(
        "SELECT t.id, t.title, t.branch_name FROM tasks t "
        "JOIN task_links l ON l.parent_id = t.id "
        "WHERE l.child_id = ? ORDER BY t.created_at",
        (epic_id,),
    ).fetchall()
    for srow in story_rows:
        entry: dict[str, Any] = {"task_id": srow["id"], "title": srow["title"]}
        if srow["branch_name"]:
            entry["branch"] = srow["branch_name"]
        run_row = conn.execute(
            "SELECT summary, metadata FROM task_runs "
            "WHERE task_id = ? AND outcome = 'completed' "
            "ORDER BY started_at DESC LIMIT 1",
            (srow["id"],),
        ).fetchone()
        if run_row is not None:
            meta = run_row["metadata"]
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = None
            if isinstance(meta, dict):
                for key in (
                    "demo_url",
                    "integration_sha",
                    "commit",
                    "run_record",
                    "verdict",
                    "qa_evidence",
                ):
                    if meta.get(key):
                        entry[key] = meta[key]
        # E4-S1/S2: Guard conformance-evidence fields against Headroom placeholders
        from hermes_cli.headroom_guard import assert_no_placeholders
        try:
            assert_no_placeholders(
                {k: v for k, v in entry.items() if k in ("qa_evidence", "screenshots", "functional_test_results", "prd_conformance_matrix")},
                context=f"harvest_artifacts:{srow['id']}"
            )
        except Exception:
            _log.warning("headroom_guard: placeholder detected in artifact evidence for %s — clearing field", srow["id"])
            for _field in ("qa_evidence", "screenshots", "functional_test_results", "prd_conformance_matrix"):
                entry.pop(_field, None)
        artifacts.append(entry)
    return artifacts


# ---------------------------------------------------------------------------
# B4/B5 author-aware conformance guard helpers
# ---------------------------------------------------------------------------

#: Max conformance fix-story retries before escalate-once (WI-QA4 bound).
WI_QA4_MAX_CONFORMANCE_RETRIES: int = int(
    os.environ.get("HERMES_CONFORMANCE_MAX_RETRIES", "3")
)

#: ADD-ON C v2 WI-C3 — max times the L0 deterministic gate routes a review task
#: back to the builder before escalating once. Env-overridable, same style as the
#: conformance retry bound above.
HERMES_L0_GATE_MAX_RETRIES: int = int(
    os.environ.get("HERMES_L0_GATE_MAX_RETRIES", "3")
)

#: Keywords in an epic title that flag it as high-risk for cross-check (B5).
_HIGH_RISK_TITLE_KEYWORDS = frozenset(
    {"auth", "payment", "credential", "secret", "token", "oauth", "phase-3", "phase3"}
)


def _normalize_provider(lane_or_model: str) -> str:
    """Collapse a lane/model string to its provider family.

    Returns ``"claude"``, ``"gemini"``, or ``"other"``.
    """
    if not lane_or_model:
        return "other"
    s = lane_or_model.lower()
    if any(k in s for k in ("gemini", "jake-vertex", "vertex")):
        return "gemini"
    if any(k in s for k in ("claude", "anthropic")):
        return "claude"
    return "other"


def _get_epic_author_provider(conn: sqlite3.Connection, epic_id: str) -> str:
    """Infer the author provider family for an epic from chain telemetry.

    Scans story-completion and review events attached to the epic for a
    ``model_lane`` / ``lane`` field and returns the most common provider
    family (``"claude"`` or ``"gemini"``).  Falls back to ``"claude"``
    when no chain telemetry is recorded (the overwhelmingly common case on
    Jake today).
    """
    counts: dict[str, int] = {}
    try:
        events = kb.list_events(conn, epic_id)
    except Exception:
        events = []
    for e in events:
        kind = getattr(e, "kind", "") or ""
        if kind not in (
            "story_completed",
            "completion_recorded",
            "review_approved",
            "build_complete",
        ):
            continue
        payload = getattr(e, "payload", None) or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {}
        lane = payload.get("model_lane") or payload.get("lane") or ""
        provider = _normalize_provider(lane)
        if provider in ("claude", "gemini"):
            counts[provider] = counts.get(provider, 0) + 1
    if not counts:
        return "claude"  # default: Jake builders are Claude-Code
    return max(counts, key=lambda k: counts[k])


def _required_conformance_provider(author_provider: str) -> str:
    """Return the required review provider given the author provider."""
    if author_provider == "gemini":
        return "claude"
    return "gemini"  # claude or unknown → gemini


def _is_high_risk_epic(conn: sqlite3.Connection, epic_id: str, title: str) -> bool:
    """Return True if the epic is flagged as security-sensitive / Phase-3.

    Checks the task title for known keywords AND looks for an explicit
    ``security_sensitive`` or ``phase_3_flagged`` metadata key.
    """
    # Title scan
    title_lower = (title or "").lower()
    if any(kw in title_lower for kw in _HIGH_RISK_TITLE_KEYWORDS):
        return True
    # Metadata scan (upsert_task_metadata stores extra_json as a JSON blob)
    try:
        row = conn.execute(
            "SELECT extra_json FROM task_metadata WHERE task_id = ?", (epic_id,)
        ).fetchone()
        if row and row["extra_json"]:
            extra = json.loads(row["extra_json"]) if isinstance(row["extra_json"], str) else row["extra_json"]
            if extra.get("security_sensitive") or extra.get("phase_3_flagged"):
                return True
    except Exception:
        pass
    return False


def _harvest_conformance_verdicts(conn: sqlite3.Connection, epic_id: str) -> dict:
    """Read the latest conformance verdict events recorded on the epic task.

    Returns dict with keys ``security`` / ``performance`` / ``accessibility``,
    each a dict or ``None``.  Each dict may include a ``"xcheck"`` sub-key
    with the second independent-provider verdict (B5).
    """
    verdicts: dict[str, Any] = {}
    for axis, kind in (
        ("security", "conformance_verdict_security"),
        ("performance", "conformance_verdict_perf"),
        ("accessibility", "conformance_verdict_a11y"),
        # Advisory axis (decision-experience-first-builds-v1, Stage 3): surfaced in the
        # G3 acceptance packet only — NEVER on any blocking/crosscheck path.
        ("design_quality", "conformance_verdict_design_quality"),
    ):
        row = _has_event(conn, epic_id, kind)
        if row is not None:
            try:
                payload = row["payload"]
                v = json.loads(payload) if isinstance(payload, str) else payload
            except Exception:
                v = {"verdict": "unknown", "error": "parse_failed"}
            # B5: also harvest the cross-check opinion
            xrow = _has_event(conn, epic_id, f"{kind}_xcheck")
            if xrow is not None:
                try:
                    xp = xrow["payload"]
                    v["xcheck"] = json.loads(xp) if isinstance(xp, str) else xp
                except Exception:
                    v["xcheck"] = {"verdict": "unknown", "error": "parse_failed"}
            verdicts[axis] = v
    return verdicts


# Advisory conformance axes are surfaced in the G3 packet but NEVER gate integration;
# the invariant is enforced in record_conformance_verdict (not merely documented).
_ADVISORY_CONFORMANCE_AXES = frozenset({"design_quality"})


def record_conformance_verdict(
    conn: sqlite3.Connection,
    epic_id: str,
    axis: str,
    verdict: str,
    *,
    lane: str,
    signed: bool = True,
    findings: "list | None" = None,
    run_record: "str | None" = None,
    crosscheck: bool = False,
) -> None:
    """Record a Robin-signed conformance verdict on the epic task.

    Called by the verdict courier (record-robin-verdict.sh equivalent for
    conformance).

    :param axis: ``"security"`` | ``"perf"`` | ``"a11y"`` | ``"design_quality"``
        (``design_quality`` is ADVISORY — recorded + surfaced in the G3 packet,
        never on a blocking/crosscheck path; decision-experience-first-builds-v1).
    :param verdict: ``"pass"`` | ``"fail"`` | ``"skip"``
    :param crosscheck: True when this is the second independent-provider
        opinion required for high-risk epics (B5).

    B4 author-aware independence enforcement: the provider of *lane* must
    differ from the provider that authored the epic.  A same-provider
    verdict is recorded as ``verdict_rejected`` (extends the
    signed-empty=rejected rule — no self-review on any axis).
    """
    kind_map = {
        "security": "conformance_verdict_security",
        "perf": "conformance_verdict_perf",
        "a11y": "conformance_verdict_a11y",
        "design_quality": "conformance_verdict_design_quality",  # advisory; never blocks
    }
    kind = kind_map.get(axis)
    if not kind:
        raise ValueError(f"Unknown conformance axis: {axis!r}")
    # ENFORCED invariant (Robin run 540): an advisory axis can NEVER be recorded on the
    # crosscheck/blocking path. Previously documented-only; now code-enforced.
    if axis in _ADVISORY_CONFORMANCE_AXES and crosscheck:
        raise ValueError(
            f"advisory conformance axis {axis!r} cannot be recorded on the crosscheck/"
            f"blocking path; it never gates integration"
        )
    if crosscheck:
        kind = f"{kind}_xcheck"

    # B4: author-aware independence gate.
    # Primary verdicts (crosscheck=False): lane-provider must != author-provider.
    # Cross-check verdicts (crosscheck=True): lane-provider must != primary verdict's
    #   lane-provider (independence from the primary, not from the author — the cross-
    #   check deliberately lets the "other" provider challenge the primary's conclusion).
    author_provider = _get_epic_author_provider(conn, epic_id)
    lane_provider = _normalize_provider(lane)
    if axis in _ADVISORY_CONFORMANCE_AXES:
        # Advisory axes never gate integration, so the B4/B5 independence gate does not
        # apply (and crosscheck=True was already rejected above). Record as-is.
        pass
    elif not crosscheck:
        required_provider = _required_conformance_provider(author_provider)
        if lane_provider not in ("other", "") and lane_provider == author_provider:
            _log.warning(
                "conformance verdict rejected on epic %s axis=%s: "
                "lane provider %r == author provider %r (primary self-review); "
                "required lane provider: %r",
                epic_id, axis, lane_provider, author_provider, required_provider,
            )
            kb._append_event(conn, epic_id, "verdict_rejected", {
                "reason": "author_provider_self_review",
                "axis": axis,
                "lane": lane,
                "lane_provider": lane_provider,
                "author_provider": author_provider,
                "required_provider": required_provider,
                "crosscheck": False,
            })
            raise ValueError(
                f"conformance_verdict_rejected: epic {epic_id} axis={axis} lane_provider "
                f"{lane_provider!r} == author_provider {author_provider!r}; "
                f"independence required (use {required_provider!r} lane)"
            )
    else:
        # Cross-check: must differ from the primary verdict's provider.
        primary_row = _has_event(conn, epic_id, kind_map[axis])
        if primary_row is not None:
            try:
                p = primary_row["payload"]
                primary_lp = (json.loads(p) if isinstance(p, str) else p).get("lane_provider", "other")
            except Exception:
                primary_lp = "other"
            if lane_provider not in ("other", "") and primary_lp not in ("other", "") and lane_provider == primary_lp:
                _log.warning(
                    "cross-check verdict rejected on epic %s axis=%s: "
                    "xcheck provider %r == primary provider %r (no independence); ",
                    epic_id, axis, lane_provider, primary_lp,
                )
                kb._append_event(conn, epic_id, "verdict_rejected", {
                    "reason": "crosscheck_same_provider_as_primary",
                    "axis": axis,
                    "lane": lane,
                    "lane_provider": lane_provider,
                    "primary_provider": primary_lp,
                    "crosscheck": True,
                })
                raise ValueError(
                    f"conformance_verdict_rejected: epic {epic_id} axis={axis} xcheck "
                    f"provider {lane_provider!r} == primary provider {primary_lp!r}; "
                    f"independence required"
                )

    payload = {
        "verdict": verdict,
        "lane": lane,
        "lane_provider": lane_provider,
        "author_provider": author_provider,
        "signed": signed,
        "findings": findings or [],
        "run_record": run_record,
        "crosscheck": crosscheck,
    }
    with kb.write_txn(conn):
        kb._append_event(conn, epic_id, kind, payload)


def _epic_for_story(conn: sqlite3.Connection, story_id: str) -> "str | None":
    """Return the epic id for *story_id*, or None.

    decompose links the epic root as a CHILD of every story (so the epic waits on the
    graph), so the epic is the story's child in ``task_links`` whose metadata marks it an
    epic. Best-effort; advisory-path use only.
    """
    try:
        row = conn.execute(
            "SELECT l.child_id FROM task_links l "
            "JOIN task_metadata m ON m.task_id = l.child_id "
            "WHERE l.parent_id = ? AND m.work_item_type = 'epic' LIMIT 1",
            (story_id,),
        ).fetchone()
    except Exception:
        return None
    return row["child_id"] if row else None


def _latest_selected_direction(task_id: str) -> "dict | None":
    """Return the approved-direction artifact from the LATEST wireframe approval for
    *task_id*, or None.

    ``list_task_approvals`` is newest-first, so the first wireframe approval is the most
    recent — this deterministically tracks the current G2 selection (including a Stage-2
    ``revision_of`` re-pick). Within the approval it returns the artifact that actually
    carries ``selected_direction_id`` (not blindly ``artifacts[0]``).
    """
    for a in kb.list_task_approvals(task_id):
        if a.get("approval_type") != "wireframe":
            continue
        for art in (a.get("artifacts") or []):
            if isinstance(art, dict) and art.get("selected_direction_id"):
                return art
    return None


def _direction_already_reviewed(events, dir_id: str) -> bool:
    """True if a ``design_review_advisory`` marker for *dir_id* already exists — the
    artifact(direction)-scoped idempotency check, so a NEW/revised direction re-runs."""
    for e in events:
        if (getattr(e, "kind", "") or "") != "design_review_advisory":
            continue
        p = getattr(e, "payload", None)
        if isinstance(p, str):
            try:
                p = json.loads(p)
            except Exception:
                p = {}
        if isinstance(p, dict) and p.get("selected_direction_id") == dir_id:
            return True
    return False


def run_advisory_design_review(
    conn: sqlite3.Connection,
    task_id: str,
    epic_id: str,
    *,
    chat,
    model: str,
    lane: str = "design-advisory",
) -> "dict | None":
    """Advisory design-quality review for a UI/website task (decision-experience-first-builds-v1).

    Sources the G2 wireframe approval's approved-direction evidence for *task_id*,
    runs the text-only, injection-safe design review, and records an ADVISORY
    ``design_quality`` conformance verdict against *epic_id* (``crosscheck=False`` so it
    NEVER gates integration; surfaced in the G3 packet by ``_harvest_conformance_verdicts``).

    Returns the review dict, or ``None`` when the task has no approved design direction
    (non-UI work / nothing to review) — a cheap skip that records no verdict.

    Wired into ``run_l1_screen_for_review_task`` (review phase, non-blocking, fail-open,
    once-per-artifact) — it runs automatically for review tasks; non-UI tasks self-skip.
    Kept advisory (never gates) until calibrated on >=3 real boards.
    """
    from hermes_cli.review_loop import design_quality as _dq

    artifact = _latest_selected_direction(task_id)
    evidence = _dq.render_design_evidence(artifact)
    if not evidence:
        return None

    result = _dq.review_design_quality(evidence, chat=chat, model=model)
    # Advisory mapping: "concerns" -> "fail" (surfaced in the packet, never blocks);
    # "insufficient_evidence" -> "skip".
    verdict = {"pass": "pass", "concerns": "fail", "insufficient_evidence": "skip"}.get(
        result.get("verdict"), "skip"
    )
    record_conformance_verdict(
        conn, epic_id, "design_quality", verdict,
        lane=lane, findings=result.get("findings") or [], crosscheck=False,
    )
    return result


def _epic_learning_signals(conn: sqlite3.Connection, epic_id: str) -> list[dict]:
    """Gather WI-9 learning signals from an epic's history for the G3
    learning_delta. Conservative + crash-safe: any recorded rejection/feedback
    event becomes a correction signal; an epic that reached G3 with NO recorded
    rejection yields a clean-acceptance success signal (capture what worked)."""
    signals: list[dict] = []
    rejected = False
    try:
        events = kb.list_events(conn, epic_id)
    except Exception:
        events = []
    for e in events:
        kind = getattr(e, "kind", "")
        payload = getattr(e, "payload", None) or {}
        if kind in ("rejected", "reject_with_fixes", "changes_requested", "fix_requested"):
            rejected = True
            signals.append({
                "source_type": "rejection",
                "correction_type": payload.get("correction_type", "defect"),
                "operator_feedback": payload.get("feedback") or payload.get("comment") or "",
                "attributed_role": payload.get("role", "builder"),
            })
    if not rejected:
        signals.append({
            "source_type": "clean_acceptance",
            "correction_type": "success",
            "operator_feedback": f"epic {epic_id} reached G3 clean",
            "attributed_role": "builder",
        })
    return signals


def _emit_epic_run_metric(epic_id: str, board: Optional[str], created_at, n_stories: int) -> None:
    """WI-15 producer (best-effort, fail-safe, DEFAULT-OFF). Append one per-run
    metric row to the vault instrumentation stream
    (``metrics/autonomy/runs.jsonl`` under HERMES_LEARNING_VAULT_ROOT) when that
    env is set — the same stream conductor_vault's ``instrumentation-report``
    reads. NEVER raises into acceptance generation; the report loader tolerates a
    torn line. Emits only the defensible run-level signal at this event
    (run_id/board/stories/wall_clock); per-role defect/escape rates need other
    emit points (follow-up)."""
    try:
        root = os.environ.get("HERMES_LEARNING_VAULT_ROOT")
        if not root:
            return
        from pathlib import Path
        d = Path(root) / "metrics" / "autonomy"
        d.mkdir(parents=True, exist_ok=True)
        now = int(time.time())
        row = {"run_id": epic_id, "board": board or "", "stories": int(n_stories or 0),
               "recorded_at": now, "source": "epic_acceptance"}
        if isinstance(created_at, (int, float)) and created_at:
            row["wall_clock_seconds"] = max(0, now - int(created_at))
        with open(d / "runs.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    except Exception:
        _log.debug("WI-15 run-metric emit skipped for epic %s", epic_id, exc_info=True)


# ---------------------------------------------------------------------------
# B6: WI-QA4 conformance fix-story auto-spawn (closes R2 — no manual follow-up)
# ---------------------------------------------------------------------------

def _conformance_spawn_fix_story(
    conn: sqlite3.Connection,
    epic: sqlite3.Row,
    reason: str,
    board: "str | None",
) -> None:
    """Spawn a conformance fix story for a failed axis, bounded by WI-QA4 retry cap.

    Deduped via ``conformance_fix_story_created`` events on the epic.  After
    ``WI_QA4_MAX_CONFORMANCE_RETRIES`` spawns, emits ``conformance_escalated``
    once and takes no further action (escalation watcher surfaces it).

    :param reason: Short slug identifying the failure type (e.g. ``"security_fail"``).
    """
    epic_id = epic["id"]
    # Count prior conformance fix stories
    try:
        events = kb.list_events(conn, epic_id)
    except Exception:
        events = []
    prior_spawns = [
        e for e in events
        if (getattr(e, "kind", "") or "") == "conformance_fix_story_created"
    ]
    prior_count = len(prior_spawns)

    if prior_count >= WI_QA4_MAX_CONFORMANCE_RETRIES:
        # Escalate-once: only emit escalation if not already emitted
        already_escalated = any(
            (getattr(e, "kind", "") or "") == "conformance_escalated"
            for e in events
        )
        if not already_escalated:
            _log.warning(
                "conformance gate: epic %s hit retry bound (%d/%d) reason=%s — escalating once",
                epic_id, prior_count, WI_QA4_MAX_CONFORMANCE_RETRIES, reason,
            )
            kb._append_event(conn, epic_id, "conformance_escalated", {
                "reason": reason,
                "retry_count": prior_count,
                "max_retries": WI_QA4_MAX_CONFORMANCE_RETRIES,
            })
        else:
            _log.debug("conformance gate: epic %s already escalated, skipping", epic_id)
        return

    # Spawn the fix story
    title = f"[conformance-fix] {epic['title']}: {reason} (retry {prior_count + 1}/{WI_QA4_MAX_CONFORMANCE_RETRIES})"
    try:
        fix_id = kb.create_task(
            conn,
            title=title,
            body=(
                f"Conformance gate blocked epic `{epic_id}` — reason: `{reason}`.\n\n"
                f"Fix the issue and re-run the conformance lane.  "
                f"This is retry {prior_count + 1} of {WI_QA4_MAX_CONFORMANCE_RETRIES} "
                f"(WI-QA4 bound; escalation after {WI_QA4_MAX_CONFORMANCE_RETRIES}).\n\n"
                f"**Parent epic:** `{epic_id}` — {epic['title']}"
            ),
            workspace_kind=epic["workspace_kind"],
            workspace_path=epic["workspace_path"],
            tenant=epic["tenant"],
            board=board,
        )
        kb._append_event(conn, epic_id, "conformance_fix_story_created", {
            "fix_task_id": fix_id,
            "reason": reason,
            "retry_number": prior_count + 1,
        })
        _log.info(
            "conformance gate: spawned fix story %s for epic %s reason=%s (retry %d/%d)",
            fix_id, epic_id, reason, prior_count + 1, WI_QA4_MAX_CONFORMANCE_RETRIES,
        )
    except Exception:
        _log.exception(
            "conformance gate: failed to spawn fix story for epic %s reason=%s",
            epic_id, reason,
        )


# ---------------------------------------------------------------------------
# ADD-ON C v2 — WI-C3 L0 deterministic gate (default-off; runs before any model
# review). Pure executor lives in hermes_cli.review_loop; this is the policy:
# where it hooks, how a FAIL routes back to the builder, and the escalate-once
# bound. Touches NO binding-verdict code.
# ---------------------------------------------------------------------------

def run_l0_gate_for_review_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    board: Optional[str] = None,
    l0_cfg: Optional[dict] = None,
) -> bool:
    """Run the L0 deterministic gate for a task sitting in ``review``.

    Called by the dispatcher (the harness) **before** a review token is spent.
    The builder never self-certifies — results are captured here and written as an
    out-of-band ``l0_attestation`` event.

    :returns: ``True`` if L0 *settled* the task this tick (FAIL → routed to the
        fix-retry loop, or exhausted → held), meaning the caller must NOT spawn a
        review. ``False`` if review should proceed (L0 passed, or already passed,
        or no checks configured).
    """
    cfg = l0_cfg or {}
    checks = cfg.get("checks") or []
    if not checks:
        return False  # nothing to gate on → let review proceed

    # Dedup: if L0 already passed for this task, don't re-run subprocesses.
    try:
        events = kb.list_events(conn, task_id)
    except Exception:
        events = []
    kinds = [(getattr(e, "kind", "") or "") for e in events]
    if "l0_gate_passed" in kinds:
        return False
    # Exhausted + escalated already → hold (no re-run, no review).
    if "l0_gate_escalated" in kinds:
        return True

    task = kb.get_task(conn, task_id)
    if task is None:
        return False
    try:
        workspace = kb.resolve_workspace(task, board=board)
    except Exception:
        _log.exception("l0_gate: workspace resolve failed for %s — proceeding to review", task_id)
        return False

    # --- run the pure executor (no DB, no LLM) ---
    from hermes_cli.review_loop.l0_gate import run_l0_gate
    from hermes_cli.review_loop.attestation import record_l0_attestation
    from hermes_cli.review_loop.metrics import emit_l0_catchrate

    try:
        result = run_l0_gate(
            workspace,
            checks,
            timeout_s=int(cfg.get("timeout_s", 600)),
            log_tail_bytes=int(cfg.get("log_tail_bytes", 8192)),
        )
    except Exception:
        _log.exception("l0_gate: executor crashed for %s — failing open to review", task_id)
        return False

    # --- record evidence out-of-band (own txn) ---
    record_l0_attestation(conn, task_id, result, board=board)

    if result.passed:
        try:
            with kb.write_txn(conn):
                kb._append_event(conn, task_id, "l0_gate_passed", {
                    "duration_s": result.duration_s,
                    "checks": [c.name for c in result.checks],
                })
        except Exception:
            _log.debug("l0_gate: l0_gate_passed emit failed for %s", task_id, exc_info=True)
        emit_l0_catchrate(
            task_id, settled_at_l0=False, passed=True, board=board,
            duration_s=result.duration_s,
        )
        return False  # review proceeds

    # --- FAIL path: bounded fix-retry, escalate-once ---
    prior_fails = sum(1 for k in kinds if k == "l0_gate_failed")
    max_retries = int(cfg.get("max_retries", HERMES_L0_GATE_MAX_RETRIES))
    on_exhaust = str(cfg.get("on_exhaust", "escalate")).lower()

    try:
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, "l0_gate_failed", {
                "failed_required": result.failed_required,
                "attempt": prior_fails + 1,
                "max_retries": max_retries,
            })
            if prior_fails + 1 < max_retries:
                # Route back to the builder: review → ready (the same transition a
                # rejected review uses). The standing dispatch path re-spawns it.
                conn.execute(
                    "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                    "claim_expires = NULL, worker_pid = NULL, current_run_id = NULL "
                    "WHERE id = ? AND status = 'review'",
                    (task_id,),
                )
                kb._append_event(conn, task_id, "l0_gate_fix_requested", {
                    "failed_required": result.failed_required,
                    "attempt": prior_fails + 1,
                })
            else:
                # Retry bound hit: escalate once.
                kb._append_event(conn, task_id, "l0_gate_escalated", {
                    "failed_required": result.failed_required,
                    "attempts": prior_fails + 1,
                    "max_retries": max_retries,
                    "on_exhaust": on_exhaust,
                })
                if on_exhaust == "block":
                    conn.execute(
                        "UPDATE tasks SET status = 'blocked', claim_lock = NULL, "
                        "claim_expires = NULL, worker_pid = NULL, current_run_id = NULL "
                        "WHERE id = ? AND status = 'review'",
                        (task_id,),
                    )
    except Exception:
        _log.exception("l0_gate: FAIL routing failed for %s", task_id)

    emit_l0_catchrate(
        task_id, settled_at_l0=True, passed=False, board=board,
        failed_required=result.failed_required, duration_s=result.duration_s,
    )
    return True  # L0 settled this tick → no review token spent


# ---------------------------------------------------------------------------
# ADD-ON C v2 — WI-C4 L1 cheap screen (default-off; NON-BINDING triage that runs
# before the binding review). Records a routine/risky signal; never blocks review,
# never writes a verdict. The signal feeds the loop-state l1 slot and (Phase 6) the
# Fusion jury sizing.
# ---------------------------------------------------------------------------

def run_l1_screen_for_review_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    board: Optional[str] = None,
    l1_cfg: Optional[dict] = None,
) -> None:
    """Run the non-binding L1 cheap screen for a task in ``review``.

    Records an ``l1_screen`` event (routine|risky + escalate). **Never blocks** the
    review and **never** writes a verdict — review always proceeds regardless. Deduped:
    runs at most once per artifact (skipped if an ``l1_screen`` event already exists).
    """
    cfg = l1_cfg or {}
    try:
        events = kb.list_events(conn, task_id)
    except Exception:
        events = []
    kinds = [(getattr(e, "kind", "") or "") for e in events]

    # Advisory design-quality pass (decision-experience-first-builds-v1): runs in the review
    # phase, NON-BLOCKING and fail-open, and INDEPENDENT of the L1-screen dedup below so a
    # Stage-2 re-pick (new selected_direction_id) gets a fresh review. ARTIFACT(direction)-
    # SCOPED idempotency: records a ``design_review_advisory`` marker carrying the reviewed
    # selected_direction_id and skips only if THAT direction was already reviewed — no
    # duplicate verdicts, but a revised direction re-runs. Non-UI tasks (no selected
    # direction) self-skip. The advisory verdict records on the epic and
    # _harvest_conformance_verdicts surfaces it in the G3 packet.
    try:
        _direction = _latest_selected_direction(task_id)
        _dir_id = _direction.get("selected_direction_id") if _direction else None
        if _dir_id and not _direction_already_reviewed(events, _dir_id):
            _epic = _epic_for_story(conn, task_id)
            if _epic:
                from hermes_cli.review_loop import ninerouter as _nr

                def _design_chat(model, messages, **kw):
                    return _nr.chat(
                        model, messages,
                        base_url=cfg.get("base_url", _nr.DEFAULT_BASE_URL),
                        key_env=cfg.get("key_env", _nr.DEFAULT_KEY_ENV),
                        timeout_s=int(cfg.get("timeout_s", 45)),
                        **kw,
                    )

                # Record the marker BEFORE the verdict so a partial failure can never
                # produce a DUPLICATE verdict on retry: if the verdict write fails after
                # this, the marker already suppresses re-review (at worst one advisory
                # review is missed — acceptable for a non-blocking axis).
                with kb.write_txn(conn):
                    kb._append_event(conn, task_id, "design_review_advisory",
                                     {"selected_direction_id": _dir_id, "epic_id": _epic})
                run_advisory_design_review(
                    conn, task_id, _epic,
                    chat=_design_chat,
                    model=cfg.get("design_model", cfg.get("model", "ag/gemini-3-flash")),
                )
    except Exception:
        _log.debug("advisory design review failed-open for %s", task_id, exc_info=True)

    if "l1_screen" in kinds:
        return  # dedup: triage once per artifact

    task = kb.get_task(conn, task_id)
    if task is None:
        return

    # Build a cheap artifact view: title + body + latest completion summary.
    parts = [f"# {task.title}"]
    if getattr(task, "body", None):
        parts.append(task.body)
    for e in events:
        if (getattr(e, "kind", "") or "") == "completed":
            p = e.payload
            if isinstance(p, str):
                try:
                    p = json.loads(p)
                except Exception:
                    p = {}
            if isinstance(p, dict) and p.get("summary"):
                parts.append(f"[build summary] {p['summary']}")
    artifact = "\n\n".join(parts)

    # Deterministic risk floor: high-risk title keywords force escalate.
    title_lower = (task.title or "").lower()
    force = any(k in title_lower for k in _HIGH_RISK_TITLE_KEYWORDS)

    try:
        from hermes_cli.review_loop.l1_screen import run_l1_screen
        result = run_l1_screen(
            artifact,
            model=cfg.get("model", "ag/gemini-3-flash"),
            base_url=cfg.get("base_url", "http://127.0.0.1:20128/v1"),
            key_env=cfg.get("key_env", "NINEROUTER_KEY"),
            timeout_s=int(cfg.get("timeout_s", 45)),
            force_escalate=force,
        )
    except Exception:
        _log.exception("l1_screen: crashed for %s — recording fail-open escalate", task_id)
        from hermes_cli.review_loop.l1_screen import L1Result
        result = L1Result(risk="risky", escalate=True, findings_count=0,
                          summary="l1 screen crashed", model="?", ok=False, error="crash")

    try:
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, "l1_screen", {
                "attested_by": "l1_screen",
                "binding": False,
                "risk": result.risk,
                "escalate": result.escalate,
                "findings_count": result.findings_count,
                "summary": result.summary,
                "model": result.model,
                "model_ok": result.ok,
                "deterministic_floor": force,
            })
    except Exception:
        _log.debug("l1_screen: event emit failed for %s", task_id, exc_info=True)


def generate_epic_acceptances(
    conn: sqlite3.Connection,
    *,
    board: Optional[str] = None,
    acceptance_assignee: str = "keith",
) -> list[str]:
    """Create the G3 acceptance task for every newly-completed epic.

    Trigger: a task with ``work_item_type='epic'`` reaches ``done`` /
    ``archived`` and has no ``acceptance_task_created`` event yet.

    The acceptance task is created and immediately ``blocked`` with a
    sticky block reason, so:
      - ``recompute_ready`` never promotes it (sticky-block rule), and
      - the dispatcher never spawns it (and its assignee names a
        control-plane lane, not a Hermes profile — double protection).

    Acceptance of one epic never gates any other task: the acceptance
    task has no children.

    Returns the list of created acceptance-task ids.
    """
    created: list[str] = []
    epic_rows = conn.execute(
        "SELECT t.id, t.title, t.workspace_kind, t.workspace_path, t.tenant, t.created_at "
        "FROM tasks t JOIN task_metadata m ON m.task_id = t.id "
        "WHERE m.work_item_type = 'epic' AND t.status IN ('done', 'archived')"
    ).fetchall()
    for epic in epic_rows:
        if _has_event(conn, epic["id"], "acceptance_task_created") is not None:
            continue
        artifacts = _harvest_epic_artifacts(conn, epic["id"])
        demo_urls = sorted(
            {a["demo_url"] for a in artifacts if a.get("demo_url")}
        )
        body_lines = [
            f"Epic `{epic['id']}` — **{epic['title']}** — is built, reviewed,",
            "QA-passed, and integrated. This is the single human gate (G3)",
            "for the whole epic (decision-autonomous-three-gate-orchestration-v1).",
            "",
        ]
        if demo_urls:
            body_lines.append("**Demo:** " + ", ".join(demo_urls))
            body_lines.append("")
        body_lines.append("**Story evidence:**")
        for a in artifacts:
            detail = ", ".join(
                f"{k}={v}" for k, v in a.items() if k not in ("task_id", "title")
            )
            body_lines.append(
                f"- {a['task_id']} — {a['title']}" + (f" ({detail})" if detail else "")
            )
        # WI-9: learning_delta + save_learning on the G3 packet. OFF unless
        # HERMES_LEARNING_DELTA_ENABLED is set (consistent default-off with the
        # other Layer-3 wirings) and fail-safe (a build error leaves the packet
        # exactly as before). Enqueue-only — candidates route to the Unified
        # Learning Inbox where review promotes; never mints a verified_protective.
        learning_delta = None
        if os.environ.get("HERMES_LEARNING_DELTA_ENABLED"):
            try:
                from hermes_cli.learning_delta import build_learning_delta

                learning_delta = build_learning_delta(
                    _epic_learning_signals(conn, epic["id"])
                )
            except Exception:
                _log.debug(
                    "learning_delta build skipped for epic %s", epic["id"], exc_info=True
                )
                learning_delta = None
        if learning_delta and learning_delta.get("candidates"):
            body_lines += [
                "",
                "**Learnings to save on accept** "
                f"(save_learning={str(learning_delta['save_learning']).lower()}; "
                "enqueue-only → Unified Learning Inbox, review promotes):",
            ]
            for _c in learning_delta["candidates"]:
                _txt = (
                    _c.get("statement") or _c.get("what_worked") or _c.get("rule")
                    or _c.get("prevention_rule") or ""
                )
                body_lines.append(f"- [{_c['type']}] {_txt}")
        # ADD-ON B E2-S1/S2/S4/S5: conformance verdicts in packet + security machine gate
        conformance = _harvest_conformance_verdicts(conn, epic["id"])

        # B5: high-risk cross-check — for security-sensitive / Phase-3 epics, require
        # both a primary verdict AND an xcheck verdict from a second independent provider.
        # Both must pass; a missing xcheck counts as insufficient evidence (no G3 yet).
        _is_high_risk = _is_high_risk_epic(conn, epic["id"], epic["title"])
        _sec = conformance.get("security") if conformance else None
        _sec_verdict = (_sec or {}).get("verdict", "skip")
        if _is_high_risk and _sec_verdict not in ("fail", "skip"):
            _xcheck = (_sec or {}).get("xcheck")
            _xcheck_verdict = (_xcheck or {}).get("verdict") if _xcheck else None
            _xcheck_provider = (_xcheck or {}).get("lane_provider") if _xcheck else None
            _primary_provider = (_sec or {}).get("lane_provider")
            if _xcheck is None:
                # Cross-check not yet recorded — no G3 until second opinion arrives.
                _log.info(
                    "conformance gate: high-risk epic %s awaiting security cross-check "
                    "(min_agreement=2, single verdict insufficient)",
                    epic["id"],
                )
                continue
            if _xcheck_verdict != "pass":
                _log.warning(
                    "conformance gate: high-risk epic %s security cross-check verdict=%s "
                    "— blocking G3",
                    epic["id"], _xcheck_verdict,
                )
                kb._append_event(conn, epic["id"], "conformance_gate_block", {
                    "axis": "security", "reason": "crosscheck_fail",
                    "primary_verdict": _sec_verdict,
                    "xcheck_verdict": _xcheck_verdict,
                    "primary_provider": _primary_provider,
                    "xcheck_provider": _xcheck_provider,
                })
                _conformance_spawn_fix_story(conn, epic, "security_xcheck_fail", board)
                continue
            if _xcheck_provider and _primary_provider and _xcheck_provider == _primary_provider:
                _log.warning(
                    "conformance gate: high-risk epic %s security cross-check provider "
                    "same as primary (%r) — independence violated, blocking G3",
                    epic["id"], _primary_provider,
                )
                kb._append_event(conn, epic["id"], "conformance_gate_block", {
                    "axis": "security", "reason": "crosscheck_independence_violation",
                    "primary_provider": _primary_provider,
                    "xcheck_provider": _xcheck_provider,
                })
                continue

        # Security gate: BLOCKING — if security verdict is fail, spawn fix story
        # instead of G3 acceptance creation (WI-QA4; B6: auto-spawn closes R2).
        if _sec_verdict == "fail":
            _log.warning(
                "conformance gate: security FAIL on epic %s — spawning fix story instead of G3",
                epic["id"],
            )
            kb._append_event(conn, epic["id"], "conformance_gate_block", {
                "axis": "security", "verdict": "fail",
                "findings": (_sec or {}).get("findings", []),
            })
            _conformance_spawn_fix_story(conn, epic, "security_fail", board)
            continue

        # Advisory axes: record in body for operator visibility
        _perf = conformance.get("performance") if conformance else None
        _a11y = conformance.get("accessibility") if conformance else None
        _criteria_lines: list[str] = []
        for _axis, _cv in (("security", _sec), ("performance", _perf), ("accessibility", _a11y)):
            if _cv is None:
                _criteria_lines.append(f"- {_axis}: not yet run")
            else:
                _criteria_lines.append(
                    f"- {_axis}: {_cv.get('verdict', '?').upper()}"
                    + (" ⚠ advisory" if _axis in ("performance", "accessibility") else "")
                    + (f" ({len(_cv.get('findings', []))} findings)" if _cv.get("findings") else "")
                )
        if _criteria_lines:
            body_lines += ["", "**Conformance verdicts:**"] + _criteria_lines
        body_lines += [
            "",
            "To **accept**: `hermes kanban accept <this-task-id> <optional note>`",
            "(or approve the epic_acceptance approval in the WebUI).",
            "To **reject**: decide the approval as `rejected` with a comment —",
            "a fix story is spawned automatically and acceptance re-requested",
            "when it lands.",
        ]
        # Packet artifacts: backward-compatible — learning_delta only added when
        # WI-9 is enabled, so default packets are exactly {epic_id, stories}.
        acceptance_packet = {"epic_id": epic["id"], "stories": artifacts}
        if learning_delta is not None:
            acceptance_packet["learning_delta"] = learning_delta
        if conformance:
            acceptance_packet["conformance_verdicts"] = conformance
        # WI-C9: annotate the G3 packet with loop capacity/confidence so the operator
        # sees whether the epic was decided at full or degraded capacity. Phase-3
        # minimal signal (conformance-derived); Phase-6 Robin Fusion replaces it with
        # the real confidence object. Additive + fail-safe.
        try:
            from hermes_cli.review_loop.state import loop_capacity
            acceptance_packet["loop_capacity"] = loop_capacity(conformance)
        except Exception:
            _log.debug("WI-C9: loop_capacity annotation skipped for %s", epic["id"], exc_info=True)
        # E4-S2: Final packet sanity — no placeholders cross to Robin
        from hermes_cli.headroom_guard import assert_no_placeholders
        try:
            assert_no_placeholders(acceptance_packet, context=f"acceptance_packet:{epic['id']}")
        except Exception as _hg_err:
            _log.error("headroom_guard: placeholder in acceptance packet for %s: %s", epic["id"], _hg_err)
            # Fail-safe: strip the offending fields rather than blocking G3 creation
            for _f in ("screenshots", "functional_test_results", "prd_conformance_matrix"):
                acceptance_packet.pop(_f, None)
        acceptance_id = kb.create_task(
            conn,
            title=f"Epic acceptance: {epic['title']}",
            body="\n".join(body_lines),
            assignee=acceptance_assignee,
            created_by="autonomy",
            workspace_kind=epic["workspace_kind"] or "scratch",
            workspace_path=epic["workspace_path"],
            tenant=epic["tenant"],
            priority=10,
            initial_status="blocked",
            board=board,
        )
        now = int(time.time())
        approval_id = "apr_" + os.urandom(4).hex()
        with kb.write_txn(conn):
            # Sticky block: an explicit 'blocked' event (no 'unblocked'
            # after it) is what makes recompute_ready leave this task
            # alone — without it, a parent-free blocked task would
            # auto-recover to ready and the dispatcher would treat the
            # G3 gate as ordinary work. The reason deliberately avoids
            # R3_GATE_PHRASES: the comment-thread "UNBLOCK:" escape
            # hatch must not bypass operator acceptance.
            kb._append_event(
                conn,
                acceptance_id,
                "blocked",
                {
                    "reason": (
                        "epic-acceptance: operator acceptance required for "
                        f"epic {epic['id']} (G3 gate; accept via `hermes "
                        f"kanban accept {acceptance_id}`)"
                    )
                },
            )
            _stamp_work_item_type(conn, acceptance_id, "acceptance")
            conn.execute(
                "INSERT INTO task_approvals ("
                " approval_id, task_id, approval_type, status,"
                " artifacts_json, options_json, selected_option_id, decision,"
                " approver, decided_at, comment, unblock_target_id,"
                " unblock_behavior, created_by, created_at, updated_at"
                ") VALUES (?, ?, 'epic_acceptance', 'pending', ?, '[]', NULL,"
                " NULL, NULL, NULL, NULL, ?, 'complete', 'autonomy', ?, ?)",
                (
                    approval_id,
                    acceptance_id,
                    json.dumps(acceptance_packet, ensure_ascii=False),
                    acceptance_id,
                    now,
                    now,
                ),
            )
            kb._append_event(
                conn,
                acceptance_id,
                "acceptance_requested",
                {
                    "epic_id": epic["id"],
                    "approval_id": approval_id,
                    "demo_urls": demo_urls,
                },
            )
            kb._append_event(
                conn,
                epic["id"],
                "acceptance_task_created",
                {"acceptance_id": acceptance_id, "approval_id": approval_id},
            )
        created.append(acceptance_id)
        # WI-15: emit the per-run instrumentation row (best-effort, default-off).
        _emit_epic_run_metric(
            epic["id"], board,
            (epic["created_at"] if "created_at" in epic.keys() else None),
            len(artifacts),
        )
        _log.info(
            "epic acceptance task %s created for epic %s", acceptance_id, epic["id"]
        )
    return created


def _latest_approval(
    conn: sqlite3.Connection, task_id: str, approval_type: str
) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT approval_id, status, decision, approver, decided_at, comment "
        "FROM task_approvals WHERE task_id = ? AND approval_type = ? "
        "ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (task_id, approval_type),
    ).fetchone()


def maybe_enqueue_learning(
    artifacts: Optional[dict], *, run_id: Optional[str] = None
) -> dict:
    """save_learning bridge (WI-9/WI-10): on G3 accept, enqueue the packet's
    ``learning_delta`` candidates into the Unified Learning Inbox via the
    conductor-vault CLI (cross-repo subprocess — the packages aren't shared).

    OFF unless HERMES_LEARNING_ENQUEUE_CMD + HERMES_LEARNING_VAULT_ROOT +
    HERMES_LEARNING_LOCK are all set. Fail-safe (never breaks accept on a bridge
    error) and ENQUEUE-ONLY (the CLI lands candidates; review promotes).
    """
    import os
    import subprocess
    import tempfile

    ld = (artifacts or {}).get("learning_delta") if isinstance(artifacts, dict) else None
    if not (isinstance(ld, dict) and ld.get("save_learning") and ld.get("candidates")):
        return {"enqueued": False, "reason": "no save_learning candidates"}
    cmd = os.environ.get("HERMES_LEARNING_ENQUEUE_CMD")
    vault = os.environ.get("HERMES_LEARNING_VAULT_ROOT")
    lock = os.environ.get("HERMES_LEARNING_LOCK")
    if not (cmd and vault and lock):
        return {"enqueued": False, "reason": "bridge not configured (off)"}
    cfile = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(ld["candidates"], fh)
            cfile = fh.name
        argv = cmd.split() + [
            "--vault", vault, "learning-enqueue",
            "--lock", lock, "--candidates-file", cfile,
        ]
        if run_id:
            argv += ["--source-run-id", str(run_id)]
        subprocess.run(argv, check=True, capture_output=True, timeout=60)
        return {"enqueued": True, "count": len(ld["candidates"])}
    except Exception:
        _log.warning("save_learning enqueue failed", exc_info=True)
        return {"enqueued": False, "reason": "error"}
    finally:
        if cfile:
            try:
                os.unlink(cfile)
            except OSError:
                pass


def sweep_acceptance_tasks(
    conn: sqlite3.Connection,
    *,
    board: Optional[str] = None,
    fix_assignee: str = "builder",
) -> dict:
    """Resolve pending acceptance tasks from operator decisions.

    - ``accepted`` event (``hermes kanban accept``) or an ``approved``
      approval row → complete the acceptance task (the epic is fully
      closed; pending approval rows are marked approved for audit
      symmetry).
    - ``rejected`` / ``changes_requested`` approval → spawn ONE fix
      story (deduped via the ``fix_story_created`` event) carrying the
      rejection comment; when the fix story later completes, a fresh
      pending approval + ``acceptance_requested`` event re-arm the gate.
    """
    completed: list[str] = []
    fix_stories: list[str] = []
    rearmed: list[str] = []
    rows = conn.execute(
        "SELECT t.id, t.title, t.workspace_kind, t.workspace_path, t.tenant "
        "FROM tasks t JOIN task_metadata m ON m.task_id = t.id "
        "WHERE m.work_item_type = 'acceptance' "
        "AND t.status IN ('blocked', 'scheduled')"
    ).fetchall()
    for row in rows:
        tid = row["id"]
        approval = _latest_approval(conn, tid, "epic_acceptance")
        accepted_ev = _has_event(conn, tid, "accepted")
        approved = approval is not None and approval["status"] == "approved"
        if accepted_ev is not None or approved:
            approver = (approval["approver"] if approved else None) or "operator"
            if approval is not None and approval["status"] == "pending":
                with kb.write_txn(conn):
                    conn.execute(
                        "UPDATE task_approvals SET status='approved', "
                        "decision='approved', approver=?, decided_at=?, "
                        "updated_at=? WHERE approval_id=? AND status='pending'",
                        (
                            approver,
                            int(time.time()),
                            int(time.time()),
                            approval["approval_id"],
                        ),
                    )
            # Atomic gate release. Mirrors record_task_acceptance's one-step
            # path and — unlike complete_task — moves a gate out of
            # ``scheduled`` (the state it self-corrects into per
            # decision-kanban-approval-gates-scheduled-not-blocked-v1), so an
            # already-stranded accepted gate recovers on the next tick too.
            with kb.write_txn(conn):
                released = kb._release_epic_acceptance_gate(conn, tid, approver)
            if not released:
                # Fall back to the standard completion path for any gate the
                # release helper declined (e.g. it had open children) so the
                # behavior is never *less* than before this fix.
                kb.complete_task(
                    conn,
                    tid,
                    result=f"Epic accepted by {approver}.",
                    summary=f"Epic accepted by {approver}.",
                    board=board,
                )
            # save_learning bridge (off unless the enqueue env is configured).
            # Reads the G3 packet's learning_delta and enqueues its candidates
            # into the Unified Learning Inbox; never breaks accept on error.
            try:
                _arts_row = (
                    conn.execute(
                        "SELECT artifacts_json FROM task_approvals WHERE approval_id=?",
                        (approval["approval_id"],),
                    ).fetchone()
                    if approval is not None
                    else None
                )
                _arts = (
                    json.loads(_arts_row["artifacts_json"])
                    if _arts_row and _arts_row["artifacts_json"]
                    else {}
                )
                maybe_enqueue_learning(_arts, run_id=tid)
            except Exception:
                _log.debug("save_learning enqueue skipped for %s", tid, exc_info=True)
            completed.append(tid)
            continue
        if approval is not None and approval["status"] in (
            "rejected",
            "changes_requested",
        ):
            fix_ev = _has_event(conn, tid, "fix_story_created")
            decided_at = int(approval["decided_at"] or 0)
            fix_after_decision = (
                fix_ev is not None and int(fix_ev["created_at"] or 0) >= decided_at
            )
            if not fix_after_decision:
                reason = approval["comment"] or "(no rejection comment provided)"
                story_id = kb.create_task(
                    conn,
                    title=f"Fix epic acceptance rejection: {row['title']}",
                    body=(
                        f"Acceptance task {tid} was rejected by "
                        f"{approval['approver'] or 'operator'}.\n\n"
                        f"Rejection comment:\n{reason}\n\n"
                        "Address the findings, then complete this task with a "
                        "summary of what changed. Acceptance will be "
                        "re-requested automatically."
                    ),
                    assignee=fix_assignee,
                    created_by="autonomy",
                    workspace_kind=row["workspace_kind"] or "scratch",
                    workspace_path=row["workspace_path"],
                    tenant=row["tenant"],
                    priority=5,
                    board=board,
                )
                with kb.write_txn(conn):
                    _stamp_work_item_type(conn, story_id, "story")
                    kb._append_event(
                        conn,
                        tid,
                        "fix_story_created",
                        {
                            "story_id": story_id,
                            "approval_id": approval["approval_id"],
                        },
                    )
                fix_stories.append(story_id)
            else:
                # Fix story exists — re-arm the gate when it has landed.
                fix_payload = fix_ev["payload"]
                if isinstance(fix_payload, str):
                    try:
                        fix_payload = json.loads(fix_payload)
                    except (json.JSONDecodeError, TypeError):
                        fix_payload = {}
                story_id = (fix_payload or {}).get("story_id")
                if story_id:
                    srow = conn.execute(
                        "SELECT status, completed_at FROM tasks WHERE id = ?",
                        (story_id,),
                    ).fetchone()
                    if srow is not None and srow["status"] in ("done", "archived"):
                        now = int(time.time())
                        new_approval = "apr_" + os.urandom(4).hex()
                        with kb.write_txn(conn):
                            conn.execute(
                                "INSERT INTO task_approvals ("
                                " approval_id, task_id, approval_type, status,"
                                " artifacts_json, options_json,"
                                " selected_option_id, decision, approver,"
                                " decided_at, comment, unblock_target_id,"
                                " unblock_behavior, created_by, created_at,"
                                " updated_at"
                                ") VALUES (?, ?, 'epic_acceptance', 'pending',"
                                " ?, '[]', NULL, NULL, NULL, NULL, NULL, ?,"
                                " 'complete', 'autonomy', ?, ?)",
                                (
                                    new_approval,
                                    tid,
                                    json.dumps(
                                        {"rework_story": story_id},
                                        ensure_ascii=False,
                                    ),
                                    tid,
                                    now,
                                    now,
                                ),
                            )
                            kb._append_event(
                                conn,
                                tid,
                                "acceptance_requested",
                                {
                                    "approval_id": new_approval,
                                    "rework_story": story_id,
                                },
                            )
                        rearmed.append(tid)
    return {"completed": completed, "fix_stories": fix_stories, "rearmed": rearmed}


# ---------------------------------------------------------------------------
# Unintegrated-work sweep
# ---------------------------------------------------------------------------


def _git(args: list[str], cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603 -- fixed argv, no shell
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=15,
    )


def _latest_verdict(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Return the verdict word from the most recent ``verdict_recorded``
    event on *task_id* (lower-cased), or ``None`` if never reviewed."""
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? "
        "AND kind = 'verdict_recorded' ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if not row:
        return None
    payload = row["payload"]
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(payload, dict):
        return None
    return str(payload.get("verdict", "")).lower() or None


def _has_passing_verdict(conn: sqlite3.Connection, task_id: str) -> bool:
    """True when the task's *latest* recorded review verdict is a pass.

    Uses the latest verdict (not "any pass") so a task that passed once,
    was reworked, and then drew a CHANGES_REQUESTED does not integrate on
    the strength of the stale pass."""
    return _latest_verdict(conn, task_id) in ("pass", "pass-with-notes")


# Integration-branch push guard (WI-5 safety floor). The integrator lane is
# the ONLY lane permitted to push, and ONLY to the designated integration
# branch — never with --force / history rewrite, never to
# main/release/production. Encoded as enforceable code here so the integrator
# tool layer can refuse BEFORE running git, rather than relying solely on the
# integrator SOUL instruction text in the integrate-task body.
PROTECTED_BRANCH_PREFIXES = ("main", "master", "release", "prod", "production")


def validate_integration_push(
    target_branch: str,
    *,
    integration_branch: str,
    force: bool = False,
) -> tuple[bool, Optional[str]]:
    """Authorize an integrator push. Returns ``(ok, reason)``.

    Rules (per §0.3 never-automate + the integrator exception):
      * force-push / history rewrite is NEVER allowed;
      * the push target must be EXACTLY the designated integration branch;
      * main / master / release* / prod* / production are always refused,
        even if one were (mis)configured as the integration branch.

    The integrator tool layer MUST call this and refuse the push when ``ok``
    is False; builder lanes never push at all.
    """
    tgt = (target_branch or "").strip()
    integ = (integration_branch or "").strip()
    if force:
        return False, "force-push / history rewrite is never permitted"
    if not tgt:
        return False, "no push target branch given"
    low = tgt.lower()
    first_seg = low.split("/", 1)[0]
    for pfx in PROTECTED_BRANCH_PREFIXES:
        # Match an exact name or the first path segment (so "release/1.2" ->
        # "release"). Precise: blocks main/master/release*/prod*/production
        # without refusing unrelated feature branches like "release-notes".
        if first_seg == pfx:
            return False, f"push to protected branch {tgt!r} is never permitted"
    if not integ:
        return False, "no integration branch configured"
    if tgt != integ:
        return False, (
            f"integrator may push ONLY to the integration branch {integ!r}, "
            f"not {tgt!r}"
        )
    return True, None


def integration_push(
    workspace: str,
    *,
    integration_branch: str,
    target_branch: Optional[str] = None,
    remote: str = "origin",
    dry_run: bool = False,
) -> dict:
    """The integrator lane's ONLY sanctioned push path (WI-5 tool layer).

    Calls :func:`validate_integration_push` (``force`` is ALWAYS False — this
    tool can never force-push) and REFUSES to push unless the target is exactly
    the designated integration branch and not a protected branch. On approval it
    runs ``git -C <workspace> push <remote> <target>`` (never ``--force``).

    The integrator SOUL directs the lane to push exclusively via this tool, so
    the guard is enforced in code rather than by prose alone. Returns a result
    dict ``{"ok", "pushed", ...}``; NEVER raises on a refusal (so a caller/CLI
    can report it cleanly) and never pushes when ``ok`` is False.
    """
    import subprocess

    target = (target_branch or integration_branch or "").strip()
    ok, reason = validate_integration_push(
        target, integration_branch=integration_branch, force=False
    )
    if not ok:
        return {"ok": False, "pushed": False, "reason": reason, "target": target}
    if dry_run:
        return {"ok": True, "pushed": False, "dry_run": True, "target": target, "remote": remote}
    try:
        p = subprocess.run(  # noqa: S603 -- fixed argv, no shell, no --force
            ["git", "-C", str(workspace), "push", str(remote), target],
            capture_output=True, text=True, timeout=300,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"ok": True, "pushed": False, "target": target, "remote": remote,
                "error": f"{type(exc).__name__}: {exc}"}
    if p.returncode != 0:
        return {"ok": True, "pushed": False, "target": target, "remote": remote,
                "error": (p.stderr or p.stdout or "").strip()[-600:]}
    return {"ok": True, "pushed": True, "target": target, "remote": remote,
            "output": (p.stdout or p.stderr or "").strip()[-600:]}


def budget_pause_if_over(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    budget_cfg: Optional[dict],
    usage_class: str = "subscription_quota",
    projected_cash_usd: float = 0.0,
    board: Optional[str] = None,
) -> Optional[str]:
    """Budget Guard (WI-7) dispatch hook.

    If the run would breach the cash budget, park *task_id* behind a
    ``pause_for_approval`` gate card and return the gate id; otherwise return
    ``None`` (allow the spawn). Defaults to ``subscription_quota`` (the fleet
    norm — $0 cash) so a run is only ever paused when it is EXPLICITLY
    ``api_cash`` over cap, keeping normal dispatch unaffected.

    NOTE: the dispatcher wiring (calling this from ``dispatch_once``'s spawn
    loop, behind a default-off ``autonomy_cfg['budget']`` flag, with a per-run
    usage_class / projected-cost source) is the remaining WI-7 step — see the
    build plan. This helper is the tested enforcement primitive it will call.
    """
    from hermes_cli import budget_guard as _bg

    decision = _bg.decision_from_config(
        budget_cfg, usage_class=usage_class, projected_cash_usd=projected_cash_usd
    )
    if decision.allow:
        return None
    return kb.create_gate_task(
        conn,
        title=f"Budget pause: approval required for {task_id}",
        body=(
            f"Autonomous run for `{task_id}` was paused by Budget Guard "
            f"(WI-7).\n\nReason: {decision.reason}\n\n"
            "Approve this gate to authorize the spend, or adjust the project "
            "budget cap; the parked task then dispatches."
        ),
        created_by="budget-guard",
        children=[task_id],
        board=board,
    )


def find_unintegrated_done_tasks(
    conn: sqlite3.Connection,
    *,
    integration_branch: str,
    board: Optional[str] = None,
    integrator_assignee: str = "integrator",
    require_review: bool = True,
) -> list[str]:
    """Auto-create integrate tasks for done work that never merged.

    A done task with a recorded ``branch_name`` whose commits are not
    ancestors of *integration_branch* (checked in the task's workspace
    repo) gets one ``integrate`` task created for the integrator lane.
    Deduped via the ``integrate_task_created`` event on the source task.

    **Review gate:** when *require_review* is true (default), a done task
    is integration-eligible ONLY if its latest recorded review verdict is
    a pass. Done work with no verdict — or a CHANGES_REQUESTED/BLOCK — is
    held from integration (a one-time ``integration_held_pending_review``
    event is emitted so the morning report can surface it) rather than
    silently merging unreviewed code. This closes the gap where the
    "build → review → integrate" sequence let review be skipped entirely.
    """
    created: list[str] = []
    rows = conn.execute(
        "SELECT t.id, t.title, t.branch_name, t.workspace_kind, "
        "       t.workspace_path, t.tenant "
        "FROM tasks t LEFT JOIN task_metadata m ON m.task_id = t.id "
        "WHERE t.status = 'done' AND t.branch_name IS NOT NULL "
        "  AND t.branch_name != '' AND t.workspace_path IS NOT NULL "
        "  AND COALESCE(m.work_item_type, 'unclassified') "
        "      NOT IN ('integrate', 'acceptance', 'epic') "
        "ORDER BY t.completed_at DESC"
    ).fetchall()
    for row in rows:
        if len(created) >= MAX_INTEGRATE_TASKS_PER_TICK:
            break
        if _has_event(conn, row["id"], "integrate_task_created") is not None:
            continue
        ws = row["workspace_path"]
        if not ws or not os.path.isdir(os.path.join(ws, ".git")) and not os.path.isfile(
            os.path.join(ws, ".git")
        ):
            continue
        try:
            have_branch = _git(
                ["rev-parse", "--verify", "--quiet", row["branch_name"]], ws
            )
            have_target = _git(
                ["rev-parse", "--verify", "--quiet", integration_branch], ws
            )
            if have_branch.returncode != 0 or have_target.returncode != 0:
                continue
            ancestor = _git(
                ["merge-base", "--is-ancestor", row["branch_name"], integration_branch],
                ws,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if ancestor.returncode == 0:
            continue  # already integrated
        if ancestor.returncode != 1:
            continue  # indeterminate — don't guess

        # ── Review gate ──────────────────────────────────────────────
        # Never integrate work whose latest review verdict isn't a pass.
        # Hold it (once-emitted event → visible in the morning report)
        # instead of silently merging unreviewed / changes-requested code.
        if require_review and not _has_passing_verdict(conn, row["id"]):
            if _has_event(conn, row["id"], "integration_held_pending_review") is None:
                latest = _latest_verdict(conn, row["id"]) or "none"
                with kb.write_txn(conn):
                    kb._append_event(
                        conn,
                        row["id"],
                        "integration_held_pending_review",
                        {
                            "reason": (
                                "done with unmerged branch but latest review "
                                f"verdict is {latest!r} (not a pass); held from "
                                "integration until a signed PASS is on record"
                            ),
                            "branch": row["branch_name"],
                        },
                    )
                _log.info(
                    "integration held for %s: latest verdict=%s (need pass)",
                    row["id"],
                    latest,
                )
            continue

        integrate_id = kb.create_task(
            conn,
            title=f"Integrate: {row['title']}",
            body=(
                f"Source task {row['id']} is done but its branch "
                f"`{row['branch_name']}` has commits not on "
                f"`{integration_branch}`.\n\n"
                f"Repo: {ws}\n"
                f"1. Merge `{row['branch_name']}` into `{integration_branch}` "
                "(no force-push, no history rewrites, integration branch "
                "ONLY — see integrator SOUL).\n"
                "2. Run the project's smoke checks.\n"
                "3. Restart the tailnet demo service for this project.\n"
                "4. Complete this task with metadata: demo_url, "
                "integration_sha.\n"
                "On merge conflict or smoke failure: kanban_block with a "
                "structured reason — never guess."
            ),
            assignee=integrator_assignee,
            created_by="autonomy",
            workspace_kind="dir",
            workspace_path=ws,
            tenant=row["tenant"],
            priority=3,
            board=board,
        )
        with kb.write_txn(conn):
            _stamp_work_item_type(conn, integrate_id, "integrate")
            kb._append_event(
                conn,
                row["id"],
                "integrate_task_created",
                {
                    "integrate_id": integrate_id,
                    "integration_branch": integration_branch,
                },
            )
        created.append(integrate_id)
        _log.info(
            "integrate task %s created for unmerged done task %s (branch %s)",
            integrate_id,
            row["id"],
            row["branch_name"],
        )
    return created


# ---------------------------------------------------------------------------
# Dispatcher entry point
# ---------------------------------------------------------------------------


# --- In-loop Robin verdict courier -----------------------------------------
# Records Robin's signed review verdicts automatically so a task leaves the
# review phase without a human running record-robin-verdict.sh by hand. This is
# the PRODUCER of the ``verdict_recorded`` event that review_loop already
# consumes. Out-of-band + HMAC integrity is enforced downstream by
# record_review_verdict; this sweep only discovers WHICH signed verdicts are
# pending on Robin and feeds them in. Fail-open: an unreachable Robin just
# leaves the verdicts queued for a later tick (never crashes the tick).

_ROBIN_VERDICT_DIR = "~/.hermes/verdicts"


def _host_verdict_key_path() -> str:
    """Absolute path to the shared Robin verdict key on the HOST home.

    Worker profiles run with HOME=<host>/.hermes/profiles/<p>/home where the key
    is absent; strip that suffix so verification uses the real host key.
    """
    home = os.path.expanduser("~")
    marker = "/.hermes/profiles/"
    if marker in home:
        home = home.split(marker, 1)[0]
    return os.path.join(home, ".hermes", "credentials", "robin-verdict-key")


def _robin_list_verdict_files(robin_ssh: str, *, timeout: int = 20) -> dict:
    """Map task_id -> newest signed-verdict path on Robin (out-of-band).

    Returns {} on any ssh failure so an unreachable Robin never breaks the tick.
    """
    try:
        proc = subprocess.run(
            ["ssh", robin_ssh, "ls -t %s/*.json 2>/dev/null" % _ROBIN_VERDICT_DIR],
            capture_output=True, text=True, timeout=timeout,
        )
    except Exception:
        _log.warning("verdict_courier: cannot reach %s to list verdicts (queued)", robin_ssh)
        return {}
    newest: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        path = line.strip()
        if not path:
            continue
        base = path.rsplit("/", 1)[-1]
        tid = base.split("__", 1)[0]  # <task_id>__<commit>__<ts>.json
        if tid and tid not in newest:  # ls -t: first occurrence is newest
            newest[tid] = path
    return newest


def _verdict_already_recorded(conn: sqlite3.Connection, task_id: str, signature: str) -> bool:
    """True if a verdict_recorded event for this task already carries this verdict.

    Matches on the FULL HMAC signature (record_review_verdict stores it on the
    event). Falls back to the 16-char ``signature_prefix`` only for legacy events
    written before the full signature was stored; the prefix is never used when a
    full signature is present, so two distinct signatures sharing a 16-char prefix
    are never conflated.
    """
    sig = signature or ""
    sig16 = sig[:16]
    for row in conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'verdict_recorded'",
        (task_id,),
    ).fetchall():
        try:
            ev = json.loads(row["payload"]) or {}
        except Exception:
            continue
        stored = ev.get("signature")
        if stored:
            if stored == sig:
                return True
        elif ev.get("signature_prefix") == sig16:
            return True
    return False


def sweep_pending_robin_verdicts(
    conn: sqlite3.Connection,
    *,
    board: Optional[str] = None,
    robin_ssh: str = "robin",
    key_path: Optional[str] = None,
) -> dict:
    """Record signed Robin verdicts pending on Robin's host (in-loop courier).

    For each task on this board with a fresh signed verdict on Robin and no
    matching ``verdict_recorded`` event, fetch it out-of-band and honor it via
    :func:`record_review_verdict` (which re-verifies channel + HMAC + task match,
    then transitions the card). Idempotent (deduped by signature) and fail-open.

    Board scoping: ``conn`` is the per-board kanban DB (kanban_db.connect opens
    one DB per board; there is no cross-board ``tasks`` table), so a verdict for
    a task on another board is simply absent from this connection and is never
    recorded here. ``board`` is forwarded to record_review_verdict for the record.
    """
    pending = _robin_list_verdict_files(robin_ssh)
    if not pending:
        return {"recorded": [], "rejected": [], "seen": 0}
    # Candidates = Robin verdict task_ids present in THIS board's DB. conn is
    # board-scoped (one DB per board; tasks has no board column), so the
    # IN-filter against this connection scopes candidates to the current board
    # -- same per-board-conn model as sweep_acceptance_tasks.
    placeholders = ",".join("?" * len(pending))
    on_board = {
        r["id"]
        for r in conn.execute(
            "SELECT id FROM tasks WHERE id IN (%s)" % placeholders, tuple(pending)
        ).fetchall()
    }
    key_path = key_path or _host_verdict_key_path()
    recorded: list[str] = []
    rejected: list[str] = []
    for tid in on_board:
        tmp = None
        try:
            tmp = tempfile.mkdtemp(prefix="verdict-courier-")
            local = os.path.join(tmp, "verdict.json")
            scp = subprocess.run(
                ["scp", "-q", "%s:%s" % (robin_ssh, pending[tid]), local],
                capture_output=True, text=True, timeout=30,
            )
            if scp.returncode != 0 or not os.path.exists(local):
                _log.warning("verdict_courier: scp failed for %s (queued)", tid)
                continue
            with open(local, encoding="utf-8") as fh:
                signed = json.load(fh)
            payload = signed.get("payload") or {}
            signature = str(signed.get("signature") or "").strip()
            if not payload or not signature:
                continue
            if _verdict_already_recorded(conn, tid, signature):
                continue
            payload_json = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            res = record_review_verdict(
                conn, tid, payload_json, signature,
                fetched_via="robin-ssh", key_path=key_path, board=board,
            )
            if res.get("ok"):
                recorded.append(tid)
            else:
                rejected.append("%s: %s" % (tid, res.get("reason")))
        except Exception:
            _log.exception("verdict_courier: failed honoring verdict for %s", tid)
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)
    if recorded or rejected:
        _log.info("verdict_courier: recorded=%s rejected=%s", recorded, rejected)
    return {"recorded": recorded, "rejected": rejected, "seen": len(on_board)}


def run_autonomy_tick(
    conn: sqlite3.Connection,
    *,
    board: Optional[str] = None,
    cfg: Optional[dict] = None,
) -> dict:
    """Run the enabled autonomy sweeps for one dispatcher tick.

    *cfg* is the merged ``kanban.autonomy`` mapping for this board
    (gateway merges global + ``kanban.boards.<slug>`` overrides):

    .. code-block:: yaml

        autonomy:
          epic_acceptance:
            enabled: true
            assignee: keith
            fix_assignee: builder
          integration:
            enabled: true
            branch: integration
            assignee: integrator
          verdict_courier:
            enabled: true
            robin_ssh: robin

    Every sweep is individually guarded; a failure in one never stops
    the others or the dispatcher tick (same isolation philosophy as the
    watcher loops).
    """
    cfg = cfg or {}
    out: dict[str, Any] = {}
    ea_cfg = cfg.get("epic_acceptance") or {}
    if ea_cfg.get("enabled"):
        try:
            out["acceptances_created"] = generate_epic_acceptances(
                conn,
                board=board,
                acceptance_assignee=str(ea_cfg.get("assignee") or "keith"),
            )
        except Exception:
            _log.exception("autonomy: generate_epic_acceptances failed")
        try:
            out["acceptance_sweep"] = sweep_acceptance_tasks(
                conn,
                board=board,
                fix_assignee=str(ea_cfg.get("fix_assignee") or "builder"),
            )
        except Exception:
            _log.exception("autonomy: sweep_acceptance_tasks failed")
    integ_cfg = cfg.get("integration") or {}
    if integ_cfg.get("enabled") and integ_cfg.get("branch"):
        try:
            out["integrate_created"] = find_unintegrated_done_tasks(
                conn,
                integration_branch=str(integ_cfg["branch"]),
                board=board,
                integrator_assignee=str(integ_cfg.get("assignee") or "integrator"),
                require_review=bool(integ_cfg.get("require_review", True)),
            )
        except Exception:
            _log.exception("autonomy: find_unintegrated_done_tasks failed")
    vc_cfg = cfg.get("verdict_courier") or {}
    if vc_cfg.get("enabled"):
        try:
            out["verdict_courier"] = sweep_pending_robin_verdicts(
                conn,
                board=board,
                robin_ssh=str(vc_cfg.get("robin_ssh") or "robin"),
                key_path=vc_cfg.get("key_path") or None,
            )
        except Exception:
            _log.exception("autonomy: sweep_pending_robin_verdicts failed")

    return out
