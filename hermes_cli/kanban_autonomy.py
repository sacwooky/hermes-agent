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
import sqlite3
import subprocess
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

    # Normalize Robin's verdict vocabulary to the two lifecycle outcomes.
    # send-review.sh emits BUILD_READY / PASS / PASS_WITH_NOTES /
    # CHANGES_REQUESTED / BLOCK / REVISE (any case); map to pass|block.
    raw_verdict = str(payload.get("verdict", "")).strip().lower().replace("-", "_")
    PASS_SET = {"pass", "build_ready", "pass_with_notes", "approve", "approved", "ok"}
    BLOCK_SET = {"block", "blocked", "changes_requested", "revise", "reject",
                 "rejected", "do_not_build", "fail"}
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

    :param axis: ``"security"`` | ``"perf"`` | ``"a11y"``
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
    }
    kind = kind_map.get(axis)
    if not kind:
        raise ValueError(f"Unknown conformance axis: {axis!r}")
    if crosscheck:
        kind = f"{kind}_xcheck"

    # B4: author-aware independence gate.
    # Primary verdicts (crosscheck=False): lane-provider must != author-provider.
    # Cross-check verdicts (crosscheck=True): lane-provider must != primary verdict's
    #   lane-provider (independence from the primary, not from the author — the cross-
    #   check deliberately lets the "other" provider challenge the primary's conclusion).
    author_provider = _get_epic_author_provider(conn, epic_id)
    lane_provider = _normalize_provider(lane)
    if not crosscheck:
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
        "WHERE m.work_item_type = 'acceptance' AND t.status = 'blocked'"
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
    return out
