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
        artifacts.append(entry)
    return artifacts


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
        "SELECT t.id, t.title, t.workspace_kind, t.workspace_path, t.tenant "
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
        body_lines += [
            "",
            "To **accept**: `hermes kanban accept <this-task-id> <optional note>`",
            "(or approve the epic_acceptance approval in the WebUI).",
            "To **reject**: decide the approval as `rejected` with a comment —",
            "a fix story is spawned automatically and acceptance re-requested",
            "when it lands.",
        ]
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
                    json.dumps(
                        {"epic_id": epic["id"], "stories": artifacts},
                        ensure_ascii=False,
                    ),
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
