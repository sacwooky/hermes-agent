"""Hermes Console — Portfolio Board dashboard plugin (read-only V1.0).

Mounted at ``/api/plugins/hermes-console/portfolio/`` by the dashboard
plugin system. Every route is a thin GET-only adapter over the canonical
Hermes Kanban database (``hermes_cli.kanban_db``) — Portfolio is a *view*
over the source of truth, not a shadow task system. V1.0 has no
mutation, no log, and no secret surfaces.

Identity model
--------------
Every task is referenced by a composite ``portfolio_id`` of the form
``"{board_slug}:{task_id}"``. ``task_id`` is per-board, so the
composite is the only globally-unique handle. All API payloads include
both ``portfolio_id`` and the original ``(board_slug, task_id)`` pair.

Canonical statuses
------------------
``triage, todo, scheduled, ready, running, blocked, review, done, archived``
— matches :data:`hermes_cli.kanban_db.VALID_STATUSES`. The plugin does
NOT copy the older WebUI bridge column set that omits ``scheduled`` /
``review``.

Redaction policy
----------------
* Worker log content is collapsed and never returned by default.
* ``workspace_path``, ``stored_path``, and ``raw_log_path`` are stripped
  or replaced with ``"hidden"`` in every response.
* ``run.error`` and ``run.summary`` are preview-truncated to 600 chars.
* Comment bodies are preview-truncated to 500 chars with secret-pattern
  redaction.
* Depth-limited recursion caps the payload size so a single comment with
  a 50 MB blob cannot OOM the dashboard.

V1.0 does not implement
-----------------------
drag/drop mutation, comment posting, link edits, approval decisions,
attachment upload/delete, lifecycle writes, metadata migration.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from typing import Any, Iterable, Optional

from fastapi import APIRouter, HTTPException, Query

from hermes_cli import kanban_db

log = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Mirrors :data:`hermes_cli.kanban_db.VALID_STATUSES`. The plugin enforces
# this client-side too so the dashboard never falls back to the legacy
# WebUI column set that omits ``scheduled`` / ``review``.
CANONICAL_STATUSES: list[str] = [
    "triage",
    "todo",
    "scheduled",
    "ready",
    "running",
    "blocked",
    "review",
    "done",
    "archived",
]

# Statuses shown in the Board view by default. ``archived`` is hidden
# behind the ``include_archived`` filter and the history toggle.
VISIBLE_BOARD_STATUSES: list[str] = [s for s in CANONICAL_STATUSES if s != "archived"]

# Subagent roles surfaced in the role filter. Matches the PRD's required
# coverage (Section 5) plus common operator roles. The list is
# descriptive only — we never invent a value that doesn't appear on a
# real task.
KNOWN_SUBAGENT_ROLES: list[str] = [
    "orchestrator",
    "builder",
    "reviewer",
    "qa",
    "researcher",
    "ops-watch",
    "maintainer",
    "km-agent",
]

# Agent profiles surfaced in the profile filter. Today these are
# best-effort (no metadata column yet) and return empty until the
# T9 metadata layer lands. The PRD requires the filter to exist so
# operators can reason about fleet lanes even when nothing matches.
KNOWN_AGENT_PROFILES: list[str] = [
    "jake",
    "jake-cloud",
    "morgan",
    "loki",
]

# Work item types. The metadata layer is not yet deployed, so V1.0
# returns ``unclassified`` for every task and never fabricates a more
# specific type.
KNOWN_ITEM_TYPES: list[str] = [
    "project",
    "epic",
    "feature",
    "story",
    "task",
    "unclassified",
]

# Lifecycle / readiness states. Same status as work item types — no
# metadata layer yet, so the filter exists with documented "N/A"
# behavior and the UI surfaces "Lifecycle: Not set".
KNOWN_LIFECYCLE_STATES: list[str] = [
    "intake",
    "discovery",
    "planned",
    "building",
    "mvp",
    "demo-ready",
    "public-delivery",
]

# Default page size. Backlog table virtualisation-ready; PRD notes
# observed live scale is 445 tasks / 9 boards, so 200 is a comfortable
# one-shot.
DEFAULT_LIMIT = 200
MAX_LIMIT = 2000

# Truncation limits for the sanitisation stack.
_PREVIEW_CHARS = 500          # comment / event body preview
_RUN_PREVIEW_CHARS = 600      # run summary / error preview
_SAFE_PREVIEW_CHARS = 280     # text surfaced in lists and headers
_MAX_DEPTH = 4                # recursion cap for _sanitize_payload
_MAX_LIST_ITEMS = 50          # items kept in sanitised lists
_MAX_DICT_KEYS = 80           # keys kept in sanitised dicts
_MAX_STRING_CHARS = 500       # string length cap inside payloads

# Secret patterns. Conservative — the dashboard never renders these
# in plain text. The list matches what the canonical Kanban plugin
# uses for its own safe-preview helpers.
_SECRET_TEXT_RE = re.compile(
    r"(?i)"
    r"(api[_-]?key|token|secret|password|passwd|access[_-]?key)"
    r"(\s*[:=]\s*)([^\s,;'\"\}]+)"
)
_BEARER_RE = re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+")
_PEM_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----")

# Fields the sanitiser treats as sensitive by name. Listed centrally so
# the redaction policy is auditable in one place.
SENSITIVE_PAYLOAD_KEYS: set[str] = {
    "stored_path",
    "raw_log",
    "raw_log_path",
    "raw_payload",
    "secret",
    "secrets",
    "credentials",
    "api_key",
    "access_token",
    "private_key",
}


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------

def _portfolio_id(board_slug: str, task_id: str) -> str:
    """Build the canonical composite portfolio id.

    Format is ``"{board_slug}:{task_id}"`` — both halves are guaranteed
    non-empty (validated by :func:`_check_board_slug` /
    :func:`_check_task_id`). Slug never contains ``:`` because the
    slug normaliser rejects it.
    """
    return f"{board_slug}:{task_id}"


def _check_board_slug(slug: str) -> str:
    """Validate a board slug, returning it unchanged.

    Rejects anything that could break the composite-id parse rule
    (slugs containing ``:``), or empty strings. Slug lowercasing is
    applied so the dashboard treats ``Hermes-Console`` and
    ``hermes-console`` as the same board.
    """
    if not isinstance(slug, str) or not slug:
        raise HTTPException(status_code=400, detail="board_slug is required")
    if ":" in slug or "/" in slug or "\\" in slug or ".." in slug:
        raise HTTPException(
            status_code=400, detail="board_slug contains illegal characters",
        )
    return slug.lower()


def _check_task_id(task_id: str) -> str:
    """Validate a task id, returning it unchanged.

    Rejects empty values and any string that already contains a ``:`` —
    if we let those through, parsing the composite id later becomes
    ambiguous. Existing task ids are ``t_<hex>`` so the constraint
    matches reality.
    """
    if not isinstance(task_id, str) or not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")
    if ":" in task_id:
        raise HTTPException(
            status_code=400, detail="task_id must not contain ':' (reserved by composite id)",
        )
    return task_id


def _parse_csv(value: Optional[str]) -> list[str]:
    """Parse a comma-separated query value into a list of trimmed tokens.

    Empty values and empty tokens are dropped. The function never
    raises — the per-field validators below decide what to do with
    the resulting list.
    """
    if not value:
        return []
    return [t.strip() for t in value.split(",") if t and t.strip()]


def _parse_statuses(value: Optional[str]) -> Optional[list[str]]:
    """Parse a ``statuses=`` query value, validating against the canonical set.

    Returns ``None`` when the value is empty (meaning "no status
    filter"). Raises 400 on the first unknown status so the browser
    shows a useful error instead of silently dropping the filter.
    """
    tokens = _parse_csv(value)
    if not tokens:
        return None
    for s in tokens:
        if s not in CANONICAL_STATUSES:
            raise HTTPException(
                status_code=400,
                detail=f"unknown status {s!r}; must be one of {CANONICAL_STATUSES}",
            )
    return tokens


def _parse_known(
    tokens: list[str], allowed: Iterable[str], *, label: str,
) -> list[str]:
    """Filter ``tokens`` to values that appear in ``allowed``.

    Unknown values are dropped silently for descriptive dimensions
    (tags, item_types, lifecycle) because the canonical Kanban DB
    has no metadata table yet — strict validation would 400 every
    future value and break filters that haven't been wired up.
    """
    allowed_set = {a for a in allowed}
    return [t for t in tokens if t in allowed_set]


def _resolve_limit(value: Optional[int]) -> int:
    """Clamp a ``limit`` query value to ``[1, MAX_LIMIT]``.

    Defaults to :data:`DEFAULT_LIMIT` when None. The dashboard never
    asks for a negative or zero page, and never for more than
    :data:`MAX_LIMIT` even when the underlying DB is larger.
    """
    if value is None:
        return DEFAULT_LIMIT
    if value < 1:
        return 1
    if value > MAX_LIMIT:
        return MAX_LIMIT
    return int(value)


def _resolve_offset(value: Optional[int]) -> int:
    """Clamp ``offset`` to non-negative."""
    if value is None or value < 0:
        return 0
    return int(value)


# ---------------------------------------------------------------------------
# Sanitisation stack
# ---------------------------------------------------------------------------

def _redact_string(value: str) -> str:
    """Apply the regex redaction rules to a single string.

    Order: bearer tokens, PEM blocks, generic key/value patterns. The
    replacement sentinel (``"[REDACTED]"``) is short so it doesn't
    dominate log lines.
    """
    if not value:
        return value
    value = _BEARER_RE.sub("bearer [REDACTED]", value)
    value = _PEM_RE.sub("[REDACTED PRIVATE KEY]", value)
    value = _SECRET_TEXT_RE.sub(r"\1\2[REDACTED]", value)
    return value


def _sanitize_payload(value: Any, depth: int = 0) -> Any:
    """Recursively sanitise an arbitrary JSON value.

    - Lists are truncated to :data:`_MAX_LIST_ITEMS` items.
    - Dicts are truncated to :data:`_MAX_DICT_KEYS` keys, and any key in
      :data:`SENSITIVE_PAYLOAD_KEYS` is dropped.
    - Strings are redacted (:func:`_redact_string`) and truncated to
      :data:`_MAX_STRING_CHARS`.
    - At :data:`_MAX_DEPTH` the value is replaced with a ``"…"`` marker
      so deeply-nested payloads (e.g. pasted JSON blobs) cannot blow up
      the response.

    The function never raises — a bad payload produces a safe
    placeholder, not a 500.
    """
    if depth >= _MAX_DEPTH:
        return "…"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > _MAX_STRING_CHARS:
            value = value[:_MAX_STRING_CHARS] + "…"
        return _redact_string(value)
    if isinstance(value, list):
        if len(value) > _MAX_LIST_ITEMS:
            value = value[:_MAX_LIST_ITEMS] + ["…(truncated)"]
        return [_sanitize_payload(v, depth + 1) for v in value]
    if isinstance(value, dict):
        if len(value) > _MAX_DICT_KEYS:
            kept = dict(list(value.items())[:_MAX_DICT_KEYS])
            kept["__truncated__"] = f"…({len(value) - _MAX_DICT_KEYS} more keys)"
            value = kept
        sanitized: dict[str, Any] = {}
        for k, v in value.items():
            if k in SENSITIVE_PAYLOAD_KEYS:
                continue
            sanitized[k] = _sanitize_payload(v, depth + 1)
        return sanitized
    # Unknown scalar types (e.g. Decimal): coerce to string then redact.
    return _redact_string(str(value))


def _safe_preview(value: Optional[str], *, limit: int = _SAFE_PREVIEW_CHARS) -> Optional[str]:
    """Return a redacted, length-capped preview of a free-text field.

    Used for task bodies, comment bodies, and event payloads that we
    want to surface in list views without exposing secrets or paying
    for a full markdown render.
    """
    if not value:
        return value
    if len(value) > limit:
        value = value[:limit] + "…"
    return _redact_string(value)


def _attachment_to_public(att: kanban_db.Attachment) -> dict[str, Any]:
    """Serialise a :class:`kanban_db.Attachment` for the public API.

    ``stored_path`` is the absolute on-disk path; the UI must never see
    it. The browser downloads attachments through the existing
    ``/api/plugins/kanban/tasks/{id}/attachments/{id}`` route, which
    validates containment.
    """
    return {
        "id": att.id,
        "filename": att.filename,
        "content_type": att.content_type,
        "size": int(att.size or 0),
        "uploaded_by": att.uploaded_by,
        "created_at": int(att.created_at or 0),
    }


def _run_to_public(r: kanban_db.Run) -> dict[str, Any]:
    """Serialise a :class:`kanban_db.Run` for the public API.

    ``summary`` and ``error`` are preview-truncated to keep the detail
    payload bounded. ``metadata`` (often a dict with workspace
    metadata) is sanitised through :func:`_sanitize_payload` so a
    pasted-secret inside a run's metadata blob is redacted.
    """
    return {
        "id": r.id,
        "task_id": r.task_id,
        "profile": r.profile,
        "step_key": r.step_key,
        "status": r.status,
        "outcome": r.outcome,
        "started_at": int(r.started_at) if r.started_at else None,
        "ended_at": int(r.ended_at) if r.ended_at else None,
        "last_heartbeat_at": int(r.last_heartbeat_at) if r.last_heartbeat_at else None,
        "summary_preview": _safe_preview(r.summary, limit=_RUN_PREVIEW_CHARS),
        "error_preview": _safe_preview(r.error, limit=_RUN_PREVIEW_CHARS),
        "metadata": _sanitize_payload(r.metadata) if r.metadata else None,
    }


def _comment_to_public(c: kanban_db.Comment) -> dict[str, Any]:
    """Serialise a :class:`kanban_db.Comment` for the public API.

    Bodies are preview-truncated and redacted. ``redacted=True`` is set
    so the UI can show a small badge without having to re-scan the
    body.
    """
    body = c.body or ""
    redacted = bool(
        _BEARER_RE.search(body)
        or _PEM_RE.search(body)
        or _SECRET_TEXT_RE.search(body)
    )
    preview = _safe_preview(body, limit=_PREVIEW_CHARS)
    return {
        "id": c.id,
        "task_id": c.task_id,
        "author": c.author,
        "body_preview": preview,
        "redacted": redacted,
        "created_at": int(c.created_at or 0),
    }


def _event_to_public(
    board_slug: str, e: kanban_db.Event,
) -> dict[str, Any]:
    """Serialise a :class:`kanban_db.Event` with the composite id."""
    return {
        "portfolio_id": _portfolio_id(board_slug, e.task_id),
        "board_slug": board_slug,
        "task_id": e.task_id,
        "event_id": e.id,
        "kind": e.kind,
        "payload": _sanitize_payload(e.payload),
        "created_at": int(e.created_at or 0),
        "run_id": e.run_id,
    }


def _links_for(
    conn: sqlite3.Connection, task_id: str,
) -> dict[str, list[dict[str, str]]]:
    """Return parent/child link info for a task.

    Returns dicts of ``{task_id, board_slug}`` so the UI can render
    portfolio_id without doing its own join. We never have to chase
    cross-board links (V1 limitation) so the board_slug here is the
    same as the parent task's board.
    """
    parent_rows = conn.execute(
        "SELECT parent_id FROM task_links WHERE child_id = ?", (task_id,),
    ).fetchall()
    child_rows = conn.execute(
        "SELECT child_id FROM task_links WHERE parent_id = ?", (task_id,),
    ).fetchall()
    parents = [
        {"task_id": r["parent_id"], "portfolio_id": r["parent_id"]}
        for r in parent_rows
    ]
    children = [
        {"task_id": r["child_id"], "portfolio_id": r["child_id"]}
        for r in child_rows
    ]
    return {"parents": parents, "children": children}


# ---------------------------------------------------------------------------
# Work item metadata — V1.1 reads from the task_metadata table (T01/T02).
# When no row exists for a task we fall back to the V1.0 defaults
# (work_item_type='unclassified', lifecycle_state=None,
# agent_profile=None, tags=[]), so the V1.0 contract is preserved for
# boards that haven't yet classified a card.
# ---------------------------------------------------------------------------

# V1.0 fallback values used when a task has no row in task_metadata.
# Kept as module constants so the read path is auditable in one place.
_DEFAULT_WORK_ITEM_TYPE = "unclassified"
_DEFAULT_LIFECYCLE_STATE: Optional[str] = None
_DEFAULT_AGENT_PROFILE: Optional[str] = None
_DEFAULT_TAGS: list[str] = []


def _read_task_metadata(
    conn: sqlite3.Connection, task_id: str,
) -> dict[str, Any]:
    """Return the V1.1 metadata for ``task_id`` with V1.0 defaults on miss.

    Performs a single SELECT on the caller's connection — the per-item
    portfolio list and detail endpoints already hold a connection open,
    so we reuse it rather than opening a fresh one via
    :func:`kanban_db.get_task_metadata`. The returned shape matches
    :func:`kanban_db.get_task_metadata` minus bookkeeping columns.

    Keys: ``work_item_type`` (str, default ``"unclassified"``),
    ``lifecycle_state`` (Optional[str]), ``agent_profile``
    (Optional[str]), ``tags`` (list[str]). Tolerant of a corrupt
    ``tags_json`` — falls back to ``[]`` rather than raising.
    """
    row = conn.execute(
        "SELECT work_item_type, lifecycle_state, agent_profile, tags_json "
        "FROM task_metadata WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return {
            "work_item_type": _DEFAULT_WORK_ITEM_TYPE,
            "lifecycle_state": _DEFAULT_LIFECYCLE_STATE,
            "agent_profile": _DEFAULT_AGENT_PROFILE,
            "tags": list(_DEFAULT_TAGS),
        }
    raw_tags = row["tags_json"]
    try:
        parsed_tags = json.loads(raw_tags) if raw_tags else []
    except (TypeError, ValueError):
        parsed_tags = []
    if not isinstance(parsed_tags, list):
        parsed_tags = []
    tags: list[str] = [t for t in parsed_tags if isinstance(t, str)]
    return {
        "work_item_type": row["work_item_type"] or _DEFAULT_WORK_ITEM_TYPE,
        "lifecycle_state": row["lifecycle_state"],
        "agent_profile": row["agent_profile"],
        "tags": tags,
    }


def _compute_progress(
    task: kanban_db.Task, conn: sqlite3.Connection,
) -> Optional[dict[str, Any]]:
    """Compute progress from children for a task.

    Returns ``None`` when the task has no children (no inferable
    percent) — the UI shows ``Progress: N/A`` instead of ``0%``.
    """
    child_rows = conn.execute(
        "SELECT child_id, status FROM task_links "
        "JOIN tasks ON tasks.id = task_links.child_id "
        "WHERE task_links.parent_id = ?",
        (task.id,),
    ).fetchall()
    if not child_rows:
        return None
    total = len(child_rows)
    done = sum(1 for r in child_rows if r["status"] in ("done",))
    return {
        "completed": done,
        "total": total,
        "percent": round((done / total) * 100, 1) if total else 0,
    }


def _needs_keith(task: kanban_db.Task) -> bool:
    """Best-effort signal that the task is waiting on Keith.

    V1 derives this from the title/body text (keywords like
    ``approval``, ``mockup``, ``wireframe``) plus the assignee being
    unset. We deliberately do not make this signal authoritative —
    the filter result is informational, not used to gate work.
    """
    if not task.assignee:
        # Unassigned with a keyword that suggests Keith-decision content.
        text = (task.title or "") + " " + (task.body or "")
        lowered = text.lower()
        if any(
            kw in lowered
            for kw in (
                "approval", "approve", "mockup", "wireframe",
                "needs keith", "r3", "r3-gate", "r3 gate",
            )
        ):
            return True
    return False


def _is_stale(task: kanban_db.Task) -> bool:
    """Return True when a non-terminal task has been silent too long.

    A task is "stale" if it has been in any non-terminal status for
    more than 24 h and has not been updated in that window. ``updated``
    is approximated by ``last_heartbeat_at`` for running tasks, falling
    back to ``started_at`` or ``created_at``.
    """
    if task.status in ("done", "archived"):
        return False
    now = int(time.time())
    last_touch = (
        int(task.last_heartbeat_at)
        if task.last_heartbeat_at
        else (int(task.started_at) if task.started_at else int(task.created_at or 0))
    )
    return (now - int(last_touch)) > 24 * 3600


def _item_to_public(
    board_slug: str, task: kanban_db.Task, *,
    conn: Optional[sqlite3.Connection] = None,
    latest_summary: Optional[str] = None,
) -> dict[str, Any]:
    """Serialise a :class:`kanban_db.Task` for the list endpoint.

    ``conn`` is optional — when provided, the link counts and comment
    counts are computed with a single aggregate query. When ``None``
    (e.g. on the detail endpoint) the caller is responsible for
    computing them.

    The shape matches the PRD's ``GET /portfolio/items`` payload
    (Section 6.3). All fields are explicit so the dashboard never
    has to handle ``undefined`` for missing metadata.
    """
    progress = None
    link_counts = {"parents": 0, "children": 0}
    comment_count = 0
    run_count = 0
    attachment_count = 0
    if conn is not None:
        progress = _compute_progress(task, conn)
        link_counts = _links_counts(conn, task.id)
        comment_count = _row_count(conn, "task_comments", "task_id", task.id)
        run_count = _row_count(conn, "task_runs", "task_id", task.id)
        attachment_count = _row_count(conn, "task_attachments", "task_id", task.id)
        # V1.1: read V1.1 metadata (work_item_type / lifecycle_state /
        # agent_profile / tags) from the task_metadata table, falling
        # back to V1.0 defaults when no row exists for this task.
        meta = _read_task_metadata(conn, task.id)
    else:
        meta = {
            "work_item_type": _DEFAULT_WORK_ITEM_TYPE,
            "lifecycle_state": _DEFAULT_LIFECYCLE_STATE,
            "agent_profile": _DEFAULT_AGENT_PROFILE,
            "tags": list(_DEFAULT_TAGS),
        }
    return {
        "portfolio_id": _portfolio_id(board_slug, task.id),
        "board_slug": board_slug,
        "task_id": task.id,
        "title": task.title,
        "body_preview": _safe_preview(task.body),
        "status": task.status,
        "status_group": task.status,
        "assignee": task.assignee,
        "subagent_role": task.assignee,  # subagent_role == assignee in V1
        "agent_profile": meta["agent_profile"],
        "priority": int(task.priority or 0),
        "created_by": task.created_by,
        "created_at": int(task.created_at or 0),
        "updated_at": int(task.last_heartbeat_at or task.started_at or task.created_at or 0),
        "completed_at": int(task.completed_at) if task.completed_at else None,
        "tenant": task.tenant,
        "workspace_kind": task.workspace_kind,
        "workspace_path_safe": "hidden",  # never expose raw path
        "link_counts": link_counts,
        "comment_count": comment_count,
        "run_count": run_count,
        "attachment_count": attachment_count,
        "work_item_type": meta["work_item_type"],
        "lifecycle_state": meta["lifecycle_state"],
        "tags": meta["tags"],
        "progress": progress,
        "needs_keith": _needs_keith(task),
        "stale": _is_stale(task),
        "approval_state": None,
        "source": "kanban",
        "latest_summary": _safe_preview(latest_summary, limit=_PREVIEW_CHARS),
    }


def _links_counts(conn: sqlite3.Connection, task_id: str) -> dict[str, int]:
    """Count parents and children for a single task.

    Two cheap aggregate queries; the task_links table is small
    relative to ``task_events`` and these run only for visible items.
    """
    parents = conn.execute(
        "SELECT COUNT(*) AS c FROM task_links WHERE child_id = ?", (task_id,),
    ).fetchone()["c"]
    children = conn.execute(
        "SELECT COUNT(*) AS c FROM task_links WHERE parent_id = ?", (task_id,),
    ).fetchone()["c"]
    return {"parents": int(parents), "children": int(children)}


def _row_count(
    conn: sqlite3.Connection, table: str, column: str, value: str,
) -> int:
    """Count rows in ``table`` where ``column = value``.

    Note: ``table`` and ``column`` are always hardcoded constants at
    the call sites — never user input. A future maintainer who
    changes this should keep the audit trail intact.
    """
    # Safe f-string: callers pass literal strings only. See call sites.
    row = conn.execute(
        f"SELECT COUNT(*) AS c FROM {table} WHERE {column} = ?",  # noqa: S608
        (value,),
    ).fetchone()
    return int(row["c"])


# ---------------------------------------------------------------------------
# Board discovery — multi-board aggregation
# ---------------------------------------------------------------------------

def _safe_connect(board_slug: str) -> Optional[sqlite3.Connection]:
    """Open a per-board connection, returning ``None`` on missing DB.

    Skips boards whose DB file is absent (e.g. a board directory with
    metadata but no tasks yet). The list endpoints never raise on
    missing boards; the detail endpoint does (404) so the caller can
    distinguish "not found" from "system error".
    """
    try:
        return kanban_db.connect(board=board_slug)
    except Exception as exc:
        log.warning("portfolio: failed to open board %s: %s", board_slug, exc)
        return None


def _list_boards_filtered(
    boards_arg: Optional[list[str]], *, include_archived: bool = False,
) -> list[dict[str, Any]]:
    """Apply the board filter to the canonical board inventory.

    Returns the raw board metadata dicts (so the caller can pull
    ``slug``/``name``/``archived``). When the caller passes an
    explicit list, the inventory is filtered to it; unknown slugs
    are silently dropped (a misspelled board is a UX no-op, not a
    400). The ``include_archived`` flag controls whether archived
    boards are eligible for inclusion.
    """
    inventory = kanban_db.list_boards(include_archived=include_archived)
    allowed: Optional[set[str]] = set(boards_arg) if boards_arg else None
    out: list[dict[str, Any]] = []
    for meta in inventory:
        slug = (meta.get("slug") or "").lower()
        if not slug:
            continue
        if allowed is not None and slug not in allowed:
            continue
        out.append(meta)
    return out


# ---------------------------------------------------------------------------
# Filter parsing for ``GET /portfolio/items``
# ---------------------------------------------------------------------------

def _parse_items_query(
    boards: Optional[str],
    statuses: Optional[str],
    assignees: Optional[str],
    q: Optional[str],
    agent_profiles: Optional[str],
    item_types: Optional[str],
    tags: Optional[str],
    lifecycle: Optional[str],
    include_archived: bool,
    include_done: bool,
    view: Optional[str],
    limit: Optional[int],
    offset: Optional[int],
) -> dict[str, Any]:
    """Parse + normalise the ``GET /portfolio/items`` query parameters.

    The returned dict has stable keys consumed by :func:`_query_items`.
    Validation errors raise ``HTTPException(400)`` with a clear
    message; otherwise the dashboard gets a clean filter object.

    V1.1 step 4 — the four ``task_metadata`` dimensions are parsed
    here and consumed by :func:`_task_passes_filter`. Unknown
    descriptive values (a future tag the catalogue hasn't catalogued
    yet) are dropped silently rather than 400-ing the dashboard; a
    strict 400 would break filters that haven't been wired up.
    ``tags`` keeps every token (no allowlist) because the tag set is
    open-ended by design; the other three are checked against the
    canonical KNOWN_* constant so the filter dropdown never offers
    values that can never match.
    """
    status_list = _parse_statuses(statuses)
    board_list = _parse_csv(boards) if boards else None
    if board_list:
        board_list = [_check_board_slug(b) for b in board_list]
    assignee_list = _parse_csv(assignees) if assignees else None
    # V1.1 metadata dimensions. ``tags`` accepts any token (open
    # vocabulary), the other three are filtered against the
    # KNOWN_* catalogue so unknown values don't silently match
    # nothing.
    agent_profiles_list = _parse_csv(agent_profiles) if agent_profiles else None
    if agent_profiles_list:
        agent_profiles_list = _parse_known(
            agent_profiles_list, KNOWN_AGENT_PROFILES, label="agent_profiles",
        )
    item_types_list = _parse_csv(item_types) if item_types else None
    if item_types_list:
        item_types_list = _parse_known(
            item_types_list, KNOWN_ITEM_TYPES, label="item_types",
        )
    lifecycle_list = _parse_csv(lifecycle) if lifecycle else None
    if lifecycle_list:
        lifecycle_list = _parse_known(
            lifecycle_list, KNOWN_LIFECYCLE_STATES, label="lifecycle",
        )
    tags_list = _parse_csv(tags) if tags else None
    view_norm = (view or "backlog").lower()
    if view_norm not in ("backlog", "board"):
        raise HTTPException(
            status_code=400, detail="view must be 'backlog' or 'board'",
        )
    return {
        "boards": board_list,
        "statuses": status_list,
        "assignees": assignee_list,
        "q": (q or "").strip().lower() or None,
        "agent_profiles": agent_profiles_list or None,
        "item_types": item_types_list or None,
        "tags": tags_list or None,
        "lifecycle": lifecycle_list or None,
        "include_archived": bool(include_archived),
        "include_done": bool(include_done),
        "view": view_norm,
        "limit": _resolve_limit(limit),
        "offset": _resolve_offset(offset),
    }


def _task_passes_filter(
    task: kanban_db.Task,
    flt: dict[str, Any],
    meta: Optional[dict[str, Any]] = None,
) -> bool:
    """Return True if ``task`` matches the parsed filter dict.

    Filter semantics (PRD Section 6.3 / T2 Section 6.3, V1.1 step 4):
      * AND across dimensions;
      * OR within a dimension (the parser already split CSV into
        lists, so membership check is OR);
      * empty dimension means "all";
      * ``include_done``/``include_archived`` are visibility
        controls, not statuses — they sit outside the canonical
        status set;
      * V1.1 metadata dimensions (``agent_profiles`` /
        ``item_types`` / ``tags`` / ``lifecycle``) match on the
        ``task_metadata`` row. The ``tags`` dimension uses OR
        semantics across the CSV list but AND semantics against
        a single task — i.e. the task must carry at least one
        of the requested tags. Full AND-of-tags is a follow-up.
    """
    # Visibility controls come first — they shortcut the rest.
    if task.status == "archived" and not flt["include_archived"]:
        return False
    if task.status == "done" and not flt["include_done"]:
        return False
    if flt["statuses"] and task.status not in flt["statuses"]:
        return False
    if flt["assignees"] and (task.assignee or "") not in flt["assignees"]:
        return False
    if flt["q"]:
        text = (task.title or "") + " " + (task.body or "")
        if flt["q"] not in text.lower():
            return False
    # V1.1 metadata dimensions. ``meta`` is always provided by
    # :func:`_collect_items` (which has a connection open). The
    # per-filter OR-within semantics mean: a single match in any
    # of the requested values is enough to keep the task.
    if any((flt["agent_profiles"], flt["item_types"],
            flt["tags"], flt["lifecycle"])):
        if meta is None:
            # Caller didn't pre-fetch — defensively return False
            # so a misconfigured call site can't accidentally let
            # every task through the metadata filter.
            return False
        if (flt["agent_profiles"]
                and (meta.get("agent_profile") or "") not in flt["agent_profiles"]):
            return False
        if (flt["item_types"]
                and (meta.get("work_item_type") or "") not in flt["item_types"]):
            return False
        if (flt["lifecycle"]
                and (meta.get("lifecycle_state") or "") not in flt["lifecycle"]):
            return False
        if flt["tags"]:
            task_tags = set(meta.get("tags") or [])
            if not task_tags.intersection(flt["tags"]):
                return False
    return True


def _collect_items(flt: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Walk every selected board, returning the merged, filtered item list.

    Returns a tuple of ``(items, latest_event_ids)`` where
    ``latest_event_ids`` is a ``{board_slug: last_event_id}`` map
    that the events stream uses to skip work on idle boards.

    Items are returned in priority-DESC then created-ASC order,
    matching the canonical sort in :func:`kanban_db.list_tasks`.
    """
    boards = _list_boards_filtered(
        flt["boards"], include_archived=flt["include_archived"],
    )
    items: list[dict[str, Any]] = []
    latest_event_ids: dict[str, int] = {}
    for meta in boards:
        slug = (meta.get("slug") or "").lower()
        if not slug:
            continue
        conn = _safe_connect(slug)
        if conn is None:
            continue
        try:
            rows = kanban_db.list_tasks(
                conn, include_archived=flt["include_archived"], limit=None,
            )
            last = conn.execute(
                "SELECT COALESCE(MAX(id), 0) AS c FROM task_events"
            ).fetchone()
            latest_event_ids[slug] = int(last["c"] or 0)
            summaries = kanban_db.latest_summaries(
                conn, [t.id for t in rows],
            )
            for task in rows:
                # V1.1: read metadata once per task and share it
                # between the filter and the serializer. When no
                # task_metadata row exists, the helper returns the
                # V1.0 defaults (work_item_type='unclassified',
                # lifecycle_state=None, agent_profile=None, tags=[])
                # so an unclassified task is correctly excluded by
                # any non-empty metadata filter.
                meta_for_task = _read_task_metadata(conn, task.id)
                if not _task_passes_filter(task, flt, meta_for_task):
                    continue
                items.append(_item_to_public(
                    slug, task, conn=conn,
                    latest_summary=summaries.get(task.id),
                ))
        finally:
            try:
                conn.close()
            except Exception:
                pass
    # PRD default: priority desc, then created_at asc.
    items.sort(key=lambda it: (-int(it["priority"]), int(it["created_at"])))
    return items, latest_event_ids


# ---------------------------------------------------------------------------
# Facet computation
# ---------------------------------------------------------------------------

def _facet_for(
    items: list[dict[str, Any]], key: str, *,
    allowed: Optional[Iterable[str]] = None,
    cast: Any = str,
) -> list[dict[str, Any]]:
    """Compute a ``{value, count}`` facet for a single dimension.

    Values not in ``allowed`` are dropped (this is how
    ``item_types``/``lifecycle``/``tags`` skip strings that have no
    business appearing in the filter UI). Empty / None values are
    dropped too — the dashboard's "no tag" pill is a separate
    "Untagged" toggle, not a facet entry.
    """
    counts: dict[Any, int] = {}
    for it in items:
        raw = it.get(key)
        if raw is None or raw == "" or raw == []:
            continue
        if isinstance(raw, list):
            values = raw
        else:
            values = [raw]
        for v in values:
            v = cast(v)
            if v is None or v == "":
                continue
            if allowed is not None and v not in allowed:
                continue
            counts[v] = counts.get(v, 0) + 1
    return [{"value": k, "count": counts[k]} for k in sorted(counts)]


def _facets_for(
    items: list[dict[str, Any]], latest_event_ids: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    """Compute the full ``facets`` object for the items payload.

    Boards and statuses are always returned (even when the count is
    zero) so the dropdown chrome renders a complete list. The other
    dimensions drop zero-count entries because the dashboard uses
    them as toggles, not selectors.
    """
    # Boards come from the inventory, not the items (a board with
    # only archived tasks is still a real board).
    board_meta = _list_boards_filtered(None, include_archived=True)
    board_facets = [
        {
            "value": (m.get("slug") or "").lower(),
            "count": sum(1 for it in items if it["board_slug"] == (m.get("slug") or "").lower()),
        }
        for m in board_meta
        if m.get("slug")
    ]
    # Statuses always enumerate from the canonical set, even when
    # zero items match — this matches the original implementation
    # (T6 verification) and gives the dropdown predictable chrome.
    status_counts = {s: 0 for s in CANONICAL_STATUSES}
    for it in items:
        if it["status"] in status_counts:
            status_counts[it["status"]] += 1
    status_facets = [
        {"value": s, "count": status_counts[s]} for s in CANONICAL_STATUSES
    ]
    return {
        "boards": board_facets,
        "statuses": status_facets,
        "assignees": _facet_for(items, "assignee"),
        "agent_profiles": _facet_for(
            items, "agent_profile", allowed=KNOWN_AGENT_PROFILES,
        ),
        "item_types": _facet_for(
            items, "work_item_type", allowed=KNOWN_ITEM_TYPES,
        ),
        "tags": _facet_for(items, "tags"),
        "lifecycle": _facet_for(
            items, "lifecycle_state", allowed=KNOWN_LIFECYCLE_STATES,
        ),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/summary")
def portfolio_summary() -> dict[str, Any]:
    """Return the small header payload the dashboard renders in its top bar.

    Counts are computed by walking every board's task list — a
    single ``list_tasks`` call per board keeps the latency
    predictable. ``latest_event_id_by_board`` is included so the
    client can show "X new events" badges without polling the
    full events endpoint.
    """
    boards = _list_boards_filtered(None, include_archived=True)
    total_items = 0
    active_items = 0
    blocked_items = 0
    needs_keith_items = 0
    latest_event_ids: dict[str, int] = {}
    for meta in boards:
        slug = (meta.get("slug") or "").lower()
        if not slug:
            continue
        conn = _safe_connect(slug)
        if conn is None:
            continue
        try:
            rows = kanban_db.list_tasks(conn, include_archived=True, limit=None)
            total_items += len(rows)
            for t in rows:
                if t.status in ("done", "archived"):
                    continue
                active_items += 1
                if t.status == "blocked":
                    blocked_items += 1
                if _needs_keith(t):
                    needs_keith_items += 1
            last = conn.execute(
                "SELECT COALESCE(MAX(id), 0) AS c FROM task_events"
            ).fetchone()
            latest_event_ids[slug] = int(last["c"] or 0)
        finally:
            try:
                conn.close()
            except Exception:
                pass
    return {
        "source": "hermes-kanban",
        "generated_at": int(time.time()),
        "boards": len(boards),
        "items": total_items,
        "active_items": active_items,
        "blocked_items": blocked_items,
        "needs_keith_items": needs_keith_items,
        "latest_event_id_by_board": latest_event_ids,
        "selected_scope": None,
    }


@router.get("/boards")
def portfolio_boards(
    include_archived: bool = Query(False, description="Include archived boards"),
) -> dict[str, Any]:
    """Return the board inventory with per-board task counts.

    The ``current`` field surfaces the active board slug from the
    canonical resolution chain. ``total`` is the raw task count;
    per-status counts are intentionally omitted to keep the payload
    small — the status facet is computed by ``GET /portfolio/items``.
    """
    boards = _list_boards_filtered(None, include_archived=include_archived)
    out: list[dict[str, Any]] = []
    for meta in boards:
        slug = (meta.get("slug") or "").lower()
        if not slug:
            continue
        conn = _safe_connect(slug)
        total = 0
        done = 0
        if conn is not None:
            try:
                rows = kanban_db.list_tasks(conn, include_archived=True, limit=None)
                total = len(rows)
                done = sum(1 for t in rows if t.status == "done")
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        out.append({
            "board_slug": slug,
            "name": meta.get("name") or slug,
            "description": meta.get("description"),
            "archived": bool(meta.get("archived")),
            "counts": {"done": done},
            "total": total,
            "agent_profile": None,
            "project_key": slug,
        })
    try:
        current = kanban_db.get_current_board()
    except Exception:
        current = None
    return {"boards": out, "current": current}


@router.get("/items")
def portfolio_items(
    boards: Optional[str] = Query(
        None, description="Comma-separated board slugs (default: all)"
    ),
    statuses: Optional[str] = Query(
        None, description="Comma-separated canonical statuses (OR within)"
    ),
    assignees: Optional[str] = Query(
        None, description="Comma-separated assignees / subagent roles"
    ),
    q: Optional[str] = Query(
        None, description="Substring search across title and body"
    ),
    # V1.1 step 4 — task_metadata filter dimensions. Each is a CSV
    # of values; OR within the dimension, AND across dimensions.
    agent_profiles: Optional[str] = Query(
        None, description=(
            "Comma-separated agent profiles from KNOWN_AGENT_PROFILES "
            "(OR within; AND across other dimensions)"
        ),
    ),
    item_types: Optional[str] = Query(
        None, description=(
            "Comma-separated work item types (project/epic/feature/story/"
            "task/unclassified) (OR within; AND across other dimensions)"
        ),
    ),
    tags: Optional[str] = Query(
        None, description=(
            "Comma-separated tags; matches tasks carrying any of them "
            "(OR semantics; full AND-of-tags is a follow-up)"
        ),
    ),
    lifecycle: Optional[str] = Query(
        None, description=(
            "Comma-separated lifecycle states from KNOWN_LIFECYCLE_STATES "
            "(OR within; AND across other dimensions)"
        ),
    ),
    include_archived: bool = Query(False),
    include_done: bool = Query(True),
    view: Optional[str] = Query("backlog"),
    limit: Optional[int] = Query(None),
    offset: Optional[int] = Query(None),
) -> dict[str, Any]:
    """Return the merged backlog/board item list across boards.

    ``view=backlog`` (default) returns a flat, paged list. ``view=board``
    adds a ``by_status`` map so the dashboard can render the Kanban
    columns in one round-trip — the underlying list is the same.

    V1.1 step 4 adds the four ``task_metadata`` filter dimensions
    (``agent_profiles`` / ``item_types`` / ``tags`` / ``lifecycle``).
    These are now wired up to the underlying task_metadata table —
    pre-V1.1 they were accepted but never matched because every
    item was unclassified. The dashboard's filter dropdowns for
    these dimensions are now functional: the ``/filters`` endpoint
    returns the catalogue, ``/items`` honours the query string,
    and the ``facets`` payload returns real counts.
    """
    flt = _parse_items_query(
        boards=boards, statuses=statuses, assignees=assignees, q=q,
        agent_profiles=agent_profiles, item_types=item_types,
        tags=tags, lifecycle=lifecycle,
        include_archived=include_archived, include_done=include_done,
        view=view, limit=limit, offset=offset,
    )
    items, latest_event_ids = _collect_items(flt)
    total = len(items)
    page_items = items[flt["offset"]:flt["offset"] + flt["limit"]]
    facets = _facets_for(page_items, latest_event_ids)
    payload: dict[str, Any] = {
        "items": page_items,
        "page": {
            "limit": flt["limit"],
            "offset": flt["offset"],
            "total": total,
        },
        "facets": facets,
        "view": flt["view"],
    }
    if flt["view"] == "board":
        # Enumerate columns from the canonical set (NOT a hardcoded
        # list) so the dashboard renders ``scheduled`` and ``review``
        # even when no items match.
        by_status: dict[str, list[dict[str, Any]]] = {
            s: [] for s in VISIBLE_BOARD_STATUSES
        }
        for it in page_items:
            if it["status"] in by_status:
                by_status[it["status"]].append(it)
        payload["by_status"] = by_status
    return payload


@router.get("/item/{board_slug}/{task_id}")
def portfolio_item(board_slug: str, task_id: str) -> dict[str, Any]:
    """Return the full detail payload for one task.

    Payload sections match the PRD Section 6.4 (and the Hermes
    Kanban story/detail information order from Section 1 of the
    PRD). ``security.hidden_fields`` lists the field names the
    sanitiser dropped so the UI can show a "redacted" badge.
    """
    board_slug = _check_board_slug(board_slug)
    task_id = _check_task_id(task_id)
    conn = _safe_connect(board_slug)
    if conn is None:
        raise HTTPException(status_code=404, detail="board not found")
    try:
        task = kanban_db.get_task(conn, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        comments = [
            _comment_to_public(c) for c in kanban_db.list_comments(conn, task_id)
        ]
        events = [
            _event_to_public(board_slug, e) for e in kanban_db.list_events(conn, task_id)
        ]
        attachments = [
            _attachment_to_public(a) for a in kanban_db.list_attachments(conn, task_id)
        ]
        runs = [
            _run_to_public(r) for r in kanban_db.list_runs(conn, task_id)
        ]
        links = _links_for(conn, task_id)
        # Best-effort hierarchy: walk parent_id chain to root. V1
        # does not have typed project/epic/feature metadata, so the
        # path is just the id sequence.
        hierarchy_path: list[dict[str, str]] = []
        cursor = task_id
        for _ in range(16):  # cap to prevent infinite loops on bad data
            row = conn.execute(
                "SELECT parent_id FROM task_links WHERE child_id = ? LIMIT 1",
                (cursor,),
            ).fetchone()
            if not row or not row["parent_id"]:
                break
            hierarchy_path.append({
                "task_id": row["parent_id"],
                "portfolio_id": _portfolio_id(board_slug, row["parent_id"]),
            })
            cursor = row["parent_id"]
        hierarchy_path.reverse()
        latest_summary = kanban_db.latest_summary(conn, task_id)
        item = _item_to_public(
            board_slug, task, conn=conn, latest_summary=latest_summary,
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return {
        "portfolio_id": _portfolio_id(board_slug, task_id),
        "board_slug": board_slug,
        "task_id": task_id,
        "task": {k: v for k, v in item.items() if k != "source"},
        "metadata": {
            "work_item_type": item["work_item_type"],
            "hierarchy_path": hierarchy_path,
            "lifecycle_state": item["lifecycle_state"],
            "tags": item["tags"],
            "agent_profile": item["agent_profile"],
            "progress": item["progress"],
            "needs_keith": item["needs_keith"],
            "stale": item["stale"],
        },
        "comments": comments,
        "events": events,
        "links": links,
        "runs": runs,
        "attachments": attachments,
        "approvals": [],
        "logs": {
            "available": False,
            "default_collapsed": True,
            "content_preview": None,
            "reveal_required": True,
        },
        "security": {
            "redacted": True,
            "hidden_fields": sorted(SENSITIVE_PAYLOAD_KEYS),
        },
    }


@router.get("/events")
def portfolio_events(
    board: Optional[str] = Query(
        None, description="Limit to one board slug (default: all)"
    ),
    since_id: Optional[int] = Query(
        None, ge=0, description="Return events with id > since_id"
    ),
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    """Return recent ``task_events`` rows, optionally filtered by board.

    Used by the dashboard to detect new events between polls.
    The Portfolio event stream normalises the row with the
    composite ``portfolio_id`` so the browser can route it back to
    its open detail panel without doing a second join.
    """
    since = int(since_id) if since_id is not None else 0
    boards = _list_boards_filtered(
        [board] if board else None, include_archived=True,
    )
    events: list[dict[str, Any]] = []
    for meta in boards:
        slug = (meta.get("slug") or "").lower()
        if not slug:
            continue
        conn = _safe_connect(slug)
        if conn is None:
            continue
        try:
            rows = conn.execute(
                "SELECT * FROM task_events WHERE id > ? ORDER BY id ASC LIMIT ?",
                (since, int(limit)),
            ).fetchall()
        finally:
            try:
                conn.close()
            except Exception:
                pass
        for r in rows:
            try:
                payload = json.loads(r["payload"]) if r["payload"] else None
            except Exception:
                payload = None
            events.append({
                "portfolio_id": _portfolio_id(slug, r["task_id"]),
                "board_slug": slug,
                "task_id": r["task_id"],
                "event_id": int(r["id"]),
                "kind": r["kind"],
                "payload": _sanitize_payload(payload),
                "created_at": int(r["created_at"] or 0),
                "run_id": int(r["run_id"]) if r["run_id"] is not None else None,
            })
    events.sort(key=lambda e: e["event_id"])
    if len(events) > limit:
        events = events[-limit:]
    return {"events": events, "count": len(events)}


@router.get("/filters")
def portfolio_filters() -> dict[str, Any]:
    """Return the filter dimension catalogue for the filter-bar UI.

    Each entry names a filter dimension, the value type, and the
    canonical allowed-values set. The dashboard uses this to
    render the dropdown chrome and to validate user-typed values
    before they hit ``/items``.

    No aggregate counts are returned here — the items endpoint
    already attaches those as ``facets``. This endpoint is the
    *static* shape contract.
    """
    return {
        "source": "hermes-kanban",
        "generated_at": int(time.time()),
        "dimensions": [
            {
                "key": "boards",
                "label": "Board",
                "type": "csv",
                "values": "facet",
                "description": "Filter to one or more boards",
            },
            {
                "key": "statuses",
                "label": "Status",
                "type": "csv",
                "values": CANONICAL_STATUSES,
                "description": "Canonical Hermes statuses (OR within)",
            },
            {
                "key": "assignees",
                "label": "Subagent role",
                "type": "csv",
                "values": "facet",
                "description": "Builder, reviewer, etc.",
            },
            {
                "key": "q",
                "label": "Keyword",
                "type": "string",
                "values": None,
                "description": "Substring search across title and body",
            },
            # V1.1 step 4 — the four task_metadata filter
            # dimensions. ``tags`` has an open vocabulary so values
            # is a facet (the dashboard surfaces whatever the
            # current item set has tagged); the other three are
            # closed vocabularies keyed by the canonical KNOWN_*
            # constant in plugin_api.py.
            {
                "key": "agent_profiles",
                "label": "Agent profile",
                "type": "csv",
                "values": KNOWN_AGENT_PROFILES,
                "description": (
                    "Agent profile lane (jake / jake-cloud / morgan / "
                    "loki); OR within, AND across other dimensions"
                ),
            },
            {
                "key": "item_types",
                "label": "Item type",
                "type": "csv",
                "values": KNOWN_ITEM_TYPES,
                "description": (
                    "Work item type (project / epic / feature / story / "
                    "task / unclassified); OR within, AND across other "
                    "dimensions"
                ),
            },
            {
                "key": "tags",
                "label": "Tags",
                "type": "csv",
                "values": "facet",
                "description": (
                    "Open-vocabulary tag list; OR-within (task matches "
                    "if it carries any of the requested tags), AND "
                    "across other dimensions"
                ),
            },
            {
                "key": "lifecycle",
                "label": "Lifecycle",
                "type": "csv",
                "values": KNOWN_LIFECYCLE_STATES,
                "description": (
                    "Lifecycle / readiness state; OR within, AND across "
                    "other dimensions"
                ),
            },
            {
                "key": "include_archived",
                "label": "Include archived",
                "type": "boolean",
                "values": [True, False],
                "description": "Toggle the archived lane on/off",
            },
            {
                "key": "include_done",
                "label": "Include done",
                "type": "boolean",
                "values": [True, False],
                "description": "Toggle the done lane on/off",
            },
        ],
        "facets": {
            "boards": "from inventory",
            "statuses": CANONICAL_STATUSES,
            "assignees": "from items",
            "agent_profiles": KNOWN_AGENT_PROFILES,
            "item_types": KNOWN_ITEM_TYPES,
            "tags": "from items",
            "lifecycle": KNOWN_LIFECYCLE_STATES,
        },
        "canonical_statuses": CANONICAL_STATUSES,
        "visible_statuses": VISIBLE_BOARD_STATUSES,
        "subagent_roles": KNOWN_SUBAGENT_ROLES,
        "agent_profiles": KNOWN_AGENT_PROFILES,
        "work_item_types": KNOWN_ITEM_TYPES,
        "lifecycle_states": KNOWN_LIFECYCLE_STATES,
        "limits": {"default": DEFAULT_LIMIT, "max": MAX_LIMIT},
    }


# ---------------------------------------------------------------------------
# V1.0 mutation guard
# ---------------------------------------------------------------------------
#
# FastAPI would 404 any non-GET route, but we want a clean 405 with a
# pointer to the V1.1 work. The /405 sentinel covers ``POST`` /
# ``PATCH`` / ``DELETE`` against the read-only surface so the dashboard
# doesn't show a confusing 404 when an operator clicks a disabled
# action button.

@router.api_route(
    "/{path:path}",
    methods=["POST", "PUT", "PATCH", "DELETE"],
)
def _v1_0_mutation_disabled(path: str) -> None:
    """Return 405 for any mutation verb against the read-only surface.

    The V1.0 Portfolio API is intentionally GET-only. V1.1 will add
    the ``PATCH /item/{board}/{task_id}``,
    ``POST /item/{board}/{task_id}/comments``, and link edit routes
    listed in the T2 design. Until then, every mutation is refused
    with a single, consistent error so the dashboard can render
    "Coming in V1.1" without per-call branching.
    """
    raise HTTPException(
        status_code=405,
        detail=(
            "Portfolio Board V1.0 is read-only; mutation endpoints land in "
            "V1.1 (see Hermes Console T2 §17)."
        ),
    )
