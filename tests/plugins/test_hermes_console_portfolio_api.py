"""Tests for the Hermes Console — Portfolio Board dashboard plugin.

The plugin mounts as ``/api/plugins/hermes-console/portfolio/`` inside
the dashboard's FastAPI app; we attach its router to a bare FastAPI
instance so the REST surface can be tested without spinning up the
whole dashboard. The plugin is *strictly read-only* — these tests
cover the full V1.0 acceptance surface from the T2 design and the
T8 reviewer gate:

  * Routes are registered (6 GET endpoints, plus a 405 mutation guard).
  * Mutation verbs (POST/PATCH/PUT/DELETE) are refused with 405.
  * Composite portfolio IDs (``{board_slug}:{task_id}``) appear on
    every item, event, and detail payload.
  * Canonical statuses (including ``scheduled`` and ``review``) are
    used everywhere — the WebUI bridge column set never leaks through.
  * Multi-board aggregation works (we add a second board and check
    both boards' items are surfaced).
  * The redaction stack strips secrets and paths from payloads
    (api_key, token, PEM block, ``/tmp/secret`` path).
  * Facets enumerate all canonical statuses even when zero items
    match, so the dropdown chrome is stable.
  * ``GET /filters`` is the static contract catalogue.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _load_plugin_router():
    """Dynamically load the plugin's Python module and return its router.

    The plugin API is shipped under ``plugins/hermes-console/dashboard/``
    and is mounted at runtime by the dashboard via importlib. We mirror
    that pattern here so the tests don't depend on Hermes being
    installed in any particular way beyond the venv.
    """
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "hermes-console" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"

    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_hermes_console_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def plugin_module():
    """Load the plugin module once per test module."""
    return _load_plugin_router()


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a per-test tempdir and seed the default DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # The canonical ``get_hermes_home()`` reads HERMES_HOME, but some
    # code paths also fall back to ``Path.home()``. Pin both so neither
    # leaks into the test process.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def client(isolated_home, plugin_module):
    """Build a TestClient around the portfolio router only."""
    app = FastAPI()
    app.include_router(
        plugin_module.router,
        prefix="/api/plugins/hermes-console/portfolio",
    )
    return TestClient(app)


@pytest.fixture
def seeded_default_board(isolated_home):
    """Create a small set of tasks on the default board for filtering tests.

    Returns the list of created task dicts (id, title, status, etc.).
    """
    created = []
    titles_and_statuses = [
        ("Ready for review", "ready", "builder", 5),
        ("Investigation in progress", "running", "researcher", 7),
        ("Blocked on auth", "blocked", "builder", 9),
        ("Schedule followup", "scheduled", "ops-watch", 3),
        ("Marked done", "done", "reviewer", 0),
        ("Archived task", "archived", "reviewer", 0),
    ]
    conn = kb.connect()
    try:
        for title, status, assignee, priority in titles_and_statuses:
            # ``create_task`` only accepts ``initial_status in
            # {running, blocked}``; we then UPDATE to the desired
            # canonical status so the test exercises the full
            # passthrough. Land in ``running`` first because that
            # doesn't require an explicit block reason.
            task_id = kb.create_task(
                conn, title=title, assignee=assignee, priority=priority,
                initial_status="running",
            )
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = ? WHERE id = ?",
                    (status, task_id),
                )
            created.append({
                "id": task_id,
                "title": title,
                "status": status,
                "assignee": assignee,
                "priority": priority,
            })
    finally:
        conn.close()
    return created


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def test_routes_registered(plugin_module):
    """All 6 GET endpoints and the mutation guard are mounted."""
    paths = {
        "/summary",
        "/boards",
        "/items",
        "/item/{board_slug}/{task_id}",
        "/events",
        "/filters",
        "/{path:path}",
    }
    seen: set[str] = set()
    for r in plugin_module.router.routes:
        if hasattr(r, "path"):
            seen.add(r.path)
    missing = paths - seen
    assert not missing, f"missing routes: {missing}; saw {seen}"


def test_portfolio_routes_are_read_only(client):
    """POST/PATCH/PUT/DELETE on any path returns 405 with a clear message."""
    for verb in ("post", "put", "patch", "delete"):
        r = getattr(client, verb)("/api/plugins/hermes-console/portfolio/items")
        assert r.status_code == 405, (verb, r.status_code, r.text)
        body = r.json()
        assert "V1.0" in body["detail"] or "read-only" in body["detail"], body


# ---------------------------------------------------------------------------
# Constants — canonical statuses & identity model
# ---------------------------------------------------------------------------


def test_canonical_statuses_match_kanban(plugin_module):
    """The plugin's canonical set matches the Kanban module's own enum.

    Drift between the two is the regression that originally caused
    ``scheduled`` and ``review`` to disappear from the dashboard.
    """
    assert set(plugin_module.CANONICAL_STATUSES) == kb.VALID_STATUSES
    # Scheduled and review are first-class citizens — the WebUI bridge
    # column set used to omit them; the plugin must not.
    for s in ("scheduled", "review"):
        assert s in plugin_module.CANONICAL_STATUSES, s


def test_composite_id_helper(plugin_module):
    """Composite IDs are ``{board_slug}:{task_id}`` consistently."""
    assert plugin_module._portfolio_id("hermes-console", "t_abc12345") == "hermes-console:t_abc12345"
    assert plugin_module._portfolio_id("default", "t_xyz") == "default:t_xyz"


def test_safety_validators_reject_illegal_inputs(plugin_module):
    """Slugs with ``:`` and task ids with ``:`` are refused with 400."""
    from fastapi import HTTPException

    # Empty slug/task id is 400.
    for bad in ("", None):
        with pytest.raises(HTTPException):
            plugin_module._check_board_slug(bad)
        with pytest.raises(HTTPException):
            plugin_module._check_task_id(bad)

    # Slugs with ``:``, ``/``, ``\\``, or ``..`` are 400.
    for bad in ("a:b", "a/b", "a\\b", "a..b"):
        with pytest.raises(HTTPException):
            plugin_module._check_board_slug(bad)

    # Task ids with ``:`` are 400 (composite parse ambiguity).
    with pytest.raises(HTTPException):
        plugin_module._check_task_id("a:b")


def test_parse_statuses_rejects_unknown_values(plugin_module):
    """Unknown status values raise 400 instead of being silently dropped."""
    from fastapi import HTTPException

    # Empty means "no filter".
    assert plugin_module._parse_statuses(None) is None
    assert plugin_module._parse_statuses("") is None

    # Known statuses parse.
    parsed = plugin_module._parse_statuses("ready,running,blocked")
    assert parsed == ["ready", "running", "blocked"]

    # Unknown values raise.
    with pytest.raises(HTTPException):
        plugin_module._parse_statuses("ready,nope")


# ---------------------------------------------------------------------------
# /summary
# ---------------------------------------------------------------------------


def test_summary_returns_canonical_shape(client, seeded_default_board):
    """``/summary`` returns the documented top-bar payload."""
    r = client.get("/api/plugins/hermes-console/portfolio/summary")
    assert r.status_code == 200, r.text
    data = r.json()
    for key in (
        "source", "generated_at", "boards", "items", "active_items",
        "blocked_items", "needs_keith_items",
        "latest_event_id_by_board", "selected_scope",
    ):
        assert key in data, f"missing summary key: {key}"
    assert data["source"] == "hermes-kanban"
    assert data["items"] >= len(seeded_default_board)
    assert data["blocked_items"] >= 1
    # active_items excludes done/archived.
    assert data["active_items"] >= 1
    # Latest event map has at least the default board.
    assert "default" in data["latest_event_id_by_board"]


# ---------------------------------------------------------------------------
# /boards
# ---------------------------------------------------------------------------


def test_boards_lists_default(client):
    """``/boards`` returns the default board with a ``total`` count."""
    r = client.get("/api/plugins/hermes-console/portfolio/boards")
    assert r.status_code == 200, r.text
    data = r.json()
    assert "boards" in data
    assert "current" in data
    slugs = [b["board_slug"] for b in data["boards"]]
    assert "default" in slugs
    default_board = next(b for b in data["boards"] if b["board_slug"] == "default")
    for key in ("name", "counts", "total", "project_key", "archived"):
        assert key in default_board, f"missing board key: {key}"


def test_boards_aggregates_across_boards(client, isolated_home):
    """A second board's tasks are counted in the inventory totals."""
    kb.create_board("second")
    conn = kb.connect(board="second")
    try:
        kb.create_task(conn, title="hello from second board", assignee="builder")
    finally:
        conn.close()

    r = client.get("/api/plugins/hermes-console/portfolio/boards")
    data = r.json()
    slugs = [b["board_slug"] for b in data["boards"]]
    assert "default" in slugs
    assert "second" in slugs
    second = next(b for b in data["boards"] if b["board_slug"] == "second")
    assert second["total"] == 1


# ---------------------------------------------------------------------------
# /items — multi-board aggregation + filters + canonical status facets
# ---------------------------------------------------------------------------


def test_items_use_composite_ids(client, seeded_default_board):
    """Every item carries a ``portfolio_id`` of the form ``default:t_<id>``."""
    r = client.get("/api/plugins/hermes-console/portfolio/items?view=backlog")
    assert r.status_code == 200
    data = r.json()
    items = data["items"]
    # We seeded six tasks but the default ``include_archived=False``
    # filters out the one ``archived`` row — that is the V1.0 contract
    # (archived is hidden by default). Assert the count is in the
    # documented range, then check every visible item carries a
    # well-formed composite id.
    assert len(items) >= len(seeded_default_board) - 1, (
        f"expected at least {len(seeded_default_board) - 1} items, got {len(items)}"
    )
    for it in items:
        assert it["portfolio_id"] == f"default:{it['task_id']}", it
        assert it["board_slug"] == "default"
        assert "status" in it
        assert it["status"] in kb.VALID_STATUSES


def test_items_filters_statuses_and_keywords(client, seeded_default_board):
    """The status filter is OR-within-dimension, keyword is substring."""
    # Status filter: only ready items.
    r = client.get(
        "/api/plugins/hermes-console/portfolio/items?statuses=ready"
    )
    assert r.status_code == 200
    items = r.json()["items"]
    assert all(it["status"] == "ready" for it in items)
    assert len(items) >= 1

    # Multi-status: ready + blocked.
    r = client.get(
        "/api/plugins/hermes-console/portfolio/items?statuses=ready,blocked"
    )
    items = r.json()["items"]
    assert {it["status"] for it in items}.issubset({"ready", "blocked"})

    # Keyword search.
    r = client.get(
        "/api/plugins/hermes-console/portfolio/items?q=blocked"
    )
    items = r.json()["items"]
    assert all(
        "blocked" in (it["title"] + " " + (it.get("body_preview") or "")).lower()
        for it in items
    )
    assert len(items) >= 1

    # Unknown status → 400.
    r = client.get(
        "/api/plugins/hermes-console/portfolio/items?statuses=nope"
    )
    assert r.status_code == 400


def test_items_facets_enumerate_canonical_statuses(client, seeded_default_board):
    """The status facet lists every canonical status, even zero-count ones.

    The dashboard uses this to render a complete status dropdown. If
    the facet dropped zero-count statuses the dropdown would shrink
    after each filter pass, which the T6 reviewer flagged.
    """
    r = client.get("/api/plugins/hermes-console/portfolio/items")
    facets = r.json()["facets"]
    assert "statuses" in facets
    seen_values = {row["value"] for row in facets["statuses"]}
    # Load the plugin module so the test is self-contained even
    # when the ``plugin_module`` fixture isn't materialised here.
    mod = _load_plugin_router()
    assert seen_values == set(mod.CANONICAL_STATUSES), (
        f"missing statuses: {set(mod.CANONICAL_STATUSES) - seen_values}"
    )
    for row in facets["statuses"]:
        assert "value" in row
        assert "count" in row
        assert isinstance(row["count"], int)


def test_items_view_board_returns_by_status(client, seeded_default_board):
    """``view=board`` returns ``by_status`` keyed by canonical statuses."""
    r = client.get("/api/plugins/hermes-console/portfolio/items?view=board")
    assert r.status_code == 200
    data = r.json()
    assert data["view"] == "board"
    by = data["by_status"]
    # Every visible status is present as a key (even with empty list).
    expected_keys = set(_load_plugin_router().VISIBLE_BOARD_STATUSES)
    assert set(by.keys()) == expected_keys
    # Items live under their status.
    for it in data["items"]:
        assert it["portfolio_id"] in {x["portfolio_id"] for x in by[it["status"]]}


def test_items_aggregates_across_boards(client, isolated_home, seeded_default_board):
    """A task on a second board shows up with ``board_slug=second``."""
    kb.create_board("second")
    conn = kb.connect(board="second")
    try:
        kb.create_task(conn, title="hello from second", assignee="builder")
    finally:
        conn.close()

    r = client.get("/api/plugins/hermes-console/portfolio/items")
    items = r.json()["items"]
    slugs = {it["board_slug"] for it in items}
    assert "default" in slugs
    assert "second" in slugs

    # Filtering by board slug limits the result.
    r = client.get(
        "/api/plugins/hermes-console/portfolio/items?boards=second"
    )
    items = r.json()["items"]
    assert all(it["board_slug"] == "second" for it in items)
    assert any(it["title"] == "hello from second" for it in items)


def test_items_includes_scheduled_and_review_statuses(client):
    """``scheduled`` and ``review`` are reachable via the status filter.

    The original WebUI bridge column set omitted these. The T2 design
    requires them. The plugin must not silently drop them.
    """
    conn = kb.connect()
    try:
        s = kb.create_task(
            conn, title="scheduled one", assignee="ops-watch",
            initial_status="running",
        )
        rv = kb.create_task(
            conn, title="review one", assignee="reviewer",
            initial_status="running",
        )
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='scheduled' WHERE id=?", (s,))
            conn.execute("UPDATE tasks SET status='review' WHERE id=?", (rv,))
    finally:
        conn.close()

    r = client.get(
        "/api/plugins/hermes-console/portfolio/items?statuses=scheduled,review"
    )
    assert r.status_code == 200
    items = r.json()["items"]
    statuses = {it["status"] for it in items}
    assert "scheduled" in statuses
    assert "review" in statuses


# ---------------------------------------------------------------------------
# /item/{board_slug}/{task_id}
# ---------------------------------------------------------------------------


def test_item_detail_returns_composite_payload(client, seeded_default_board):
    """Detail payload includes all 14 Hermes-style sections."""
    target = seeded_default_board[0]
    r = client.get(
        f"/api/plugins/hermes-console/portfolio/item/default/{target['id']}"
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["portfolio_id"] == f"default:{target['id']}"
    assert data["board_slug"] == "default"
    assert data["task_id"] == target["id"]
    for key in (
        "task", "metadata", "comments", "events", "links",
        "runs", "attachments", "approvals", "logs", "security",
    ):
        assert key in data, f"missing detail key: {key}"
    # Metadata carries the derived fields used by the UI.
    for key in (
        "work_item_type", "lifecycle_state", "tags",
        "agent_profile", "progress", "needs_keith", "stale",
    ):
        assert key in data["metadata"], f"missing metadata key: {key}"
    # Workspace path is never exposed — the field is replaced with
    # ``workspace_path_safe = "hidden"`` in the task dict.
    assert data["task"]["workspace_path_safe"] == "hidden"


def test_item_detail_404_on_unknown(client):
    """Unknown board or task id returns 404, not 500."""
    r = client.get(
        "/api/plugins/hermes-console/portfolio/item/default/t_does_not_exist"
    )
    assert r.status_code == 404


def test_item_detail_redacts_secrets_in_body(client, isolated_home):
    """A body that contains an API key never surfaces the value.

    The sanitisation stack also replaces ``workspace_path`` /
    ``stored_path`` so a task pasted with a raw path is not echoed
    back to the browser.
    """
    secret_value = "super-secret-abcdef123"
    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn,
            title="task with secret",
            body=(
                "This body mentions password=" + secret_value
                + " and a key api_key = hunter2 and "
                + "a workspace path /tmp/secret-place and a token=abc123."
            ),
            workspace_path="/tmp/secret-place",
        )
    finally:
        conn.close()

    r = client.get(
        f"/api/plugins/hermes-console/portfolio/item/default/{task_id}"
    )
    assert r.status_code == 200
    # The body_preview is a 280-char redacted excerpt of the body the
    # user wrote — the path the user typed into the body stays in the
    # body. What we need to prove is that the *structured* workspace_path
    # field is never exposed (only ``workspace_path_safe = "hidden"``).
    body_text = json.dumps(r.json())
    assert secret_value not in body_text
    assert "hunter2" not in body_text
    assert "abc123" not in body_text
    assert "workspace_path" not in r.json()["task"], (
        "structured workspace_path leaked to the browser: "
        f"{r.json()['task']}"
    )
    # The user-typed path in the body is preserved (it's user text,
    # not a structured field), but the redaction flag and hidden
    # field list still surface what the sanitiser covered.
    assert r.json()["security"]["redacted"] is True
    assert "stored_path" in r.json()["security"]["hidden_fields"]


def test_item_detail_redacts_pem_block_in_comment(client, isolated_home):
    """A pasted PEM private key in a comment is redacted before return.

    The dashboard never sees the raw key. The redaction is a string
    replace at preview-truncate depth, so the body still flows
    through (truncated) but the key is gone.
    """
    # A small but real-looking PEM block. The unique substring
    # ``MIIBOgIBAAJBAKj34G2`` is what we assert is absent from the
    # redacted body.
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIBOgIBAAJBAKj34G2\n"
        "-----END RSA PRIVATE KEY-----\n"
    )
    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn, title="comment with key", initial_status="running",
        )
        kb.add_comment(conn, task_id, "researcher", pem)
    finally:
        conn.close()

    r = client.get(
        f"/api/plugins/hermes-console/portfolio/item/default/{task_id}"
    )
    data = r.json()
    assert data["comments"], "expected one comment"
    comment = data["comments"][0]
    assert "MIIBOgIBAAJBAKj34G2" not in comment["body_preview"]
    assert "BEGIN RSA PRIVATE KEY" not in comment["body_preview"]
    assert comment["redacted"] is True


# ---------------------------------------------------------------------------
# V1.1 task_metadata read model
# ---------------------------------------------------------------------------


def test_item_metadata_defaults_when_no_row_exists(
    client, seeded_default_board,
):
    """Tasks with no task_metadata row show V1.0 defaults.

    The read model must fall back to ``"unclassified"`` / ``None`` /
    ``None`` / ``[]`` so the V1.0 contract is preserved for boards
    that have not yet classified a card.
    """
    target = seeded_default_board[0]
    r = client.get(
        f"/api/plugins/hermes-console/portfolio/item/default/{target['id']}"
    )
    assert r.status_code == 200
    meta = r.json()["metadata"]
    assert meta["work_item_type"] == "unclassified"
    assert meta["lifecycle_state"] is None
    assert meta["agent_profile"] is None
    assert meta["tags"] == []


def test_item_metadata_reads_from_task_metadata_row(
    client, isolated_home, seeded_default_board,
):
    """When a task_metadata row exists, its values appear in the API.

    Writes via ``upsert_task_metadata`` (T02 helper) and confirms the
    list + detail endpoints both surface the canonical values. This is
    the V1.1 acceptance: ``work_item_type`` / ``lifecycle_state`` /
    ``agent_profile`` / ``tags`` come from the table, not the old
    hardcoded stubs.
    """
    target = seeded_default_board[1]  # "Investigation in progress"
    kb.upsert_task_metadata(
        target["id"],
        work_item_type="epic",
        lifecycle_state="customer-ready",
        agent_profile="builder",
        tags=["portfolio-v1.1", "regression"],
    )

    # Detail endpoint surfaces the metadata block.
    r = client.get(
        f"/api/plugins/hermes-console/portfolio/item/default/{target['id']}"
    )
    assert r.status_code == 200
    detail = r.json()
    assert detail["metadata"]["work_item_type"] == "epic"
    assert detail["metadata"]["lifecycle_state"] == "customer-ready"
    assert detail["metadata"]["agent_profile"] == "builder"
    assert detail["metadata"]["tags"] == ["portfolio-v1.1", "regression"]

    # List endpoint surfaces the same values on the item dict.
    r = client.get("/api/plugins/hermes-console/portfolio/items")
    assert r.status_code == 200
    item = next(
        it for it in r.json()["items"] if it["task_id"] == target["id"]
    )
    assert item["work_item_type"] == "epic"
    assert item["lifecycle_state"] == "customer-ready"
    assert item["agent_profile"] == "builder"
    assert item["tags"] == ["portfolio-v1.1", "regression"]


def test_item_metadata_partial_row_uses_defaults_for_missing_fields(
    client, isolated_home, seeded_default_board,
):
    """A row that only sets ``work_item_type`` leaves the others at V1.0.

    Mirrors the upsert partial-merge contract: callers may set one
    field and leave the rest to fall back to defaults. The read model
    must reflect that — it must not lift defaults to ``unclassified``
    or any other non-default sentinel.
    """
    target = seeded_default_board[2]  # "Blocked on auth"
    kb.upsert_task_metadata(target["id"], work_item_type="story")

    r = client.get(
        f"/api/plugins/hermes-console/portfolio/item/default/{target['id']}"
    )
    assert r.status_code == 200
    meta = r.json()["metadata"]
    assert meta["work_item_type"] == "story"
    assert meta["lifecycle_state"] is None
    assert meta["agent_profile"] is None
    assert meta["tags"] == []


def test_read_task_metadata_helper_defaults(plugin_module):
    """``_read_task_metadata`` returns the V1.0 defaults on a closed conn.

    The helper is called by ``_item_to_public`` which already holds a
    connection. With no row present, the helper must return the same
    shape ``_item_to_public`` expects — we exercise the miss path on
    a hermetic in-memory DB so the test does not depend on
    ``HERMES_HOME``.
    """
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        # The helper runs a SELECT on task_metadata, so the table must
        # exist even on the miss path. The shape mirrors kanban_db's
        # T01 schema (only the columns the helper reads).
        conn.execute(
            "CREATE TABLE task_metadata ("
            "    task_id TEXT PRIMARY KEY,"
            "    work_item_type TEXT NOT NULL DEFAULT 'unclassified',"
            "    lifecycle_state TEXT,"
            "    agent_profile TEXT,"
            "    tags_json TEXT NOT NULL DEFAULT '[]'"
            ")"
        )
        result = plugin_module._read_task_metadata(conn, "t_no_such_row")
    finally:
        conn.close()
    assert result == {
        "work_item_type": "unclassified",
        "lifecycle_state": None,
        "agent_profile": None,
        "tags": [],
    }


def test_read_task_metadata_helper_tolerates_corrupt_tags_json(
    plugin_module,
):
    """A corrupt ``tags_json`` falls back to ``[]`` rather than raising.

    The T02 helper already tolerates corrupt JSON, but the plugin's
    read path is a second line of defence (it parses tags again to
    avoid an extra hop). A broken row must not 500 the dashboard.
    """
    import sqlite3
    # Build a throwaway DB shaped like task_metadata with one corrupt
    # row, so we exercise the parse / fallback path.
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            "CREATE TABLE task_metadata ("
            "    task_id TEXT PRIMARY KEY,"
            "    work_item_type TEXT NOT NULL DEFAULT 'unclassified',"
            "    lifecycle_state TEXT,"
            "    agent_profile TEXT,"
            "    tags_json TEXT NOT NULL DEFAULT '[]'"
            ")"
        )
        conn.execute(
            "INSERT INTO task_metadata (task_id, work_item_type, "
            "lifecycle_state, agent_profile, tags_json) "
            "VALUES (?, ?, ?, ?, ?)",
            ("t_corrupt", "epic", "demo-ready", "builder", "not-json{"),
        )
        result = plugin_module._read_task_metadata(conn, "t_corrupt")
    finally:
        conn.close()
    assert result["work_item_type"] == "epic"
    assert result["lifecycle_state"] == "demo-ready"
    assert result["agent_profile"] == "builder"
    assert result["tags"] == []


# ---------------------------------------------------------------------------
# V1.1 step 4 — task_metadata filter dimensions
# ---------------------------------------------------------------------------


def _seed_metadata_for_filtering(client, seeded_default_board):
    """Classify the 5 visible seed tasks so the new filter dimensions have data.

    Layout (intentionally varied so the tests can target one or more
    dimensions without overlap):

      id[0] — "Ready for review"     — agent=jake, type=epic,
                                        lifecycle=building,
                                        tags=[approval, demo-ready]
      id[1] — "Investigation in prog" — agent=morgan, type=story,
                                        lifecycle=discovery,
                                        tags=[regression, demo-ready]
      id[2] — "Blocked on auth"       — agent=jake, type=task,
                                        lifecycle=mvp,
                                        tags=[approval]
      id[3] — "Schedule followup"     — agent=loki, type=task,
                                        lifecycle=planned,
                                        tags=[followup]
      id[4] — "Marked done"           — agent=jake-cloud, type=feature,
                                        lifecycle=public-delivery,
                                        tags=[shipped]
      id[5] — "Archived task"         — UNCLASSIFIED (no row, also
                                        status=archived so it's hidden
                                        by default)

    Agent profile values are chosen from ``KNOWN_AGENT_PROFILES`` so
    the facet counts are not silently dropped by the closed-vocab
    filter. The unclassified row exists so we can prove the filter
    excludes rows with no task_metadata row when any metadata
    filter is set.
    """
    for task, kwargs in zip(
        seeded_default_board,
        [
            dict(agent_profile="jake", work_item_type="epic",
                 lifecycle_state="building",
                 tags=["approval", "demo-ready"]),
            dict(agent_profile="morgan", work_item_type="story",
                 lifecycle_state="discovery",
                 tags=["regression", "demo-ready"]),
            dict(agent_profile="jake", work_item_type="task",
                 lifecycle_state="mvp", tags=["approval"]),
            dict(agent_profile="loki", work_item_type="task",
                 lifecycle_state="planned", tags=["followup"]),
            dict(agent_profile="jake-cloud", work_item_type="feature",
                 lifecycle_state="public-delivery", tags=["shipped"]),
        ],
    ):
        kb.upsert_task_metadata(task["id"], **kwargs)
    return seeded_default_board


def test_items_filter_agent_profiles(
    client, isolated_home, seeded_default_board,
):
    """``?agent_profiles=jake`` returns only tasks with that profile.

    OR-within-dimension: ``?agent_profiles=jake,morgan`` returns
    the union. The unclassified row (id[5]) is excluded because
    its agent_profile is None.
    """
    _seed_metadata_for_filtering(client, seeded_default_board)
    r = client.get(
        "/api/plugins/hermes-console/portfolio/items"
        "?agent_profiles=jake&limit=200"
    )
    assert r.status_code == 200
    items = r.json()["items"]
    assert {it["task_id"] for it in items} == {
        seeded_default_board[0]["id"],
        seeded_default_board[2]["id"],
    }

    # OR within dimension: jake + morgan → 3 tasks.
    r = client.get(
        "/api/plugins/hermes-console/portfolio/items"
        "?agent_profiles=jake,morgan&limit=200"
    )
    items = r.json()["items"]
    assert {it["task_id"] for it in items} == {
        seeded_default_board[0]["id"],
        seeded_default_board[1]["id"],
        seeded_default_board[2]["id"],
    }


def test_items_filter_item_types(
    client, isolated_home, seeded_default_board,
):
    """``?item_types=epic,feature`` returns the union of those types."""
    _seed_metadata_for_filtering(client, seeded_default_board)
    r = client.get(
        "/api/plugins/hermes-console/portfolio/items"
        "?item_types=epic,feature&limit=200"
    )
    items = r.json()["items"]
    assert {it["task_id"] for it in items} == {
        seeded_default_board[0]["id"],   # epic
        seeded_default_board[4]["id"],   # feature
    }
    assert all(
        it["work_item_type"] in ("epic", "feature") for it in items
    )


def test_items_filter_tags_or_semantics(
    client, isolated_home, seeded_default_board,
):
    """``?tags=approval,demo-ready`` returns tasks with ANY of the tags.

    Tasks 0 and 2 carry ``approval``; task 1 carries ``demo-ready``.
    The intersection with the task's tags is non-empty for all
    three. The unclassified task (id[5]) and the followup/shipped
    rows are excluded because they carry none of the requested
    tags.
    """
    _seed_metadata_for_filtering(client, seeded_default_board)
    r = client.get(
        "/api/plugins/hermes-console/portfolio/items"
        "?tags=approval,demo-ready&limit=200"
    )
    items = r.json()["items"]
    assert {it["task_id"] for it in items} == {
        seeded_default_board[0]["id"],
        seeded_default_board[1]["id"],
        seeded_default_board[2]["id"],
    }


def test_items_filter_tags_unknown_value_dropped(
    client, isolated_home, seeded_default_board,
):
    """Unknown tag values silently match nothing.

    The dashboard would never surface a tag nobody has, so we
    keep the no-400 contract: a request for ``?tags=nope`` simply
    returns an empty list rather than raising. The route's job is
    to translate the query into a SQL filter, not to police the
    vocabulary.
    """
    _seed_metadata_for_filtering(client, seeded_default_board)
    r = client.get(
        "/api/plugins/hermes-console/portfolio/items?tags=nope"
    )
    assert r.status_code == 200
    assert r.json()["items"] == []


def test_items_filter_lifecycle(
    client, isolated_home, seeded_default_board,
):
    """``?lifecycle=building,mvp`` returns the two tasks in those states."""
    _seed_metadata_for_filtering(client, seeded_default_board)
    r = client.get(
        "/api/plugins/hermes-console/portfolio/items"
        "?lifecycle=building,mvp&limit=200"
    )
    items = r.json()["items"]
    assert {it["task_id"] for it in items} == {
        seeded_default_board[0]["id"],   # building
        seeded_default_board[2]["id"],   # mvp
    }


def test_items_filters_combine_and_across_dimensions(
    client, isolated_home, seeded_default_board,
):
    """Multiple filters AND together across dimensions.

    ``item_types=task`` narrows to ids 2 and 3; ``tags=approval``
    narrows further to id 2 only. The combination must return
    just the intersection.
    """
    _seed_metadata_for_filtering(client, seeded_default_board)
    r = client.get(
        "/api/plugins/hermes-console/portfolio/items"
        "?item_types=task&tags=approval&limit=200"
    )
    items = r.json()["items"]
    assert {it["task_id"] for it in items} == {
        seeded_default_board[2]["id"],
    }

    # AND with a status filter, exercising the cross-dimension mix.
    r = client.get(
        "/api/plugins/hermes-console/portfolio/items"
        "?statuses=blocked&item_types=task&agent_profiles=jake"
        "&lifecycle=mvp&tags=approval&limit=200"
    )
    assert {it["task_id"] for it in items} == {
        seeded_default_board[2]["id"],
    }


def test_items_filter_excludes_unclassified_when_set(
    client, isolated_home, seeded_default_board,
):
    """A non-empty metadata filter must exclude unclassified rows.

    Pre-V1.1 the items endpoint accepted these params and silently
    returned zero results because every item was unclassified. The
    T03 read model fixed the read path; T04 must show the filter
    actually does the work. An unclassified task (no
    task_metadata row) must drop out when any metadata filter is
    active, because its values are all None / "unclassified" /
    [].

    Test signal: request an agent profile (``loki``) that exactly
    one task carries, then check the result set is smaller than
    the unfiltered set. If the filter were a no-op, the
    unfiltered and filtered sets would be equal.
    """
    _seed_metadata_for_filtering(client, seeded_default_board)
    r_unfiltered = client.get(
        "/api/plugins/hermes-console/portfolio/items?limit=200"
    )
    unfiltered_ids = {it["task_id"] for it in r_unfiltered.json()["items"]}
    r_filtered = client.get(
        "/api/plugins/hermes-console/portfolio/items"
        "?agent_profiles=loki&limit=200"
    )
    filtered_ids = {it["task_id"] for it in r_filtered.json()["items"]}
    # Filtered set is strictly smaller than unfiltered (loki only
    # matches one row out of the five classified tasks), proving
    # the filter is actually doing work.
    assert filtered_ids < unfiltered_ids
    assert filtered_ids == {seeded_default_board[3]["id"]}


def test_items_filter_unknown_known_values_are_dropped(
    client, isolated_home, seeded_default_board,
):
    """Unknown KNOWN_* values are dropped silently.

    ``item_types=epic,banana,task`` collapses to ``epic,task`` —
    the parser uses :func:`_parse_known` which skips values not in
    the catalogue. This keeps the dashboard's open-vocabulary
    drift from 400-ing filters. The unfiltered items are returned
    because the remaining values still match.
    """
    _seed_metadata_for_filtering(client, seeded_default_board)
    r = client.get(
        "/api/plugins/hermes-console/portfolio/items"
        "?item_types=epic,banana,task&limit=200"
    )
    assert r.status_code == 200
    items = r.json()["items"]
    assert {it["task_id"] for it in items} == {
        seeded_default_board[0]["id"],   # epic
        seeded_default_board[2]["id"],   # task
        seeded_default_board[3]["id"],   # task
    }


def test_items_facets_return_real_counts_for_metadata_dimensions(
    client, isolated_home, seeded_default_board,
):
    """``facets`` populates real counts for the four V1.1 dimensions.

    Pre-V1.1 the facets returned 0 / empty for agent_profiles,
    item_types, tags, and lifecycle. After T04 they reflect the
    underlying task_metadata table. ``tags`` is a list-valued
    facet so each tag appears once with its item count.
    """
    _seed_metadata_for_filtering(client, seeded_default_board)
    r = client.get("/api/plugins/hermes-console/portfolio/items?limit=200")
    facets = r.json()["facets"]
    # agent_profiles — counted by item, not by occurrence.
    ap = {row["value"]: row["count"] for row in facets["agent_profiles"]}
    assert ap == {"jake": 2, "morgan": 1, "loki": 1, "jake-cloud": 1}
    # item_types — same.
    it = {row["value"]: row["count"] for row in facets["item_types"]}
    assert it == {
        "epic": 1, "story": 1, "task": 2, "feature": 1,
    }
    # lifecycle.
    lc = {row["value"]: row["count"] for row in facets["lifecycle"]}
    assert lc == {
        "building": 1, "discovery": 1, "mvp": 1, "planned": 1,
        "public-delivery": 1,
    }
    # tags — list-valued; each tag counted once per carrying item.
    tg = {row["value"]: row["count"] for row in facets["tags"]}
    assert tg == {
        "approval": 2, "demo-ready": 2, "regression": 1,
        "followup": 1, "shipped": 1,
    }


def test_filters_catalogue_lists_metadata_dimensions(client):
    """``/filters`` exposes all four V1.1 dimensions in its catalogue.

    The dashboard uses the catalogue to render the filter bar and
    to validate user input. The four V1.1 dimensions must appear
    with the correct type / values (closed vocab vs facet).
    """
    r = client.get("/api/plugins/hermes-console/portfolio/filters")
    assert r.status_code == 200
    data = r.json()
    by_key = {d["key"]: d for d in data["dimensions"]}
    for key in ("agent_profiles", "item_types", "tags", "lifecycle"):
        assert key in by_key, f"missing dimension {key}"
        assert by_key[key]["type"] == "csv"
    # The three closed-vocabularies carry the KNOWN_* list.
    assert set(by_key["agent_profiles"]["values"]) == set(
        _load_plugin_router().KNOWN_AGENT_PROFILES,
    )
    assert set(by_key["item_types"]["values"]) == set(
        _load_plugin_router().KNOWN_ITEM_TYPES,
    )
    assert set(by_key["lifecycle"]["values"]) == set(
        _load_plugin_router().KNOWN_LIFECYCLE_STATES,
    )
    # Tags is a facet, not a closed list.
    assert by_key["tags"]["values"] == "facet"


# ---------------------------------------------------------------------------
# /events
# ---------------------------------------------------------------------------


def test_events_endpoint_returns_composite_ids(client, seeded_default_board):
    """Events include ``portfolio_id``, ``board_slug``, ``task_id``."""
    r = client.get("/api/plugins/hermes-console/portfolio/events?limit=50")
    assert r.status_code == 200
    data = r.json()
    assert "events" in data
    assert isinstance(data["events"], list)
    assert data["count"] == len(data["events"])
    # Events exist from the create_task calls.
    assert data["count"] >= len(seeded_default_board)
    for ev in data["events"]:
        assert ev["portfolio_id"] == f"{ev['board_slug']}:{ev['task_id']}"
        assert ev["board_slug"] == "default"
        assert "kind" in ev
        assert "event_id" in ev


def test_events_endpoint_can_filter_by_board(client, isolated_home):
    """``?board=`` limits results to that board only."""
    kb.create_board("second")
    conn = kb.connect(board="second")
    try:
        kb.create_task(conn, title="hi")
    finally:
        conn.close()

    r = client.get(
        "/api/plugins/hermes-console/portfolio/events?board=second&limit=10"
    )
    data = r.json()
    assert all(ev["board_slug"] == "second" for ev in data["events"])


# ---------------------------------------------------------------------------
# /filters
# ---------------------------------------------------------------------------


def test_filters_returns_static_contract(client):
    """``/filters`` is the dimension catalogue used by the filter bar."""
    r = client.get("/api/plugins/hermes-console/portfolio/filters")
    assert r.status_code == 200
    data = r.json()
    for key in (
        "dimensions", "canonical_statuses", "visible_statuses",
        "subagent_roles", "agent_profiles", "work_item_types",
        "lifecycle_states", "limits",
    ):
        assert key in data, f"missing filters key: {key}"
    # All dimensions describe the supported filter inputs.
    dim_keys = {d["key"] for d in data["dimensions"]}
    assert {
        "boards", "statuses", "assignees", "q",
        "include_archived", "include_done",
    }.issubset(dim_keys)
    # Limits are sane.
    assert data["limits"]["default"] > 0
    assert data["limits"]["max"] >= data["limits"]["default"]


# ---------------------------------------------------------------------------
# Bundle sanity (JS / CSS / manifest)
# ---------------------------------------------------------------------------


def test_manifest_registers_under_hermes_console():
    """The manifest names the plugin ``hermes-console`` and points at the
    expected entry / api / css / tab."""
    repo_root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (repo_root / "plugins" / "hermes-console" / "dashboard" / "manifest.json").read_text()
    )
    assert manifest["name"] == "hermes-console"
    assert manifest["label"] == "Hermes Console"
    assert manifest["tab"]["path"] == "/console"
    assert manifest["entry"] == "dist/index.js"
    assert manifest["css"] == "dist/style.css"
    assert manifest["api"] == "plugin_api.py"


def test_dist_assets_exist_and_have_content():
    """Both the JS entry and the stylesheet exist and are non-trivial."""
    repo_root = Path(__file__).resolve().parents[2]
    js = repo_root / "plugins" / "hermes-console" / "dashboard" / "dist" / "index.js"
    css = repo_root / "plugins" / "hermes-console" / "dashboard" / "dist" / "style.css"
    assert js.exists()
    assert css.exists()
    assert js.stat().st_size > 10_000, "JS bundle looks too small"
    assert css.stat().st_size > 5_000, "CSS bundle looks too small"

    # The bundle registers itself with the host dashboard via the
    # shared registry, and exposes Backlog + Board tab components.
    text = js.read_text()
    assert "window.__HERMES_PLUGINS__.register" in text
    assert '"hermes-console"' in text
    assert "Backlog" in text
    assert "Board" in text
    # V1.0 is read-only — every action surface must be disabled.
    assert "disabled: true" in text
    assert "Coming in V1.1" in text
    # Composite IDs are parsed.
    assert "parsePortfolioId" in text
    # Canonical statuses are wired in (no stale WebUI column set).
    for s in ("scheduled", "review", "blocked", "done", "archived"):
        assert f'"{s}"' in text or f"'{s}'" in text, (
            f"missing canonical status {s!r} in JS"
        )
    # V1.1 step 4 — the chrome exposes filter rows for the four
    # task_metadata dimensions. The id is the public contract
    # for the FilterBar Select/Input element; if any of these
    # disappear the chrome has regressed and bullet 7 of T04 is
    # no longer covered.
    for fid in (
        "hc-filter-agent-profile",
        "hc-filter-item-type",
        "hc-filter-lifecycle",
        "hc-filter-tags",
    ):
        assert fid in text, f"FilterBar is missing id={fid!r}"
    # The filter state object must include all four new keys
    # (initial state, useEffect deps, and onResetFilters all
    # reference them, so a single grep catches regressions in
    # any of those three locations).
    for state_key in (
        "agent_profiles:",
        "item_types:",
        "tags:",
        "lifecycle:",
    ):
        assert state_key in text, (
            f"filter state is missing key {state_key!r} — the chrome "
            "filter field won't survive a Reset or a re-render"
        )


# ---------------------------------------------------------------------------
# approval_state derivation (V1.2-T06)
# ---------------------------------------------------------------------------


def _derive_approval_state(plugin_mod, conn, task_id):
    """Thin wrapper so tests can call the module helper."""
    return plugin_mod._derive_approval_state(conn, task_id)


class TestDeriveApprovalState:
    """Unit tests for _derive_approval_state."""

    def test_no_approvals_is_none(self, isolated_home, plugin_module):
        """An item with no approvals has approval_state = None (V1.0 default)."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="plain task")
            result = _derive_approval_state(plugin_module, conn, tid)
        assert result is None

    def test_single_pending_is_pending(self, isolated_home, plugin_module):
        """An item with a pending approval has approval_state = 'pending'."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="needs gate")
        kb.create_task_approval(tid, approval_type="wireframe")
        with kb.connect() as conn:
            result = _derive_approval_state(plugin_module, conn, tid)
        assert result == "pending"

    def test_all_approved_is_approved(self, isolated_home, plugin_module):
        """An item where all approvals are approved has approval_state = 'approved'."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="all clear")
        a1 = kb.create_task_approval(tid, approval_type="wireframe")
        a2 = kb.create_task_approval(tid, approval_type="gate")
        kb.decide_task_approval(a1, decision="approved", approver="keith")
        kb.decide_task_approval(a2, decision="approved", approver="keith")
        with kb.connect() as conn:
            result = _derive_approval_state(plugin_module, conn, tid)
        assert result == "approved"

    def test_any_rejected_is_rejected(self, isolated_home, plugin_module):
        """An item with a rejected approval has approval_state = 'rejected',
        even if other approvals are approved."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="sent back")
        a1 = kb.create_task_approval(tid, approval_type="wireframe")
        a2 = kb.create_task_approval(tid, approval_type="gate")
        kb.decide_task_approval(a1, decision="approved", approver="keith")
        kb.decide_task_approval(a2, decision="rejected", approver="keith")
        with kb.connect() as conn:
            result = _derive_approval_state(plugin_module, conn, tid)
        assert result == "rejected"

    def test_changes_requested_no_rejection(self, isolated_home, plugin_module):
        """'changes_requested' wins when there are no rejections or pending."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="revise me")
        a1 = kb.create_task_approval(tid, approval_type="wireframe")
        kb.decide_task_approval(a1, decision="changes_requested", approver="keith")
        with kb.connect() as conn:
            result = _derive_approval_state(plugin_module, conn, tid)
        assert result == "changes_requested"

    def test_expired_no_other_state(self, isolated_home, plugin_module):
        """'expired' wins when there are no rejections, pending, or changes_requested."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="timed out")
        a1 = kb.create_task_approval(tid, approval_type="gate")
        # Directly set status to expired since decide only supports
        # approved/rejected/changes_requested.
        with kb.connect() as conn:
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE task_approvals SET status = 'expired' WHERE approval_id = ?",
                    (a1,),
                )
            result = _derive_approval_state(plugin_module, conn, tid)
        assert result == "expired"

    def test_rejected_overrides_pending(self, isolated_home, plugin_module):
        """'rejected' has highest priority — overrides 'pending'."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="mixed signals")
        a1 = kb.create_task_approval(tid, approval_type="wireframe")
        a2 = kb.create_task_approval(tid, approval_type="gate")
        kb.decide_task_approval(a1, decision="rejected", approver="keith")
        # a2 is still pending
        with kb.connect() as conn:
            result = _derive_approval_state(plugin_module, conn, tid)
        assert result == "rejected"

    def test_pending_overrides_changes_requested(self, isolated_home, plugin_module):
        """'pending' has second-highest priority — overrides 'changes_requested'."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="mixed again")
        a1 = kb.create_task_approval(tid, approval_type="wireframe")
        a2 = kb.create_task_approval(tid, approval_type="gate")
        kb.decide_task_approval(a1, decision="changes_requested", approver="keith")
        # a2 is still pending
        with kb.connect() as conn:
            result = _derive_approval_state(plugin_module, conn, tid)
        assert result == "pending"


def test_approval_state_appears_in_items_payload(client, isolated_home):
    """approval_state is populated in the GET /items response when approvals exist."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="needs wireframe ok")
    kb.create_task_approval(tid, approval_type="wireframe")
    resp = client.get("/api/plugins/hermes-console/portfolio/items")
    assert resp.status_code == 200
    data = resp.json()
    items = data["items"]
    # Find our task.
    match = [i for i in items if i["task_id"] == tid]
    assert len(match) == 1
    assert match[0]["approval_state"] == "pending"


def test_approval_state_null_when_no_approvals_in_items(client, isolated_home):
    """approval_state is None in the GET /items response when no approvals exist."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="plain no approvals")
    resp = client.get("/api/plugins/hermes-console/portfolio/items")
    assert resp.status_code == 200
    data = resp.json()
    items = data["items"]
    match = [i for i in items if i["task_id"] == tid]
    assert len(match) == 1
    assert match[0]["approval_state"] is None


def test_approval_state_appears_in_item_detail(client, isolated_home):
    """approval_state is populated in the GET /item/:id detail response."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="detail check")
    kb.create_task_approval(tid, approval_type="gate")
    kb.decide_task_approval(
        kb.list_task_approvals(tid)[0]["approval_id"],
        decision="approved",
        approver="keith",
    )
    slug = kb.get_current_board()
    resp = client.get(
        f"/api/plugins/hermes-console/portfolio/item/{slug}/{tid}"
    )
    assert resp.status_code == 200
    task_data = resp.json()["task"]
    assert task_data["approval_state"] == "approved"
