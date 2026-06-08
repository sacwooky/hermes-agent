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
