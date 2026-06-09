from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from hermes_cli import kanban_db as kb


def _make_legacy_db(path: Path) -> None:
    """Write a kanban DB with the pre-AUTOINCREMENT (TEXT PK) schema for the
    four tables #35096 affects, keeping every other table current so the
    additive-column migration runs cleanly on top.
    """
    conn = sqlite3.connect(str(path))
    conn.executescript(kb.SCHEMA_SQL)
    conn.executescript(
        """
        DROP TABLE task_events;
        DROP TABLE task_comments;
        DROP TABLE task_runs;
        DROP TABLE kanban_notify_subs;
        CREATE TABLE task_comments (id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
            author TEXT NOT NULL, body TEXT NOT NULL, created_at INTEGER NOT NULL);
        CREATE TABLE task_events (id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
            kind TEXT NOT NULL, payload TEXT, created_at INTEGER NOT NULL);
        CREATE TABLE task_runs (id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
            profile TEXT, status TEXT NOT NULL, started_at INTEGER NOT NULL);
        CREATE TABLE kanban_notify_subs (task_id TEXT NOT NULL, platform TEXT NOT NULL,
            chat_id TEXT NOT NULL, thread_id TEXT NOT NULL DEFAULT '', user_id TEXT,
            created_at INTEGER NOT NULL, last_event_id TEXT,
            PRIMARY KEY (task_id, platform, chat_id, thread_id));
        """
    )
    conn.execute(
        "INSERT INTO tasks (id, title, status, created_at) VALUES ('task-1', 'T', 'done', 1000)"
    )
    conn.execute(
        "INSERT INTO task_comments VALUES ('c-1', 'task-1', 'agent', 'hi', 1500)"
    )
    conn.execute(
        "INSERT INTO task_events VALUES ('e-1', 'task-1', 'completed', NULL, 2000)"
    )
    conn.execute(
        "INSERT INTO task_events VALUES ('e-2', 'task-1', 'blocked', NULL, 2100)"
    )
    conn.execute(
        "INSERT INTO task_runs VALUES ('r-1', 'task-1', 'default', 'done', 1000)"
    )
    conn.execute(
        "INSERT INTO kanban_notify_subs (task_id, platform, chat_id, created_at, last_event_id) "
        "VALUES ('task-1', 'telegram', '123', 1000, 'e-1')"
    )
    conn.commit()
    conn.close()


def _setup_home(tmp_path, monkeypatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="legacy")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    return db_path


def _table_struct(conn: sqlite3.Connection, table: str):
    cols = [
        (r["name"], (r["type"] or "").upper(), r["notnull"], r["pk"])
        for r in conn.execute(f"PRAGMA table_info({table})")
    ]
    idx = sorted(
        r["name"]
        for r in conn.execute(f"PRAGMA index_list({table})")
        if not r["name"].startswith("sqlite_")
    )
    return cols, idx


def test_connect_initialization_is_thread_safe(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        try:
            barrier.wait(timeout=5)
            conn = kb.connect(board="default")
            conn.close()
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    with kb.connect(board="default") as conn:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    assert "max_retries" in cols


def test_legacy_text_pk_tables_rebuilt_to_integer_autoincrement(tmp_path, monkeypatch):
    """A pre-AUTOINCREMENT DB is migrated in place: id columns become INTEGER
    PKs, ``last_event_id`` becomes INTEGER, data is preserved, and indexes
    are recreated (DROP TABLE would otherwise take them down)."""
    db_path = _setup_home(tmp_path, monkeypatch)
    _make_legacy_db(db_path)

    with kb.connect(db_path) as conn:
        for table in ("task_events", "task_comments", "task_runs"):
            id_col = {
                r["name"]: r for r in conn.execute(f"PRAGMA table_info({table})")
            }["id"]
            assert id_col["type"].upper() == "INTEGER" and id_col["pk"] == 1

        lei = {
            r["name"]: r for r in conn.execute("PRAGMA table_info(kanban_notify_subs)")
        }
        assert lei["last_event_id"]["type"].upper() == "INTEGER"

        # Data preserved across the rebuild.
        assert len(conn.execute("SELECT * FROM task_events").fetchall()) == 2
        assert conn.execute("SELECT body FROM task_comments").fetchone()["body"] == "hi"
        assert len(conn.execute("SELECT * FROM task_runs").fetchall()) == 1
        # Non-numeric legacy cursor ("e-1") casts to 0.
        assert (
            conn.execute("SELECT last_event_id FROM kanban_notify_subs").fetchone()[
                "last_event_id"
            ]
            == 0
        )

        # Indexes restored, including idx_events_run (added by the additive pass).
        indexes = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }
        for name in (
            "idx_events_task",
            "idx_events_run",
            "idx_comments_task",
            "idx_runs_task",
            "idx_runs_status",
            "idx_notify_task",
        ):
            assert name in indexes

        # AUTOINCREMENT actually works after the rebuild.
        conn.execute(
            "INSERT INTO task_events (task_id, kind, created_at) VALUES ('task-1', 'completed', 3000)"
        )
        new_id = conn.execute(
            "SELECT id FROM task_events ORDER BY id DESC LIMIT 1"
        ).fetchone()["id"]
        assert isinstance(new_id, int) and new_id >= 1


def test_rebuilt_schema_matches_fresh_db(tmp_path, monkeypatch):
    """The rebuilt tables must be structurally identical to a fresh DB, so the
    hand-written DDL in ``_REBUILD_SPECS`` can't silently drift from SCHEMA_SQL."""
    legacy_path = _setup_home(tmp_path, monkeypatch)
    _make_legacy_db(legacy_path)
    fresh_path = kb.kanban_db_path(board="fresh")
    fresh_path.parent.mkdir(parents=True, exist_ok=True)
    kb._INITIALIZED_PATHS.discard(str(fresh_path.resolve()))

    with kb.connect(legacy_path) as migrated, kb.connect(fresh_path) as fresh:
        for table in (
            "task_events",
            "task_comments",
            "task_runs",
            "kanban_notify_subs",
        ):
            assert _table_struct(migrated, table) == _table_struct(fresh, table)


def test_migration_is_idempotent(tmp_path, monkeypatch):
    """Re-opening an already-migrated DB is a no-op and leaves data intact."""
    db_path = _setup_home(tmp_path, monkeypatch)
    _make_legacy_db(db_path)

    with kb.connect(db_path):
        pass
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with kb.connect(db_path) as conn:
        id_col = {r["name"]: r for r in conn.execute("PRAGMA table_info(task_events)")}[
            "id"
        ]
        assert id_col["type"].upper() == "INTEGER"
        assert len(conn.execute("SELECT * FROM task_events").fetchall()) == 2


def test_unseen_events_for_sub_survives_migrated_db(tmp_path, monkeypatch):
    """The crash that motivated #35096 — ``int(None)`` on a NULL cursor — is
    gone after migration; the notifier query returns an integer cursor."""
    db_path = _setup_home(tmp_path, monkeypatch)
    _make_legacy_db(db_path)

    with kb.connect(db_path) as conn:
        cursor, events = kb.unseen_events_for_sub(
            conn, task_id="task-1", platform="telegram", chat_id="123"
        )
        assert isinstance(cursor, int)
        assert isinstance(events, list)


# ---------------------------------------------------------------------------
# V1.1 task_metadata additive migration
# ---------------------------------------------------------------------------
# Source: PORTFOLIO_BOARD_T9_METADATA_APPROVAL_STORAGE_DESIGN_2026-06-04 §3.1.
# ``task_metadata`` is the V1.1 Portfolio augmentation table keyed by
# ``task_id``. The V1.0 read model treats a missing row as
# ``work_item_type='unclassified'`` with null lifecycle/agent_profile and
# ``tags=[]``, so the migration is purely additive — boards that never
# write a row keep behaving exactly as they did in V1.0.

_TASK_METADATA_COLS = [
    "task_id",
    "work_item_type",
    "lifecycle_state",
    "agent_profile",
    "tags_json",
    "metadata_json",
    "updated_at",
    "updated_by",
]
_TASK_METADATA_INDEXES = {
    "idx_metadata_agent",
    "idx_metadata_lifecycle",
    "idx_metadata_type",
}


def _task_metadata_user_indexes(conn: sqlite3.Connection) -> set[str]:
    return {
        r["name"]
        for r in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND tbl_name='task_metadata' "
            "AND name NOT LIKE 'sqlite_%'"
        )
    }


def test_task_metadata_table_present_on_fresh_db(tmp_path, monkeypatch):
    """A board opened for the first time on a V1.1+ build gets
    ``task_metadata`` with the documented columns and all 3 indexes."""
    db_path = kb.kanban_db_path(board="v11-fresh")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    with kb.connect(db_path) as conn:
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(task_metadata)")]
        assert cols == _TASK_METADATA_COLS, cols
        assert _task_metadata_user_indexes(conn) == _TASK_METADATA_INDEXES


def test_task_metadata_added_to_legacy_board_on_open(tmp_path, monkeypatch):
    """A pre-V1.1 board (schema present, ``task_metadata`` absent) gets the
    table the next time it is opened via :func:`connect`. No data loss on
    the V1.0 tables; no schema conflict."""
    db_path = _setup_home(tmp_path, monkeypatch)
    # Build a fully-V1.0 board manually: full SCHEMA_SQL minus the new table
    # and minus the new indexes. This is the exact shape a board has when
    # the V1.1 build first opens it.
    raw = sqlite3.connect(str(db_path))
    raw.executescript(kb.SCHEMA_SQL)
    raw.execute("DROP TABLE task_metadata")
    for idx in _TASK_METADATA_INDEXES:
        raw.execute(f"DROP INDEX IF EXISTS {idx}")
    raw.execute(
        "INSERT INTO tasks (id, title, status, created_at) "
        "VALUES ('legacy-1', 'pre-V1.1 task', 'todo', 1000)"
    )
    raw.commit()
    raw.close()

    # Re-open via the public kernel path. The additive migration must
    # create the table and its indexes; the pre-existing task row must
    # still be there.
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with kb.connect(db_path) as conn:
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(task_metadata)")]
        assert cols == _TASK_METADATA_COLS, cols
        assert _task_metadata_user_indexes(conn) == _TASK_METADATA_INDEXES
        survivor = conn.execute(
            "SELECT id, title, status FROM tasks WHERE id='legacy-1'"
        ).fetchone()
        assert survivor is not None
        assert (survivor["id"], survivor["title"], survivor["status"]) == (
            "legacy-1",
            "pre-V1.1 task",
            "todo",
        )


def test_task_metadata_migration_is_idempotent(tmp_path, monkeypatch):
    """Re-opening a board that already has ``task_metadata`` is a no-op and
    leaves data intact. This is the regression guard for the
    ``CREATE … IF NOT EXISTS`` pattern: a process that opens the DB,
    closes it, and re-opens it must not raise and must not duplicate
    indexes or rows."""
    db_path = kb.kanban_db_path(board="v11-idem")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    # First open — creates the table.
    with kb.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO task_metadata (task_id, work_item_type, "
            "lifecycle_state, agent_profile, updated_at, updated_by) "
            "VALUES ('t1', 'story', 'in_progress', 'jake', 5000, 'tester')"
        )

    # Force the cache to forget so the migration pass runs again on the
    # second open. ``CREATE … IF NOT EXISTS`` is what makes this safe.
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    with kb.connect(db_path) as conn:
        # Table still has exactly the 3 user indexes (no duplicates from
        # the second pass).
        assert _task_metadata_user_indexes(conn) == _TASK_METADATA_INDEXES
        # Row from the first open is still there with the same values.
        row = conn.execute(
            "SELECT work_item_type, lifecycle_state, agent_profile, updated_by "
            "FROM task_metadata WHERE task_id='t1'"
        ).fetchone()
        assert row is not None
        assert (
            row["work_item_type"],
            row["lifecycle_state"],
            row["agent_profile"],
            row["updated_by"],
        ) == (
            "story",
            "in_progress",
            "jake",
            "tester",
        )


def test_task_metadata_defaults_match_v10_read_model(tmp_path, monkeypatch):
    """The Portfolio read model treats a row with only ``task_id`` +
    ``updated_at`` as ``work_item_type='unclassified'`` with
    ``lifecycle_state=None``, ``agent_profile=None``, ``tags=[]``,
    ``metadata_json=None``, ``updated_by=None``. Those are the
    NOT-NULL / DEFAULT clauses on the schema, so a row written with
    just the required columns must match exactly."""
    db_path = kb.kanban_db_path(board="v11-defaults")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    with kb.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO task_metadata (task_id, updated_at) VALUES ('t1', 1000)"
        )
        row = conn.execute(
            "SELECT work_item_type, lifecycle_state, agent_profile, "
            "tags_json, metadata_json, updated_by "
            "FROM task_metadata WHERE task_id='t1'"
        ).fetchone()
        assert row["work_item_type"] == "unclassified"
        assert row["lifecycle_state"] is None
        assert row["agent_profile"] is None
        assert row["tags_json"] == "[]"
        assert row["metadata_json"] is None
        assert row["updated_by"] is None


def test_task_metadata_indexes_are_used_by_lookups(tmp_path, monkeypatch):
    """The 3 indexes are real: the planner picks them up for the read
    patterns the Portfolio uses (filter by work_item_type, lifecycle,
    agent_profile). Catches a future migration that accidentally drops
    them or misspells the column references."""
    db_path = kb.kanban_db_path(board="v11-idx")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    with kb.connect(db_path) as conn:
        # Insert 100 rows so the planner has a reason to prefer an index.
        for i in range(100):
            conn.execute(
                "INSERT INTO task_metadata (task_id, work_item_type, "
                "lifecycle_state, agent_profile, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    f"t{i}",
                    "story" if i % 2 else "task",
                    "in_progress",
                    "jake",
                    1000 + i,
                ),
            )
        # Each index must appear in the query plan for a query that
        # filters on its column. ``SEARCH ... USING INDEX`` confirms
        # SQLite picked the index over a full scan.
        for col in ("work_item_type", "lifecycle_state", "agent_profile"):
            plan = conn.execute(
                f"EXPLAIN QUERY PLAN SELECT task_id FROM task_metadata WHERE {col} = ?",
                ("story",),
            ).fetchall()
            plan_text = " | ".join(str(dict(r)) for r in plan)
            assert "USING INDEX" in plan_text, (
                f"index on {col} not used by planner: {plan_text}"
            )
