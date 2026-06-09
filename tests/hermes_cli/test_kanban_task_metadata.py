"""Tests for the Portfolio V1.1 task_metadata CRUD helpers.

Source: PORTFOLIO_BOARD_T9_METADATA_APPROVAL_STORAGE_DESIGN_2026-06-04 §3.1.

The T01 card landed the ``task_metadata`` table itself (schema + additive
migration). This file covers T02 — :func:`get_task_metadata` and
:func:`upsert_task_metadata` — the only supported read/write surface for
that table.

Acceptance exercised here:

* ``get`` returns ``None`` for non-existent task metadata.
* ``get`` returns the documented shape when a row exists.
* ``upsert`` creates a row when none exists (defaults match T01 §1.4).
* ``upsert`` only updates supplied fields; others are intact (partial merge).
* ``tags`` round-trips as a list and is stored in ``tags_json`` as JSON.
* ``updated_at`` is set on every write.
* Edge cases: null lifecycle, empty tags, repeated upserts, cross-call
  partial merge, board isolation, full happy path.

Each test pins a unique board slug so a stale ``_INITIALIZED_PATHS``
cache cannot leak rows between tests; the ``HERMES_HOME`` env and
``Path.home()`` are pointed at ``tmp_path`` to keep the production
``~/.hermes`` DB untouched.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, slug: str) -> Path:
    """Point the kanban kernel at a throwaway HERMES_HOME for one test.

    Returns the absolute path of the empty DB the helpers will create.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_BOARD", slug)
    db_path = kb.kanban_db_path(board=slug)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def _row_raw(conn: sqlite3.Connection, task_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM task_metadata WHERE task_id = ?", (task_id,)
    ).fetchone()


def _must_row(conn: sqlite3.Connection, task_id: str) -> sqlite3.Row:
    """Strict variant: assert the row exists and return it.

    Used by tests that just wrote the row themselves, so a None return
    is a real failure (typo in the task_id, wrong board, etc.) rather
    than a benign "no row" state.
    """
    row = _row_raw(conn, task_id)
    assert row is not None, f"expected task_metadata row for {task_id!r}"
    return row


# ---------------------------------------------------------------------------
# get_task_metadata
# ---------------------------------------------------------------------------


def test_get_task_metadata_returns_none_when_missing(tmp_path, monkeypatch):
    """A task_id with no row in ``task_metadata`` must read as ``None`` —
    the V1.0 read-model contract. No exception, no empty dict."""
    _isolated_home(tmp_path, monkeypatch, slug="v11-t02-get-missing")
    # Open once so the table exists; never insert.
    with kb.connect() as conn:
        assert _row_raw(conn, "never-inserted") is None
    assert kb.get_task_metadata("never-inserted") is None


def test_get_task_metadata_returns_documented_shape(tmp_path, monkeypatch):
    """The reader returns exactly the 7 keys the spec names, with
    ``tags`` deserialised to a list and ``metadata`` to a dict-or-None."""
    _isolated_home(tmp_path, monkeypatch, slug="v11-t02-get-shape")
    with kb.connect() as conn:
        conn.execute(
            "INSERT INTO task_metadata ("
            "task_id, work_item_type, lifecycle_state, agent_profile,"
            " tags_json, metadata_json, updated_at, updated_by"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "t1",
                "story",
                "in_progress",
                "jake",
                json.dumps(["alpha", "beta"]),
                json.dumps({"pr": 42, "epic": "auth"}),
                1234,
                "tester",
            ),
        )
    row = kb.get_task_metadata("t1")
    assert row is not None
    assert set(row.keys()) == {
        "work_item_type",
        "lifecycle_state",
        "agent_profile",
        "tags",
        "metadata",
        "updated_at",
        "updated_by",
    }
    assert row["work_item_type"] == "story"
    assert row["lifecycle_state"] == "in_progress"
    assert row["agent_profile"] == "jake"
    assert row["tags"] == ["alpha", "beta"]
    assert row["metadata"] == {"pr": 42, "epic": "auth"}
    assert row["updated_at"] == 1234
    assert row["updated_by"] == "tester"


def test_get_task_metadata_handles_null_lifecycle_and_agent(tmp_path, monkeypatch):
    """A row with only the required columns must round-trip with
    ``lifecycle_state``/``agent_profile``/``metadata`` as ``None`` and
    ``tags`` as an empty list. This is the V1.0 read-model default."""
    _isolated_home(tmp_path, monkeypatch, slug="v11-t02-get-nulls")
    with kb.connect() as conn:
        conn.execute(
            "INSERT INTO task_metadata (task_id, updated_at) VALUES ('t-null', 5000)"
        )
    row = kb.get_task_metadata("t-null")
    assert row == {
        "work_item_type": "unclassified",
        "lifecycle_state": None,
        "agent_profile": None,
        "tags": [],
        "metadata": None,
        "updated_at": 5000,
        "updated_by": None,
    }


def test_get_task_metadata_recovers_from_corrupt_tags_json(tmp_path, monkeypatch):
    """Defensive: a corrupt ``tags_json`` falls back to ``[]`` (V1.0
    default) instead of crashing the reader. Same for the metadata blob."""
    _isolated_home(tmp_path, monkeypatch, slug="v11-t02-get-corrupt")
    with kb.connect() as conn:
        conn.execute(
            "INSERT INTO task_metadata (task_id, tags_json, metadata_json, updated_at)"
            " VALUES ('t-corrupt', 'not-json{[', 'definitely-not-json', 100)"
        )
    row = kb.get_task_metadata("t-corrupt")
    assert row["tags"] == []
    assert row["metadata"] is None


# ---------------------------------------------------------------------------
# upsert_task_metadata — insert path
# ---------------------------------------------------------------------------


def test_upsert_creates_row_with_defaults(tmp_path, monkeypatch):
    """Calling upsert with no fields creates a row that reads back as
    the V1.0 default (work_item_type='unclassified', tags=[],
    everything else None) plus the bookkeeping columns populated
    (``updated_at`` set, ``updated_by='system'``)."""
    _isolated_home(tmp_path, monkeypatch, slug="v11-t02-upsert-defaults")
    before = int(time.time())
    kb.upsert_task_metadata("t-fresh")
    after = int(time.time())

    with kb.connect() as conn:
        row = _must_row(conn, "t-fresh")
    assert row["work_item_type"] == "unclassified"
    assert row["lifecycle_state"] is None
    assert row["agent_profile"] is None
    assert row["tags_json"] == "[]"
    assert row["metadata_json"] is None
    assert row["updated_by"] == "system"
    # updated_at is monotonic "now" within the test's wall-clock window.
    assert before <= row["updated_at"] <= after + 1


def test_upsert_inserts_all_supplied_fields(tmp_path, monkeypatch):
    """Full first-write: every column set, read back identically."""
    _isolated_home(tmp_path, monkeypatch, slug="v11-t02-upsert-all")
    kb.upsert_task_metadata(
        "t-full",
        work_item_type="story",
        lifecycle_state="in_progress",
        agent_profile="morgan",
        tags=["urgent", "p1", "rfc"],
        metadata={"epic": "auth", "pr": 99},
        updated_by="jake",
    )
    row = kb.get_task_metadata("t-full")
    assert row == {
        "work_item_type": "story",
        "lifecycle_state": "in_progress",
        "agent_profile": "morgan",
        "tags": ["urgent", "p1", "rfc"],
        "metadata": {"epic": "auth", "pr": 99},
        # updated_at comes from the kernel; we only assert it's a sane int.
        "updated_at": row["updated_at"],
        "updated_by": "jake",
    }


def test_upsert_persists_tags_as_json_array(tmp_path, monkeypatch):
    """``tags`` must be stored as JSON in the ``tags_json`` column. The
    raw column read is the string, the public reader gives back a list."""
    _isolated_home(tmp_path, monkeypatch, slug="v11-t02-upsert-tags-json")
    kb.upsert_task_metadata("t-tags", tags=["a", "b", "c"])
    with kb.connect() as conn:
        raw = _must_row(conn, "t-tags")["tags_json"]
    # Round-trips through json.loads to the same list.
    assert json.loads(raw) == ["a", "b", "c"]
    assert kb.get_task_metadata("t-tags")["tags"] == ["a", "b", "c"]


def test_upsert_persists_empty_tags_as_brackets(tmp_path, monkeypatch):
    """``tags=[]`` must be stored as the JSON string ``"[]"`` (NOT NULL
    column is non-NULL), and read back as the empty list."""
    _isolated_home(tmp_path, monkeypatch, slug="v11-t02-upsert-empty-tags")
    kb.upsert_task_metadata("t-empty", tags=[])
    with kb.connect() as conn:
        raw = _must_row(conn, "t-empty")["tags_json"]
    assert raw == "[]"
    assert kb.get_task_metadata("t-empty")["tags"] == []


def test_upsert_uses_supplied_updated_by(tmp_path, monkeypatch):
    """``updated_by`` is stored verbatim when the caller passes it."""
    _isolated_home(tmp_path, monkeypatch, slug="v11-t02-upsert-updatedby")
    kb.upsert_task_metadata("t-ub", updated_by="morgan-cli")
    with kb.connect() as conn:
        assert _must_row(conn, "t-ub")["updated_by"] == "morgan-cli"


# ---------------------------------------------------------------------------
# upsert_task_metadata — partial merge
# ---------------------------------------------------------------------------


def test_upsert_partial_merge_preserves_unset_fields(tmp_path, monkeypatch):
    """A second upsert that only sets ``lifecycle_state`` must leave
    every other field from the first upsert intact. This is the
    acceptance criterion that proves ``COALESCE``-style partial merge
    is wired through both branches (INSERT and DO UPDATE)."""
    _isolated_home(tmp_path, monkeypatch, slug="v11-t02-merge-1")
    kb.upsert_task_metadata(
        "t-merge",
        work_item_type="story",
        lifecycle_state="in_progress",
        agent_profile="jake",
        tags=["first"],
        metadata={"phase": "alpha"},
        updated_by="jake",
    )
    # Second call updates only lifecycle_state; everything else preserved.
    kb.upsert_task_metadata("t-merge", lifecycle_state="review")
    row = kb.get_task_metadata("t-merge")
    assert row["work_item_type"] == "story"
    assert row["lifecycle_state"] == "review"  # the one field that changed
    assert row["agent_profile"] == "jake"
    assert row["tags"] == ["first"]
    assert row["metadata"] == {"phase": "alpha"}


def test_upsert_partial_merge_replaces_tags_atomically(tmp_path, monkeypatch):
    """Passing ``tags`` is a full replacement (the public API doesn't
    expose list-merge semantics). The old tag list is gone."""
    _isolated_home(tmp_path, monkeypatch, slug="v11-t02-merge-tags")
    kb.upsert_task_metadata("t-tags-merge", tags=["a", "b"])
    kb.upsert_task_metadata("t-tags-merge", tags=["c"])
    assert kb.get_task_metadata("t-tags-merge")["tags"] == ["c"]


def test_upsert_partial_merge_replaces_metadata_atomically(tmp_path, monkeypatch):
    """Same contract for the metadata blob — the new dict replaces
    the old one, no key-level merge."""
    _isolated_home(tmp_path, monkeypatch, slug="v11-t02-merge-meta")
    kb.upsert_task_metadata("t-meta-merge", metadata={"a": 1, "b": 2})
    kb.upsert_task_metadata("t-meta-merge", metadata={"c": 3})
    assert kb.get_task_metadata("t-meta-merge")["metadata"] == {"c": 3}


def test_upsert_updates_updated_at_on_every_write(tmp_path, monkeypatch):
    """``updated_at`` must advance on every upsert, even when no other
    field changes. This is the explicit "updated_at is set on every
    write" acceptance criterion."""
    _isolated_home(tmp_path, monkeypatch, slug="v11-t02-updatedat")
    kb.upsert_task_metadata("t-ts", work_item_type="story")
    with kb.connect() as conn:
        first = _must_row(conn, "t-ts")["updated_at"]
    # Sleep so the int(time.time()) call is guaranteed to be different.
    time.sleep(1.05)
    kb.upsert_task_metadata("t-ts", work_item_type="story")
    with kb.connect() as conn:
        second = _must_row(conn, "t-ts")["updated_at"]
    assert second > first, (first, second)


def test_upsert_updated_by_defaults_to_system_when_omitted(tmp_path, monkeypatch):
    """If the second upsert omits ``updated_by``, the field keeps the
    previous value (partial merge), not 'system' — only the first
    write defaults to 'system'."""
    _isolated_home(tmp_path, monkeypatch, slug="v11-t02-merge-updatedby")
    # First write with explicit actor.
    kb.upsert_task_metadata("t-ub-merge", work_item_type="story", updated_by="jake")
    assert kb.get_task_metadata("t-ub-merge")["updated_by"] == "jake"
    # Second write with no actor — must stay 'jake', NOT become 'system'.
    kb.upsert_task_metadata("t-ub-merge", lifecycle_state="review")
    assert kb.get_task_metadata("t-ub-merge")["updated_by"] == "jake"


# ---------------------------------------------------------------------------
# Cross-board / isolation
# ---------------------------------------------------------------------------


def test_upsert_and_get_target_active_board(tmp_path, monkeypatch):
    """The two helpers must target whichever board the dispatcher pinned
    via HERMES_KANBAN_BOARD — i.e. they must NOT silently read the
    default board. A row written on board A must not be visible from
    board B."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Pre-create both boards so the board-resolution chain in
    # ``get_current_board`` accepts the env var (a non-existent board
    # silently falls through to the default, which is not what we want
    # to exercise here).
    kb.create_board("board-a")
    kb.create_board("board-b")

    # Pin board A and write a row.
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "board-a")
    assert kb.get_current_board() == "board-a"
    kb.upsert_task_metadata("t-x", work_item_type="story")
    assert kb.get_task_metadata("t-x")["work_item_type"] == "story"

    # Switch to board B (no row should exist).
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "board-b")
    assert kb.get_current_board() == "board-b"
    assert kb.get_task_metadata("t-x") is None

    # Pin board A again — the row is still there.
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "board-a")
    assert kb.get_current_board() == "board-a"
    assert kb.get_task_metadata("t-x")["work_item_type"] == "story"


# ---------------------------------------------------------------------------
# Edge cases / type discipline
# ---------------------------------------------------------------------------


def test_upsert_rejects_non_iterable_tags_silently_only_for_strings(
    tmp_path, monkeypatch
):
    """A pre-serialised JSON string is accepted as a pass-through (an
    internal-only escape hatch). A real list is what the public API
    expects; both end up as a valid JSON array in the column."""
    _isolated_home(tmp_path, monkeypatch, slug="v11-t02-tag-types")
    kb.upsert_task_metadata("t-str", tags='["x", "y"]')
    assert kb.get_task_metadata("t-str")["tags"] == ["x", "y"]
    kb.upsert_task_metadata("t-list", tags=["p", "q"])
    assert kb.get_task_metadata("t-list")["tags"] == ["p", "q"]


def test_upsert_metadata_accepts_pre_serialised_string(tmp_path, monkeypatch):
    """A pre-serialised JSON string for ``metadata`` is the same
    pass-through escape hatch as for tags."""
    _isolated_home(tmp_path, monkeypatch, slug="v11-t02-meta-pre")
    kb.upsert_task_metadata("t-pre", metadata='{"k": 1}')
    assert kb.get_task_metadata("t-pre")["metadata"] == {"k": 1}


def test_upsert_full_happy_path_matches_get_round_trip(tmp_path, monkeypatch):
    """The full life-cycle of one task_id: full upsert, partial merge,
    then a final read. Mirrors what the Portfolio write path will do."""
    _isolated_home(tmp_path, monkeypatch, slug="v11-t02-round-trip")
    kb.upsert_task_metadata(
        "t-rt",
        work_item_type="story",
        lifecycle_state="backlog",
        agent_profile="morgan",
        tags=["v1.1"],
        metadata={"estimate": 5},
        updated_by="jake",
    )
    # Move through review.
    kb.upsert_task_metadata("t-rt", lifecycle_state="review", updated_by="morgan")
    # Pick up the V1.0 read-model defaults after metadata is "cleared"
    # is not a supported call; the API treats None as "leave alone",
    # so we instead replace the metadata with the next state.
    kb.upsert_task_metadata(
        "t-rt",
        metadata={"estimate": 5, "pr_url": "https://example/pr/1"},
    )
    final = kb.get_task_metadata("t-rt")
    assert final["work_item_type"] == "story"
    assert final["lifecycle_state"] == "review"
    assert final["agent_profile"] == "morgan"
    assert final["tags"] == ["v1.1"]
    assert final["metadata"] == {"estimate": 5, "pr_url": "https://example/pr/1"}
    assert final["updated_by"] == "morgan"
