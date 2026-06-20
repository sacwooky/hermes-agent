"""Tests for the in-loop Robin verdict courier (kanban_autonomy). LLM-free."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_autonomy as ka
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _cp(stdout="", rc=0):
    return subprocess.CompletedProcess(["x"], rc, stdout=stdout, stderr="")


def test_robin_list_parses_newest_per_task(monkeypatch):
    out = (
        "/h/.hermes/verdicts/t_aaa__c1__200.json\n"
        "/h/.hermes/verdicts/t_aaa__c1__100.json\n"
        "/h/.hermes/verdicts/t_bbb__c2__150.json\n"
    )
    monkeypatch.setattr(ka.subprocess, "run", lambda *a, **k: _cp(stdout=out))
    got = ka._robin_list_verdict_files("robin")
    assert got == {
        "t_aaa": "/h/.hermes/verdicts/t_aaa__c1__200.json",
        "t_bbb": "/h/.hermes/verdicts/t_bbb__c2__150.json",
    }


def test_robin_list_fail_open(monkeypatch):
    def boom(*a, **k):
        raise OSError("robin unreachable")
    monkeypatch.setattr(ka.subprocess, "run", boom)
    assert ka._robin_list_verdict_files("robin") == {}


def test_verdict_already_recorded(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="r", assignee="builder")
        assert ka._verdict_already_recorded(conn, tid, "abcdef0123456789zz") is False
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "verdict_recorded", {"signature_prefix": "abcdef0123456789"})
        assert ka._verdict_already_recorded(conn, tid, "abcdef0123456789zz") is True
        assert ka._verdict_already_recorded(conn, tid, "ffffffffffffffff") is False


def test_sweep_no_pending_is_noop(kanban_home, monkeypatch):
    monkeypatch.setattr(ka, "_robin_list_verdict_files", lambda *a, **k: {})
    with kb.connect() as conn:
        assert ka.sweep_pending_robin_verdicts(conn) == {"recorded": [], "rejected": [], "seen": 0}


def test_sweep_records_then_is_idempotent(kanban_home, tmp_path, monkeypatch):
    key = tmp_path / "robin-verdict-key"
    key.write_bytes(b"test-shared-hmac-key-0123456789abcdef")
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="review me", assignee="builder")
        # run_record omitted on purpose: it is optional, so no vault needed.
        payload = {"task_id": tid, "verdict": "pass", "model_lane": "claude-code", "findings": []}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        sig = ka.sign_verdict_payload(canonical.encode(), key_path=str(key))
        vf = tmp_path / "verdicts"; vf.mkdir()
        signed_file = vf / f"{tid}__c1__100.json"
        signed_file.write_text(json.dumps({"payload": payload, "signature": sig}))

        monkeypatch.setattr(ka, "_robin_list_verdict_files", lambda *a, **k: {tid: str(signed_file)})

        def fake_scp(argv, **k):
            src = argv[2].split(":", 1)[1]
            Path(argv[3]).write_text(Path(src).read_text())
            return _cp(rc=0)
        monkeypatch.setattr(ka.subprocess, "run", fake_scp)

        res = ka.sweep_pending_robin_verdicts(conn, key_path=str(key))
        assert res["recorded"] == [tid], res
        assert ka._verdict_already_recorded(conn, tid, sig) is True

        res2 = ka.sweep_pending_robin_verdicts(conn, key_path=str(key))
        assert res2["recorded"] == [], res2


def test_sweep_skips_task_not_on_board(kanban_home, tmp_path, monkeypatch):
    """A signed verdict whose task_id is absent from this board's DB is skipped.

    conn is per-board; a verdict for a task on another board must not be fetched
    or recorded by this board's tick.
    """
    key = tmp_path / "robin-verdict-key"
    key.write_bytes(b"test-shared-hmac-key-0123456789abcdef")
    with kb.connect() as conn:
        foreign = "t_not_on_this_board"  # deliberately NOT created in this DB
        payload = {"task_id": foreign, "verdict": "pass", "model_lane": "claude-code", "findings": []}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        sig = ka.sign_verdict_payload(canonical.encode(), key_path=str(key))
        vf = tmp_path / "verdicts"; vf.mkdir()
        signed_file = vf / f"{foreign}__c1__100.json"
        signed_file.write_text(json.dumps({"payload": payload, "signature": sig}))
        monkeypatch.setattr(ka, "_robin_list_verdict_files", lambda *a, **k: {foreign: str(signed_file)})

        def boom(*a, **k):
            raise AssertionError("scp must not run for an off-board task")
        monkeypatch.setattr(ka.subprocess, "run", boom)

        res = ka.sweep_pending_robin_verdicts(conn, key_path=str(key))
        assert res == {"recorded": [], "rejected": [], "seen": 0}, res


def test_dedupe_uses_full_signature_not_prefix(kanban_home):
    """Distinct signatures sharing a 16-char prefix must NOT be conflated."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="r", assignee="builder")
        full_a = "abcdef0123456789" + "a" * 48
        full_b = "abcdef0123456789" + "b" * 48  # same first 16 chars, different signature
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "verdict_recorded",
                             {"signature": full_a, "signature_prefix": full_a[:16]})
        assert ka._verdict_already_recorded(conn, tid, full_a) is True
        assert ka._verdict_already_recorded(conn, tid, full_b) is False


def test_dedupe_legacy_prefix_fallback(kanban_home):
    """Legacy events (signature_prefix only, no full signature) still dedupe by prefix."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="r", assignee="builder")
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "verdict_recorded", {"signature_prefix": "abcdef0123456789"})
        assert ka._verdict_already_recorded(conn, tid, "abcdef0123456789" + "z" * 48) is True
