"""Slice 2 — the in-process board conductor loop (run_board_conductor).

Injectable design: no agent, no DB — fakes for run_turn / lifecycle / judge.
Covers happy path, judge-gated DONE, per-card budget block, explicit BLOCKED,
dependency re-query, and lost-claim skip.
"""

from __future__ import annotations

from hermes_cli.kanban_conductor import run_board_conductor


def _judge(verdict, reason="ok"):
    return lambda goal, resp: (verdict, reason, False, None)


def _mk_backend(cards):
    """A fake board backend recording lifecycle calls."""
    state = {
        "claimed": [],
        "completed": [],
        "blocked": [],
        "checkpoints": [],
        "remaining": list(cards),
    }

    def list_workable():
        return [c for c in state["remaining"]]

    def claim(cid):
        state["claimed"].append(cid)
        return True

    def complete(cid, summary):
        state["completed"].append((cid, summary))
        state["remaining"] = [c for c in state["remaining"] if c["id"] != cid]

    def block(cid, reason):
        state["blocked"].append((cid, reason))
        state["remaining"] = [c for c in state["remaining"] if c["id"] != cid]

    def checkpoint(cid, note):
        state["checkpoints"].append((cid, note))

    return state, dict(
        list_workable=list_workable, claim=claim, complete=complete,
        block=block, checkpoint=checkpoint,
        build_prompt=lambda card: f"work {card['id']}",
    )


def test_happy_path_completes_all_cards():
    cards = [{"id": "a", "title": "A", "body": ""}, {"id": "b", "title": "B", "body": ""}]
    state, be = _mk_backend(cards)
    res = run_board_conductor(
        board="t", run_turn=lambda p: "DONE: built it", judge=_judge("done"), **be,
    )
    assert res["completed"] == ["a", "b"]
    assert res["blocked"] == []
    assert [c for c, _ in state["completed"]] == ["a", "b"]
    assert state["claimed"] == ["a", "b"]
    # one turn per card is enough when the judge agrees
    assert res["turns_used"] == 2


def test_explicit_done_requires_judge_agreement():
    # Agent says DONE but judge says continue → keeps going until per-card budget,
    # then the card is blocked (never falsely completed).
    cards = [{"id": "a", "title": "A", "body": ""}]
    state, be = _mk_backend(cards)
    res = run_board_conductor(
        board="t", run_turn=lambda p: "DONE: i think im done",
        judge=_judge("continue", "criteria 3 not met"),
        max_turns_per_card=3, **be,
    )
    assert res["completed"] == []
    assert res["blocked"] == ["a"]
    assert "criteria 3 not met" in state["blocked"][0][1]
    assert res["turns_used"] == 3  # spent the per-card budget


def test_budget_exhaustion_blocks_card():
    cards = [{"id": "a", "title": "A", "body": ""}]
    state, be = _mk_backend(cards)
    res = run_board_conductor(
        board="t", run_turn=lambda p: "still working...",
        judge=_judge("continue"), max_turns_per_card=2, max_total_turns=2, **be,
    )
    assert res["blocked"] == ["a"]
    assert res["completed"] == []


def test_explicit_blocked_signal_stops_card_immediately():
    cards = [{"id": "a", "title": "A", "body": ""}]
    state, be = _mk_backend(cards)
    res = run_board_conductor(
        board="t", run_turn=lambda p: "BLOCKED: need the API key from a human",
        judge=_judge("continue"), **be,
    )
    assert res["blocked"] == ["a"]
    assert res["turns_used"] == 1  # stopped on the first turn
    assert "BLOCKED" in state["blocked"][0][1]


def test_dependency_requery_picks_up_newly_unblocked_card():
    # b only becomes workable after a completes. list_workable is re-queried
    # each outer pass, so the conductor must pick b up in a later pass.
    a = {"id": "a", "title": "A", "body": ""}
    b = {"id": "b", "title": "B", "body": ""}
    reveal = {"b_ready": False}
    state = {"claimed": [], "completed": [], "blocked": []}

    def list_workable():
        out = []
        if "a" not in [c for c, _ in state["completed"]]:
            out.append(a)
        if reveal["b_ready"] and "b" not in [c for c, _ in state["completed"]]:
            out.append(b)
        return out

    def claim(cid):
        state["claimed"].append(cid)
        return True

    def complete(cid, summary):
        state["completed"].append((cid, summary))
        if cid == "a":
            reveal["b_ready"] = True  # a done → b unblocks

    def block(cid, reason):
        state["blocked"].append((cid, reason))

    res = run_board_conductor(
        board="t", run_turn=lambda p: "DONE: ok", judge=_judge("done"),
        list_workable=list_workable, claim=claim, complete=complete, block=block,
        build_prompt=lambda c: "go",
    )
    assert res["completed"] == ["a", "b"]


def test_lost_claim_is_skipped_not_looped_forever():
    a = {"id": "a", "title": "A", "body": ""}
    calls = {"n": 0}

    def list_workable():
        # Always offers `a`; the conductor must NOT spin forever when it can
        # never claim it (seen-set + no-progress guard both apply).
        return [a]

    def claim(cid):
        calls["n"] += 1
        return False  # never claimable

    res = run_board_conductor(
        board="t", run_turn=lambda p: "x", judge=_judge("done"),
        list_workable=list_workable, claim=claim,
        complete=lambda *a: None, block=lambda *a: None, build_prompt=lambda c: "go",
    )
    assert res["completed"] == []
    assert res["blocked"] == []
    assert calls["n"] == 1  # one attempt, then no-progress stop


def test_no_workable_cards_is_a_clean_noop():
    res = run_board_conductor(
        board="t", run_turn=lambda p: "x", judge=_judge("done"),
        list_workable=lambda: [], claim=lambda c: True,
        complete=lambda *a: None, block=lambda *a: None, build_prompt=lambda c: "go",
    )
    assert res == {"completed": [], "blocked": [], "turns_used": 0, "cards_seen": 0}


# --------------------------------------------------------------------------
# Slice 4 — in-session idle recheck (event-driven resume without re-spawn)
# --------------------------------------------------------------------------

def test_idle_recheck_picks_up_late_card_in_same_session():
    # Board starts empty of workable cards; a card appears after 2 rechecks
    # (simulating a child unblocked mid-session). The conductor must pick it up
    # WITHOUT exiting, then complete it — all in one session.
    clock = {"t": 0.0}
    appear_after = 2
    calls = {"list": 0}
    state = {"completed": []}

    def list_workable():
        calls["list"] += 1
        if calls["list"] > appear_after and "late" not in state["completed"]:
            return [{"id": "late", "title": "L", "body": ""}]
        return []

    def fake_sleep(s):
        clock["t"] += s

    res = run_board_conductor(
        board="t", run_turn=lambda p: "DONE: ok", judge=_judge("done"),
        list_workable=list_workable, claim=lambda c: True,
        complete=lambda cid, s: state["completed"].append(cid),
        block=lambda *a: None, build_prompt=lambda c: "go",
        idle_max_seconds=60, idle_recheck_seconds=5,
        sleep=fake_sleep, monotonic=lambda: clock["t"],
    )
    assert res["completed"] == ["late"]


def test_idle_recheck_exits_after_timeout_when_no_work_appears():
    clock = {"t": 0.0}

    def fake_sleep(s):
        clock["t"] += s

    res = run_board_conductor(
        board="t", run_turn=lambda p: "x", judge=_judge("done"),
        list_workable=lambda: [], claim=lambda c: True,
        complete=lambda *a: None, block=lambda *a: None, build_prompt=lambda c: "go",
        idle_max_seconds=30, idle_recheck_seconds=5,
        sleep=fake_sleep, monotonic=lambda: clock["t"],
    )
    assert res == {"completed": [], "blocked": [], "turns_used": 0, "cards_seen": 0}
    assert clock["t"] >= 30  # waited out the idle window before exiting
