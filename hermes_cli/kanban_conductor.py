"""Continuous board conductor (slice 2).

One long-lived session drives a whole board's cards to completion in-process,
instead of the dispatcher spawning a fresh worker subprocess per card. The
conductor holds context across cards (like Claude Code), drives the *build*
work with the session agent (edits, tests, ``delegate`` for independent
sub-parts), and drives *card lifecycle* deterministically in Python
(``claim`` / ``complete`` / ``block`` on the kanban DB directly). It carries no
``HERMES_KANBAN_TASK``, so the single-card kanban lifecycle *tools* do not
attach and cannot bind the session to one card — bookkeeping is Python's job.

This module is fully decoupled from the CLI for testability: callers inject
``run_turn`` (str -> str, one agent turn in the same session), the card
enumeration / lifecycle callbacks, and a per-card build-prompt builder. The
production wiring lives in ``cli._run_kanban_conductor_q``.

Human gates and the review handoff are NOT driven here: a gated card (awaiting a
human answer) or a review-lane card is simply left for its existing path and the
event-wake (slice 4) resumes the conductor when the answer/verdict lands. The
conductor drives only the *buildable, un-gated* work continuously.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from hermes_cli.goals import DEFAULT_MAX_TURNS, judge_goal

# A conductor session has no kanban lifecycle tools (no HERMES_KANBAN_TASK), so
# unlike the goal loop we do NOT tell it to call kanban_complete/kanban_block —
# Python finalizes the card. We ask it to do the real work and signal done in
# plain text; the auxiliary judge decides completion against the card body.
CONDUCTOR_CONTINUATION_TEMPLATE = (
    "[Continuing this task — the judge says it is not finished yet]\n"
    "Reason: {reason}\n\n"
    "Take the next concrete step: edit files, run the relevant tests/build, and "
    "delegate independent sub-parts if that is faster. When the task genuinely "
    "meets its acceptance criteria, end your message with a line starting "
    "'DONE:' summarizing what changed and how you verified it. If you are truly "
    "blocked on something only a human can resolve, end with a line starting "
    "'BLOCKED:' and the reason."
)

# Default per-card budget: a card should finish in a handful of turns; the total
# budget bounds the whole board so a runaway card can never starve the session.
DEFAULT_MAX_TURNS_PER_CARD = 12


def _coerce_verdict(goal_text: str, last_response: str, judge: Callable) -> tuple[str, str]:
    """Run the judge and reduce it to (verdict, reason).

    ``wait`` / ``skipped`` are coerced to ``continue`` — the conductor has no
    wait-barrier concept and an unreachable judge should keep the loop moving
    rather than wedge it.
    """
    verdict, reason, _parse_failed, _wait = judge(goal_text, last_response)
    if verdict in ("wait", "skipped"):
        verdict = "continue"
    return verdict, reason


def _explicit_signal(response: str) -> Optional[str]:
    """Detect an explicit DONE:/BLOCKED: signal in the agent's message.

    A fast path so the conductor doesn't spend a judge call when the agent has
    already declared terminal state. Returns 'done', 'blocked', or None.
    """
    for line in reversed((response or "").splitlines()):
        s = line.strip()
        if s.upper().startswith("DONE:"):
            return "done"
        if s.upper().startswith("BLOCKED:"):
            return "blocked"
    return None


def run_board_conductor(
    *,
    board: str,
    run_turn: Callable[[str], str],
    list_workable: Callable[[], List[Dict[str, Any]]],
    claim: Callable[[str], bool],
    complete: Callable[[str, str], None],
    block: Callable[[str, str], None],
    build_prompt: Callable[[Dict[str, Any]], str],
    checkpoint: Optional[Callable[[str, str], None]] = None,
    judge: Optional[Callable] = None,
    max_total_turns: int = DEFAULT_MAX_TURNS,
    max_turns_per_card: int = DEFAULT_MAX_TURNS_PER_CARD,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Drive ``board``'s buildable cards to completion in one session.

    Returns ``{"completed": [...], "blocked": [...], "turns_used": int,
    "cards_seen": int}``.

    Injected contract:
      - ``list_workable()`` returns the currently buildable cards (ready,
        assigned, dependencies met, NOT gated / NOT in a review lane), each a
        dict with at least ``id`` and ``title``/``body``. Re-queried each outer
        pass so cards unblocked by a just-finished parent are picked up.
      - ``claim(card_id)`` atomically moves ready->running; returns False if the
        claim was lost (another writer took it) — that card is skipped this pass.
      - ``complete(card_id, summary)`` / ``block(card_id, reason)`` finalize.
      - ``build_prompt(card)`` returns the first-turn directive for a card.
      - ``checkpoint(card_id, note)`` (optional) appends a durable progress note
        so a restarted conductor can resume.
    """

    def _log(msg: str) -> None:
        if log is not None:
            try:
                log(msg)
            except Exception:
                pass

    judge = judge or judge_goal
    max_total_turns = int(max_total_turns or DEFAULT_MAX_TURNS)
    if max_total_turns < 1:
        max_total_turns = DEFAULT_MAX_TURNS
    max_turns_per_card = int(max_turns_per_card or DEFAULT_MAX_TURNS_PER_CARD)
    if max_turns_per_card < 1:
        max_turns_per_card = DEFAULT_MAX_TURNS_PER_CARD

    completed: List[str] = []
    blocked: List[str] = []
    seen: set[str] = set()
    turns_used = 0

    while turns_used < max_total_turns:
        try:
            cards = [c for c in list_workable() if c.get("id") not in seen]
        except Exception as exc:
            _log(f"conductor {board}: list_workable failed ({exc}); stopping")
            break
        if not cards:
            _log(f"conductor {board}: no workable cards remain; board drained")
            break

        progressed = False
        for card in cards:
            if turns_used >= max_total_turns:
                break
            card_id = card.get("id")
            if not card_id:
                continue
            try:
                if not claim(card_id):
                    _log(f"conductor {board}: could not claim {card_id}; skipping this pass")
                    continue
            except Exception as exc:
                _log(f"conductor {board}: claim {card_id} failed ({exc}); skipping")
                continue

            progressed = True
            seen.add(card_id)
            goal_text = "\n\n".join(
                p for p in (card.get("title") or "", card.get("body") or "") if p
            ).strip()
            prompt = build_prompt(card)
            outcome: Optional[str] = None
            reason = ""
            last_response = ""
            card_turns = 0
            while card_turns < max_turns_per_card and turns_used < max_total_turns:
                try:
                    last_response = run_turn(prompt)
                except Exception as exc:
                    _log(f"conductor {board}: run_turn on {card_id} failed ({exc})")
                    outcome, reason = "blocked", f"conductor run_turn error: {exc}"
                    break
                turns_used += 1
                card_turns += 1

                sig = _explicit_signal(last_response)
                if sig == "blocked":
                    outcome, reason = "blocked", "agent signalled BLOCKED"
                    break
                if sig == "done":
                    # Trust an explicit DONE only if the judge agrees.
                    verdict, reason = _coerce_verdict(goal_text, last_response, judge)
                    if verdict == "done":
                        outcome = "done"
                        break
                    # judge disagrees — keep going with its reason.
                    prompt = CONDUCTOR_CONTINUATION_TEMPLATE.format(reason=reason)
                    continue

                verdict, reason = _coerce_verdict(goal_text, last_response, judge)
                if verdict == "done":
                    outcome = "done"
                    break
                prompt = CONDUCTOR_CONTINUATION_TEMPLATE.format(reason=reason)

            if outcome == "done":
                try:
                    complete(card_id, _completion_summary(last_response))
                    completed.append(card_id)
                    _log(f"conductor {board}: completed {card_id} in {card_turns} turn(s)")
                    if checkpoint:
                        checkpoint(card_id, f"conductor: completed after {card_turns} turn(s)")
                except Exception as exc:
                    _log(f"conductor {board}: complete {card_id} failed ({exc})")
            else:
                block_reason = reason or "conductor: per-card turn budget exhausted before completion"
                try:
                    block(card_id, block_reason)
                    blocked.append(card_id)
                    _log(f"conductor {board}: blocked {card_id} — {block_reason}")
                    if checkpoint:
                        checkpoint(card_id, f"conductor: blocked — {block_reason}")
                except Exception as exc:
                    _log(f"conductor {board}: block {card_id} failed ({exc})")

        if not progressed:
            _log(f"conductor {board}: no card claimable this pass; stopping")
            break

    return {
        "completed": completed,
        "blocked": blocked,
        "turns_used": turns_used,
        "cards_seen": len(seen),
    }


def _completion_summary(response: str, *, limit: int = 600) -> str:
    """Extract a short completion summary from the agent's final message.

    Prefers the explicit ``DONE:`` line; falls back to the tail of the message.
    """
    for line in reversed((response or "").splitlines()):
        s = line.strip()
        if s.upper().startswith("DONE:"):
            return s[len("DONE:"):].strip()[:limit] or "completed"
    tail = (response or "").strip()
    return (tail[-limit:] if tail else "completed").strip() or "completed"
