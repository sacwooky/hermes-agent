#!/usr/bin/env python3
"""Hermes Arena — Bradley-Terry/Elo agent benchmark (Layer 3, GATED).

The "inject our real agent benchmark scores" layer. Instead of trusting only a
third-party intelligence index, we run OUR OWN blind A/B comparisons on OUR real
lanes (builder, reviewer, decomposer, …): two anonymised model outputs → an
LLM judge picks the better → a win is recorded. We fit a Bradley-Terry/Elo rating
(MAP, regularised toward a public LMArena seed prior) so each model gets a
"how good is it *on our work*" score anchored to an industry baseline.

This mirrors LMArena's methodology (Bradley-Terry MLE over pairwise votes) but on
our tasks with an LLM judge instead of human voters.

═══════════════════════════════════════════════════════════════════════════════
  SAFETY — spends real tokens (2 generations + 1 judge per match). GATED:
  default --dry-run; requires --run --confirm-spend AND NINEROUTER_KEY.
═══════════════════════════════════════════════════════════════════════════════
Output: web/public/data/model-benchmark-arena.json  (consumed by arena_overlay.py)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import aa_baseline  # noqa: E402
from roster import ROSTER  # noqa: E402
from live_eval_harness import LANE_TASKS, _parse_completion  # noqa: E402 (reuse prompts+parser)

REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
OUT_PATH = os.path.join(REPO_ROOT, "web", "public", "data", "model-benchmark-arena.json")
BASE_URL = os.environ.get("NINEROUTER_BASE_URL", "http://127.0.0.1:20128/v1")

# Default judge — a strong, available model not in the contestant pool ideally.
DEFAULT_JUDGE = "vertex/gemini-3.5-flash"

# Public LMArena Elo anchors (text arena, 2026-06 snapshot). Opus 4.8 = 1486 sourced;
# others are reasonable priors. Any model without an anchor derives its seed from
# the pinned AA v4.1 score (so every model has a sensible Bayesian prior).
LMARENA_ANCHORS = {
    "cc/claude-opus-4-8": 1486,
    "openrouter/anthropic/claude-fable-5": 1492,
    "cx/gpt-5.5": 1480,
    "vertex/gemini-3.1-pro-preview": 1476,
    "cc/claude-opus-4-7": 1470,
    "vertex/gemini-3.5-flash": 1462,
    "cx/gpt-5.4": 1460,
    "openrouter/x-ai/grok-4.3": 1450,
    "cc/claude-sonnet-4-6": 1448,
}
LMARENA_ASOF = "2026-06"


def _disp_by_router():
    return {m.get("router"): m["display"] for m in ROSTER if m.get("router")}


def seed_elo(rid, display):
    """LMArena anchor if known, else derived from the pinned AA v4.1 score."""
    if rid in LMARENA_ANCHORS:
        return float(LMARENA_ANCHORS[rid])
    score, _ = aa_baseline.lookup(display)
    if score is None:
        score = 25.0
    return round(1300.0 + score * 3.5, 1)   # AA 55→1492, 30→1405, 16→1356


def _call(rid, prompt, key, timeout=90, max_tokens=600):
    body = json.dumps({
        "model": rid, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text, _ = _parse_completion(resp.read().decode("utf-8", "replace"))
        return text or ""
    except (urllib.error.HTTPError, Exception):  # noqa: BLE001
        return ""


JUDGE_TMPL = (
    "You are a strict technical evaluator. A user gave this task:\n\n<task>\n{task}\n</task>\n\n"
    "Two assistants answered. Judge ONLY on CORRECTNESS and completeness — does the "
    "answer actually work and handle the edge cases the task implies? Ignore length, "
    "style, and verbosity. If one answer is correct and the other has a bug, misses an "
    "edge case, or is wrong, the correct one wins. Reply TIE ONLY if both are equally "
    "correct (or equally wrong).\n\n"
    "<assistant_A>\n{a}\n</assistant_A>\n\n<assistant_B>\n{b}\n</assistant_B>\n\n"
    "Reply with exactly one token: A, B, or TIE."
)

# HARD, correctness-checkable tasks that actually separate a flagship from a mini
# (subtle bugs, edge cases, O(n) constraints, tricky logic). Replaces the short
# lane prompts, which lacked discriminating power (see vault run 462/this run).
HARD_TASKS = {
    "median_bug": "This function should return the median of a list of numbers but is buggy:\n\ndef median(xs):\n    xs.sort()\n    return xs[len(xs)//2]\n\nList every bug and give a corrected version. (Hint: consider even-length lists, the empty list, and whether the caller's list should be mutated.) Return the fixed code and a one-line note per bug.",
    "race_condition": "Under multiple threads this loses increments:\n\ncounter = 0\ndef worker():\n    global counter\n    for _ in range(100000):\n        counter += 1\n\nExplain precisely WHY (one sentence) and give a corrected version that stays reasonably efficient (don't just wrap the whole loop in one lock if avoidable). Return the explanation and the fixed code.",
    "lru_cache": "Implement an LRU cache class `LRU(capacity)` with TRUE O(1) `get(key)` and `put(key, value)` (evicting the least-recently-used on overflow). Return only the code.",
    "parse_duration": "Write `parse_duration(s)` converting strings like '1h30m', '45s', '2d', '1h30m10s' to total seconds. Support units d/h/m/s in any combination, require at least one unit, and raise ValueError on malformed input such as '', '10', '1x', '1h1h' (repeated unit), or '1m30h' (out of order). Return only the code.",
    "second_highest_sql": "Write ONE ANSI-SQL query returning, for each department, the second-highest DISTINCT salary. Departments with fewer than 2 distinct salaries must still appear, with NULL. Tables: employees(id, dept_id, salary), departments(id, name). Return only the SQL.",
    "valid_ipv4": "Write `is_valid_ipv4(s) -> bool`. Valid = exactly four dot-separated decimal octets, each 0..255, with NO leading zeros (so '01' and '00' are invalid but '0' is valid) and no surrounding/embedded whitespace. Return only the code.",
    "dedup_order": "Refactor this O(n^2) function to O(n) average time while preserving first-occurrence order, and note one input type it would break on:\n\ndef dedup(xs):\n    out = []\n    for x in xs:\n        if x not in out:\n            out.append(x)\n    return out\n\nReturn the code and the one-line caveat.",
    "logic_puzzle": "Alice, Bob, and Carol each ALWAYS lie or ALWAYS tell the truth. Alice says 'Bob is a liar.' Bob says 'Carol is a liar.' Carol says 'Alice and Bob are both liars.' Determine, with reasoning, exactly who is a truth-teller and who is a liar.",
}

# Multi-judge panel (strong, mixed families to cancel single-judge bias). First two
# judge every match; a 3rd breaks disagreements. Any judge that is a contestant IN a
# given match is dropped from that match's panel.
JUDGE_PANEL = ["cc/claude-opus-4-8", "cx/gpt-5.4", "vertex/gemini-3.1-pro-preview"]

# Active task set (set by main()).
ACTIVE_TASKS = dict(LANE_TASKS)


def judge_match(task, out_a, out_b, judge, key, timeout=90):
    """Return 'A', 'B', or 'TIE'. Caller handles position randomisation."""
    if not out_a and not out_b:
        return "TIE"
    if not out_a:
        return "B"
    if not out_b:
        return "A"
    prompt = JUDGE_TMPL.format(task=task, a=out_a[:4000], b=out_b[:4000])
    verdict = _call(judge, prompt, key, timeout=timeout, max_tokens=8).strip().upper()
    for tok in ("TIE", "A", "B"):
        if tok in verdict:
            return tok
    return "TIE"


def judge_panel(task, out_a, out_b, judges, key):
    """2 judges vote; a 3rd breaks disagreements. Returns 'A'/'B'/'TIE'."""
    if not judges:
        return "TIE"
    if len(judges) == 1:
        return judge_match(task, out_a, out_b, judges[0], key)
    v1 = judge_match(task, out_a, out_b, judges[0], key)
    v2 = judge_match(task, out_a, out_b, judges[1], key)
    if v1 == v2:
        return v1
    if len(judges) >= 3:                       # disagreement → tiebreaker
        votes = [v1, v2, judge_match(task, out_a, out_b, judges[2], key)]
        for tok in ("A", "B"):
            if votes.count(tok) >= 2:
                return tok
        return "TIE"
    return "TIE"                                # 2 judges split, no tiebreaker


def run_match(rid_i, rid_j, task_key, judges, key, rng):
    """Blind A/B on ACTIVE_TASKS[task_key]: 1 if i beats j, 0 if j, 0.5 tie."""
    task = ACTIVE_TASKS.get(task_key)
    out_i = _call(rid_i, task, key)
    out_j = _call(rid_j, task, key)
    swap = rng.random() < 0.5     # randomise which is shown as "A"
    a, b = (out_j, out_i) if swap else (out_i, out_j)
    panel = [jd for jd in judges if jd not in (rid_i, rid_j)] or judges
    v = judge_panel(task, a, b, panel, key)
    if v == "TIE":
        return 0.5
    win_a = (v == "A")
    i_won = (win_a != swap)       # if swapped, "A" is j
    return 1.0 if i_won else 0.0


def schedule(rids, lanes, rng, extra_cross=2):
    """Adjacent pairs per lane (Swiss-ish) + a few random cross pairs."""
    pairs = []
    for lane in lanes:
        for k in range(len(rids) - 1):
            pairs.append((lane, rids[k], rids[k + 1]))
        for _ in range(extra_cross):
            a, b = rng.sample(rids, 2)
            pairs.append((lane, a, b))
    return pairs


def schedule_full(rids_by_strength, lanes, rng, knn=2, anchors=2):
    """Connected full-roster schedule that bounds cost: each model plays its
    `knn` nearest-by-strength neighbours (informative, uncertain matches) plus
    `anchors` random distant opponents (global calibration). rids must be sorted
    strongest→weakest. ~len(rids)*(knn+anchors) matches; each model gets ≈
    2*knn neighbour matches + ~2*anchors anchor matches → comfortably ≥6.
    Lanes are spread across pairs for task variety.
    """
    pairs, n = [], len(rids_by_strength)
    li = 0
    for idx, a in enumerate(rids_by_strength):
        for d in range(1, knn + 1):                 # forward neighbours (back side comes from others)
            if idx + d < n:
                pairs.append((lanes[li % len(lanes)], a, rids_by_strength[idx + d])); li += 1
        for _ in range(anchors):                    # random distant opponents
            j = rng.randrange(n)
            if j != idx:
                pairs.append((lanes[li % len(lanes)], a, rids_by_strength[j])); li += 1
    rng.shuffle(pairs)
    return pairs


def fit_elo(seed, results, passes=120, K=8, prior_pull=0.06, cap=130.0):
    """MAP Elo over ALL accumulated match results, regularised toward the seed
    prior. Sparse data → ratings stay near seed (low confidence); evidence earns
    movement. `cap` bounds deviation from seed so a tiny near-sweep can't blow up.
    """
    elo = {k: float(v) for k, v in seed.items()}
    for _ in range(passes):
        for (i, j, s) in results:    # s = score for i (1/0.5/0)
            if i not in elo or j not in elo:
                continue
            pi = 1.0 / (1.0 + 10 ** (-(elo[i] - elo[j]) / 400.0))
            elo[i] += K * (s - pi)
            elo[j] -= K * (s - pi)
        for k in elo:                # pull toward the seed prior each pass
            elo[k] += prior_pull * (seed[k] - elo[k])
    for k in elo:                    # sparse-data guard
        dev = max(-cap, min(cap, elo[k] - seed[k]))
        elo[k] = seed[k] + dev
    return {k: round(v, 1) for k, v in elo.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="", help="comma list of router ids (contestants)")
    ap.add_argument("--full", action="store_true", help="auto-select ALL eligible (available, chat, non-PRC) roster models")
    ap.add_argument("--taskset", default="hard", choices=["hard", "lanes"], help="hard differentiating tasks (default) or short lane prompts")
    ap.add_argument("--lanes", default="builder,researcher,maintainer", help="(--taskset lanes) comma list of lane keys")
    ap.add_argument("--judges", default=",".join(JUDGE_PANEL), help="comma list judge panel (2 vote + 3rd tiebreak; contestant-judges dropped per match)")
    ap.add_argument("--knn", type=int, default=3, help="(--full) nearest-strength opponents per model")
    ap.add_argument("--anchors", type=int, default=3, help="(--full) random distant opponents per model")
    ap.add_argument("--rounds", type=int, default=1, help="repeat the full schedule N times")
    ap.add_argument("--seed-rng", type=int, default=11)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--confirm-spend", action="store_true")
    ap.add_argument("--timeout", type=int, default=90)
    args = ap.parse_args()

    global ACTIVE_TASKS
    disp = _disp_by_router()
    rng = random.Random(args.seed_rng)
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    if args.taskset == "hard":
        ACTIVE_TASKS = dict(HARD_TASKS)
        task_keys = list(HARD_TASKS.keys())
    else:
        ACTIVE_TASKS = dict(LANE_TASKS)
        task_keys = [l.strip() for l in args.lanes.split(",") if l.strip() and l.strip() in LANE_TASKS]

    if args.full:
        # eligible = available, chat-capable, clean jurisdiction; sorted strongest→weakest
        elig = [m for m in ROSTER if m.get("router") and m.get("avail", True)
                and m["authority"] != "none" and not m.get("prc") and not m.get("trains")]
        rids = [m["router"] for m in sorted(elig, key=lambda m: -seed_elo(m["router"], m["display"]))]
        pairs = schedule_full(rids, task_keys, rng, knn=args.knn, anchors=args.anchors)
    else:
        rids = [m.strip() for m in args.models.split(",") if m.strip()]
        pairs = []
        for _ in range(args.rounds):
            pairs += schedule(rids, task_keys, rng)
    if len(rids) < 2:
        print("REFUSING: need >=2 contestants."); sys.exit(2)
    seeds = {r: seed_elo(r, disp.get(r, r)) for r in rids}

    n_calls = len(pairs) * 2 + len(pairs) * 2   # 2 gens + ~2 judges (3rd only on disagreement)
    print(f"Contestants ({len(rids)}) · taskset={args.taskset} ({len(task_keys)} tasks)")
    print(f"Judges: {', '.join(judges)}   Matches: {len(pairs)}  → ~{n_calls}-{len(pairs)*5} calls")
    if not (args.run and args.confirm_spend):
        print("\nSeed Elo (LMArena anchor or AA-derived):")
        for r in rids:
            print(f"  {r:42} {seeds[r]}  ({'LMArena' if r in LMARENA_ANCHORS else 'AA-derived'})")
        print("\nDRY-RUN — re-run with --run --confirm-spend (NINEROUTER_KEY set).")
        return
    key = os.environ.get("NINEROUTER_KEY")
    if not key:
        print("\nREFUSING: NINEROUTER_KEY not in environment."); sys.exit(2)

    workers = 6 if args.full else 1
    print(f"\nRUNNING {len(pairs)} matches… ({'concurrent x%d' % workers if workers > 1 else 'sequential'})")

    def _do(n, tk, i, j):
        mrng = random.Random(args.seed_rng * 100003 + n)   # per-match RNG (thread-safe)
        return n, i, j, run_match(i, j, tk, judges, key, mrng), tk

    slots = [None] * len(pairs)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_do, n, tk, i, j) for n, (tk, i, j) in enumerate(pairs)]
        done = 0
        for f in as_completed(futs):
            n, i, j, s, tk = f.result()
            slots[n] = [i, j, s]
            done += 1
            print(f"  [{done:>3}/{len(pairs)}] {tk:18} {disp.get(i,i)[:16]:16} vs {disp.get(j,j)[:16]:16} → "
                  f"{'i' if s==1 else ('j' if s==0 else 'tie')}")
    new_results = [r for r in slots if r]

    # Accumulate raw match history so Elo is always fit on EVERYTHING we've run.
    prev_results, prev_seeds = [], {}
    if os.path.exists(OUT_PATH):
        try:
            with open(OUT_PATH, encoding="utf-8") as fh:
                prev = json.load(fh)
            prev_results = prev.get("results", []) or []
            prev_seeds = {r: v.get("seed_elo") for r, v in (prev.get("ratings", {}) or {}).items()}
        except (OSError, ValueError):
            pass
    all_results = [tuple(r) for r in prev_results] + [tuple(r) for r in new_results]

    # Universe of all rated models (seeds for everyone who appears anywhere).
    all_rids = set(seeds) | {r for m in all_results for r in (m[0], m[1])}
    full_seeds = {r: seeds.get(r) or prev_seeds.get(r) or seed_elo(r, disp.get(r, r)) for r in all_rids}
    elo = fit_elo(full_seeds, all_results)

    played = {r: 0 for r in all_rids}
    wins = {r: 0.0 for r in all_rids}
    for (i, j, s) in all_results:
        played[i] += 1; played[j] += 1
        wins[i] += s; wins[j] += (1 - s)
    ratings = {r: {
        "display": disp.get(r, r), "elo": elo[r], "seed_elo": round(full_seeds[r], 1),
        "matches": played[r], "wins": round(wins[r], 1),
        "delta_vs_seed": round(elo[r] - full_seeds[r], 1),
    } for r in all_rids}
    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "method": "Bradley-Terry/Elo MAP over accumulated blind A/B matches on hard differentiating tasks; 2-judge panel + tiebreaker (contestant-judges dropped); seed = LMArena anchor or AA-v4.1-derived prior.",
            "judge": " / ".join(judges), "taskset": args.taskset,
            "lanes": task_keys,
            "matches_total": len(all_results), "matches_this_run": len(new_results),
            "lmarena_asof": LMARENA_ASOF, "aa_index_version": aa_baseline.AA_VERSION,
        },
        "ratings": ratings,
        "results": [list(r) for r in all_results],
    }
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nHermes Arena Elo (fit on {len(all_results)} total matches):")
    for r in sorted(ratings, key=lambda x: -ratings[x]["elo"]):
        rec = ratings[r]
        print(f"  {rec['elo']:>7}  ({rec['delta_vs_seed']:+6.1f} vs seed) {rec['matches']:>2}m  {rec['display']}")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
