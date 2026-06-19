#!/usr/bin/env python3
"""Hermes model benchmark — LIVE evaluation harness (GATED, never auto-runs).

This is the *measured* half of the hybrid benchmark. It fires lane-representative
prompts at real models through the local 9Router and records:

* time-to-first-token + total latency
* output throughput (tokens/sec)
* token counts → measured $ cost (joined with roster cost rates)
* (optional, --judge) an LLM-graded quality score per task

It then writes ``web/public/data/model-benchmark-measured.json``, which
``build_benchmark.py`` will merge as a ``measured`` block on each model (so the
webui page can show data-driven *and* measured dims side by side).

═══════════════════════════════════════════════════════════════════════════════
  SAFETY — READ THIS
═══════════════════════════════════════════════════════════════════════════════
This harness SPENDS REAL TOKENS against credentialed providers via the 9Router.
Per fleet standing rules, cost-bearing automation is MANUALLY GATED. Therefore:

  * Default action is --dry-run: it prints the exact call plan and the estimated
    spend, and makes ZERO network calls.
  * To actually run you must pass BOTH --run AND --confirm-spend, AND the
    NINEROUTER_KEY env var must be present. Missing any → refuses.
  * --judge (LLM grading) spends additional tokens and is off by default.
  * It never runs unattended, never writes to config.yaml, never mirrors to
    Morgan, and never touches credentials beyond reading NINEROUTER_KEY from env.

Recommended first real run: a single cheap lane + 2 models, e.g.
    python live_eval_harness.py --lanes title_generation --models \
        "vertex/gemini-2.5-flash-lite,openrouter/deepseek/deepseek-v4-flash" \
        --run --confirm-spend
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from roster import ROSTER  # noqa: E402
from lanes import LANES_BY_KEY  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
OUT_PATH = os.path.join(REPO_ROOT, "web", "public", "data", "model-benchmark-measured.json")
BASE_URL = os.environ.get("NINEROUTER_BASE_URL", "http://127.0.0.1:20128/v1")

# Representative task per lane: a prompt that exercises the lane's real skill.
# Kept short to bound spend; expand locally as needed.
LANE_TASKS = {
    "default_chat": "In 3 sentences, explain what a kanban WIP limit is and why it matters.",
    "orchestrator": "You are a PM. Decompose 'add CSV export to the reports page' into an ordered task list of 4-6 steps. Return a numbered list only.",
    "builder": "Write a Python function `slugify(s)` that lowercases, trims, and replaces non-alphanumeric runs with single hyphens. Return only the code.",
    "maintainer": "This function has a bug: `def avg(xs): return sum(xs)/len(xs)`. Make it safe for empty input and explain the fix in one line.",
    "researcher": "List 3 distinct trade-offs between server-side and client-side rendering. One line each, no preamble.",
    "km_agent": "Summarise this into 2 bullet points for a knowledge base: 'The gateway restart now auto-repairs the SOUL identity loader; webui restarts are coupled to the gateway.'",
    "ops_watch": "Given log line 'ERROR db pool exhausted (50/50) for 90s', classify severity as INFO/WARN/CRIT and give a one-line reason.",
    "qa_functional": "Given acceptance criterion 'user can reset password via emailed link', list 3 functional test cases. One line each.",
    "triage_specifier": "Flesh out this ticket into a spec with Goal/Scope/Acceptance: 'dark mode toggle'. Keep under 120 words.",
    "kanban_decomposer": "Decompose epic 'user profile page' into 3-5 independently shippable tasks. Return a numbered list with a one-line acceptance per task.",
    "title_generation": "Generate a <=6 word title for a chat about debugging a flaky CI test. Title only.",
    "compression": "Compress to <=40 words, preserving facts: 'The team discussed migrating from REST to GraphQL, weighed caching complexity, agreed to pilot one endpoint first, and deferred the auth changes to next quarter.'",
}


def router_to_model():
    return {m["router"]: m for m in ROSTER if m.get("router")}


def estimate_spend(plan):
    """Rough $ estimate using roster cost rates and a token guess per task."""
    r2m = router_to_model()
    total = 0.0
    for rid, _task in plan:
        m = r2m.get(rid)
        if not m or m.get("cin") is None:
            continue
        # assume ~300 in / ~250 out tokens for these short tasks
        total += (300 * m["cin"] + 250 * m["cout"]) / 1_000_000
    return round(total, 5)


def build_plan(lane_keys, model_filter):
    plan = []
    for lk in lane_keys:
        task = LANE_TASKS.get(lk)
        if not task:
            continue
        lane = LANES_BY_KEY.get(lk)
        # candidate models = lane current primary + fallbacks, or an explicit filter
        if model_filter:
            rids = model_filter
        else:
            rids = [lane["current_primary"], *lane["current_fallbacks"]]
            rids = [r for r in rids if r and r != "auto"]
        for rid in rids:
            plan.append((rid, task))
    return plan


def _parse_completion(raw: str):
    """Extract (text, usage) from a chat-completions body that may be plain
    JSON, JSON-with-trailing-data, or an SSE stream. Returns (text, usage|{})."""
    raw = (raw or "").strip()
    if not raw:
        return "", {}
    # SSE: lines of `data: {...}` (+ optional `data: [DONE]`).
    if raw.startswith("data:") or "\ndata:" in raw:
        text, usage = "", {}
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]" or not payload:
                continue
            try:
                obj = json.loads(payload)
            except ValueError:
                continue
            ch = (obj.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            text += delta.get("content") or ch.get("message", {}).get("content") or ""
            if obj.get("usage"):
                usage = obj["usage"]
        return text, usage
    # Plain JSON, possibly with trailing data → decode just the first object.
    try:
        obj, _end = json.JSONDecoder().raw_decode(raw)
    except ValueError:
        return "", {}
    ch = (obj.get("choices") or [{}])[0]
    text = ch.get("message", {}).get("content") or ch.get("text") or ""
    return text, (obj.get("usage") or {})


def call_model(rid, prompt, key, timeout=120):
    """One STREAMING chat completion through the 9Router. Returns metrics.

    Streams so we can capture time-to-first-token (TTFT). The reported
    ``throughput_tps`` is GENERATION-ONLY: output_tokens / (total − TTFT). This
    isolates raw token-emission speed from prompt-processing + hidden reasoning,
    which a reasoning model (e.g. GPT-5.5) burns *before* the first visible token.
    Measuring end-to-end (output / total) unfairly tanks such models on short
    tasks where that startup cost dominates. ``effective_tps`` (output / total) is
    also returned for reference. (2026-06-17)

    Robust to providers that ignore ``stream`` and return one JSON body (then we
    fall back to whole-body parse with no TTFT → gen-rate == effective rate).
    Sends NO ``temperature`` — newer flagships reject it (400 + 30s breaker).
    """
    import urllib.request
    import urllib.error

    body = json.dumps({
        "model": rid,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    t0 = time.monotonic()
    t_first = t_last = None
    n_chunks = 0
    text_parts, raw_all, usage = [], [], {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                s = raw_line.decode("utf-8", "replace")
                raw_all.append(s)
                line = s.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]" or not payload:
                    continue
                try:
                    obj = json.loads(payload)
                except ValueError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                ch = (obj.get("choices") or [{}])
                delta = (ch[0].get("delta") if ch else {}) or {}
                piece = delta.get("content") or ""
                if piece:
                    now = time.monotonic()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    n_chunks += 1
                    text_parts.append(piece)
        t_end = time.monotonic()
        total = t_end - t0
        text = "".join(text_parts)
        # Provider ignored stream → parse the whole body, no per-token timing.
        if not text and t_first is None:
            text, usage = _parse_completion("".join(raw_all))
        out_tok = usage.get("completion_tokens")
        est = out_tok is None
        if est:  # estimate from characters when usage is absent
            out_tok = max(1, round(len(text) / 4)) if text else 0
        ttft = (t_first - t0) if t_first is not None else None
        effective = round(out_tok / total, 1) if total > 0 and out_tok else None

        # Generation-only rate = output / span over which tokens actually arrived
        # (t_last − t_first). Trust it ONLY on genuine incremental streaming:
        # ≥3 content chunks spanning a real interval. 9Router BUFFERS the stream
        # for some passthrough providers (Vertex/Gemini, OpenRouter) — the whole
        # body lands in one chunk, so the span collapses and the rate is garbage.
        # In that case (and as a sanity cap) fall back to the effective rate.
        gen_tps = None
        if t_first is not None and t_last is not None and n_chunks >= 3:
            span = t_last - t_first
            if span >= 0.15:
                cand = out_tok / span
                if cand <= 300:           # reject buffered/garbage spikes
                    gen_tps = round(cand, 1)
        throughput = gen_tps if gen_tps is not None else effective
        return {
            "ok": True, "latency_s": round(total, 3),
            "ttft_s": round(ttft, 3) if ttft is not None else None,
            "n_chunks": n_chunks,
            "streamed": gen_tps is not None,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": out_tok,
            "tokens_estimated": est,
            "throughput_tps": throughput,     # gen-only when truly streamed, else effective
            "gen_tps": gen_tps,
            "effective_tps": effective,
            "chars": len(text or ""),
        }
    except urllib.error.HTTPError as e:  # noqa: BLE001
        return {"ok": False, "error": f"HTTP {e.code}", "latency_s": round(time.monotonic() - t0, 3)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:120], "latency_s": round(time.monotonic() - t0, 3)}


# ── Two-request throughput method ─────────────────────────────────────────────
# gen-rate = Δoutput_tokens / Δlatency across two runs of the SAME long-output
# prompt at two max_tokens caps. The fixed per-call overhead (prompt processing +
# hidden reasoning + TTFT) is ~constant across the pair, so it cancels in the
# difference, leaving pure generation speed. Unlike the streaming method this is
# IMMUNE to 9Router stream-buffering — it's the right tool for buffered REASONING
# models (Gemini 3.x Pro), which streaming can't measure fairly. (2026-06-17)
TWO_REQ_PROMPT = (
    "Write a thorough, multi-paragraph technical explanation of how TCP congestion "
    "control works. Cover slow start, congestion avoidance, fast retransmit, fast "
    "recovery, and modern algorithms (CUBIC, BBR). Be detailed and complete — aim "
    "for several hundred words of prose."
)


def _single_call(rid, key, max_tokens, timeout):
    """One non-streaming call; returns (latency_s, output_tokens) or None."""
    import urllib.request
    import urllib.error
    body = json.dumps({
        "model": rid,
        "messages": [{"role": "user", "content": TWO_REQ_PROMPT}],
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
    except (urllib.error.HTTPError, Exception):  # noqa: BLE001
        return None
    dt = time.monotonic() - t0
    text, usage = _parse_completion(raw)
    out_tok = usage.get("completion_tokens")
    if out_tok is None:
        out_tok = max(1, round(len(text) / 4)) if text else 0
    return (dt, out_tok)


def two_request_rate(rid, key, timeout=120, pairs=3, lo=64, hi=384):
    """Median gen-rate (tok/s) from `pairs` low/high max_tokens pairs."""
    rates, raw = [], []
    for _ in range(pairs):
        a = _single_call(rid, key, lo, timeout)
        b = _single_call(rid, key, hi, timeout)
        if not a or not b:
            continue
        (la, ta), (lb, tb) = a, b
        raw.append({"lo": [round(la, 3), ta], "hi": [round(lb, 3), tb]})
        if lb > la and tb > ta:
            rates.append((tb - ta) / (lb - la))
    if not rates:
        return None
    rates.sort()
    return {
        "throughput_mean_tps": round(rates[len(rates) // 2], 1),  # median, fed to speed dim
        "method": "two-request", "ok": True,
        "runs": len(rates), "pairs_raw": raw,
    }


def e2e_rate(rid, key, timeout=120, runs=3, max_tokens=512):
    """END-TO-END throughput on one substantial task = output_tokens / total_latency
    at a fixed large max_tokens. Thinking IS counted (unlike two-request/gen-only)
    but amortized over ~hundreds of tokens, so a reasoning model isn't crushed by
    short-task overhead NOR credited as 'fastest' for burst speed it can't sustain
    end-to-end. This is the speed-dim source. (operator choice 2026-06-17)"""
    rates, lats, toks = [], [], []
    for _ in range(runs):
        r = _single_call(rid, key, max_tokens, timeout)  # uses TWO_REQ_PROMPT
        if not r:
            continue
        lat, tok = r
        if lat > 0 and tok > 0:
            rates.append(tok / lat)
            lats.append(lat)
            toks.append(tok)
    if not rates:
        return None
    rates.sort(); lats.sort(); toks.sort()
    return {
        "throughput_mean_tps": round(rates[len(rates) // 2], 1),  # median e2e tps → speed dim
        "effective_tps_mean": round(rates[len(rates) // 2], 1),
        "method": "end-to-end-512tok", "ok": True, "runs": len(rates),
        "latency_median_s": round(lats[len(lats) // 2], 2),
        "out_tok_median": toks[len(toks) // 2],
    }


def _run_e2e(args):
    rids = [m.strip() for m in args.models.split(",") if m.strip()]
    if not rids:
        print("REFUSING: --e2e needs --models <router ids>.")
        sys.exit(2)
    print(f"End-to-end speed method (512-token standard task) · models: {', '.join(rids)}")
    print(f"Planned calls: {len(rids) * 3}  (3 runs per model)")
    if not (args.run and args.confirm_spend):
        print("\nDRY-RUN — re-run with --run --confirm-spend (NINEROUTER_KEY set).")
        return
    key = os.environ.get("NINEROUTER_KEY")
    if not key:
        print("\nREFUSING: NINEROUTER_KEY not in environment. No calls made.")
        sys.exit(2)
    print("\nEXECUTING end-to-end measurements…")
    measured = {}
    for rid in rids:
        res = e2e_rate(rid, key, timeout=args.timeout)
        if res:
            measured[rid] = res
            print(f"  {rid:<42} {res['throughput_mean_tps']} tps e2e"
                  f"  ({res['out_tok_median']} tok / {res['latency_median_s']}s, {res['runs']} runs)")
        else:
            print(f"  {rid:<42} FAILED (no valid runs)")
    if measured:
        _merge_measured_into_json(measured)
    else:
        print("No measurements produced; nothing written.")


def _merge_measured_into_json(new_measured):
    """Accumulate new per-rid measurements into the measured JSON (replace per rid)."""
    prev_measured, prev_raw = {}, {}
    if os.path.exists(OUT_PATH):
        try:
            with open(OUT_PATH, encoding="utf-8") as fh:
                prev = json.load(fh)
            prev_measured = prev.get("measured", {}) or {}
            prev_raw = prev.get("raw", {}) or {}
        except (OSError, ValueError):
            pass
    prev_measured.update(new_measured)
    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "base_url": BASE_URL,
            "note": "Measured throughput via 9Router (accumulated). Entries with method=two-request use Δtokens/Δlatency (immune to stream buffering).",
        },
        "measured": prev_measured,
        "raw": prev_raw,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {OUT_PATH} ({len(prev_measured)} models total)")


def _run_two_req(args):
    rids = [m.strip() for m in args.models.split(",") if m.strip()]
    if not rids:
        print("REFUSING: --two-req needs --models <router ids>.")
        sys.exit(2)
    print(f"Two-request speed method · models: {', '.join(rids)}")
    print(f"Planned calls: {len(rids) * 3 * 2}  (3 pairs × 2 max_tokens caps per model)")
    if not (args.run and args.confirm_spend):
        print("\nDRY-RUN — re-run with --run --confirm-spend (NINEROUTER_KEY set).")
        return
    key = os.environ.get("NINEROUTER_KEY")
    if not key:
        print("\nREFUSING: NINEROUTER_KEY not in environment. No calls made.")
        sys.exit(2)
    print("\nEXECUTING two-request measurements…")
    measured = {}
    for rid in rids:
        res = two_request_rate(rid, key, timeout=args.timeout)
        if res:
            measured[rid] = {k: v for k, v in res.items() if k != "pairs_raw"}
            print(f"  {rid:<42} gen {res['throughput_mean_tps']} tps  ({res['runs']} valid pairs)")
        else:
            print(f"  {rid:<42} FAILED (no valid pairs — output may not exceed the low cap)")
    if measured:
        _merge_measured_into_json(measured)
    else:
        print("No measurements produced; nothing written.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lanes", default="", help="comma list of lane keys (default: all with tasks)")
    ap.add_argument("--models", default="", help="comma list of router ids to test on every lane (overrides lane defaults)")
    ap.add_argument("--run", action="store_true", help="actually make network calls (else dry-run)")
    ap.add_argument("--confirm-spend", action="store_true", help="required alongside --run; acknowledges token spend")
    ap.add_argument("--judge", action="store_true", help="LLM-grade each output (extra spend; not yet implemented)")
    ap.add_argument("--two-req", action="store_true", help="measure raw generation speed via two-request Δtokens/Δlatency (diagnostic; excludes thinking); uses --models")
    ap.add_argument("--e2e", action="store_true", help="measure END-TO-END speed on a 512-token standard task (the speed-dim source); uses --models")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    if args.e2e:
        return _run_e2e(args)
    if args.two_req:
        return _run_two_req(args)

    lane_keys = [k.strip() for k in args.lanes.split(",") if k.strip()] or list(LANE_TASKS.keys())
    model_filter = [m.strip() for m in args.models.split(",") if m.strip()]
    plan = build_plan(lane_keys, model_filter)
    est = estimate_spend(plan)

    print(f"Lanes: {', '.join(lane_keys)}")
    print(f"Planned calls: {len(plan)}   estimated spend: ~${est} (short-task heuristic)")
    if args.judge:
        print("  --judge requested: NOT YET IMPLEMENTED (would add ~1 grading call per task).")

    if not (args.run and args.confirm_spend):
        print("\nDRY-RUN — no network calls made. Call plan:")
        for rid, task in plan:
            print(f"  {rid:<42} ⇐ {task[:60]}…")
        print("\nTo execute: re-run with --run --confirm-spend (and NINEROUTER_KEY set).")
        return

    key = os.environ.get("NINEROUTER_KEY")
    if not key:
        print("\nREFUSING: NINEROUTER_KEY not in environment. No calls made.")
        sys.exit(2)

    print("\nEXECUTING live calls through the 9Router…")
    results = {}
    for rid, task in plan:
        metrics = call_model(rid, task, key, timeout=args.timeout)
        results.setdefault(rid, []).append(metrics)
        status = "ok" if metrics.get("ok") else f"ERR {metrics.get('error','')}"
        ttft = metrics.get("ttft_s")
        mode = "gen" if metrics.get("streamed") else "eff"
        print(f"  {rid:<42} {metrics.get('latency_s')}s  {metrics.get('throughput_tps') or '-'} tps[{mode}]"
              f"  (gen {metrics.get('gen_tps') or '-'}, eff {metrics.get('effective_tps') or '-'},"
              f" ttft {ttft if ttft is not None else '-'}s)  {status}")

    # Aggregate per model. throughput_mean_tps is GENERATION-ONLY (output / (total
    # − TTFT)); effective + TTFT are kept for transparency. build_benchmark maps
    # throughput_mean_tps → the speed dim.
    def _median(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else None

    measured = {}
    for rid, runs in results.items():
        ok = [r for r in runs if r.get("ok")]
        if not ok:
            measured[rid] = {"runs": len(runs), "ok": 0}
            continue
        tps = [r["throughput_tps"] for r in ok if r.get("throughput_tps")]
        eff = [r["effective_tps"] for r in ok if r.get("effective_tps")]
        ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
        measured[rid] = {
            "runs": len(runs), "ok": len(ok),
            "latency_median_s": _median([r["latency_s"] for r in ok]),
            "throughput_mean_tps": round(sum(tps) / len(tps), 1) if tps else None,
            "effective_tps_mean": round(sum(eff) / len(eff), 1) if eff else None,
            "ttft_median_s": _median(ttfts),
        }

    # Merge into any existing results so partial re-sweeps accumulate (a Gemini
    # re-run shouldn't wipe the other 26 models' measurements).
    prev_measured, prev_raw = {}, {}
    if os.path.exists(OUT_PATH):
        try:
            with open(OUT_PATH, encoding="utf-8") as fh:
                prev = json.load(fh)
            prev_measured = prev.get("measured", {}) or {}
            prev_raw = prev.get("raw", {}) or {}
        except (OSError, ValueError):
            pass
    prev_measured.update(measured)
    prev_raw.update(results)
    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "base_url": BASE_URL, "lanes": lane_keys, "calls": len(plan),
            "note": "Measured latency/throughput via 9Router (accumulated across runs). Merged into model-benchmark.json by build_benchmark.py.",
        },
        "measured": prev_measured,
        "raw": prev_raw,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {OUT_PATH} ({len(prev_measured)} models total)")


if __name__ == "__main__":
    main()
