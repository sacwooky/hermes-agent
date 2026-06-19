#!/usr/bin/env python3
"""Run the live model-benchmark sweep + rebuild, writing a status file the webui
polls. Launched detached by POST /api/model-benchmark/run-sweep.

SPENDS REAL TOKENS via the 9Router. Sweeps every reachable chat model once,
merges measured latency/throughput into the dataset, and rebuilds the benchmark
JSON (so `speed` becomes measured). Progress + result land in
``$HERMES_HOME/benchmark-sweep-status.json``.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
HERMES_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
STATUS_PATH = os.path.join(HERMES_HOME, "benchmark-sweep-status.json")


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_status(**kw):
    kw.setdefault("updated_at", _now())
    tmp = STATUS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(kw, fh)
    os.replace(tmp, STATUS_PATH)


def load_key():
    k = os.environ.get("NINEROUTER_KEY")
    if k:
        return k
    envp = os.path.join(HERMES_HOME, ".env")
    if os.path.exists(envp):
        for line in open(envp, encoding="utf-8"):
            if line.startswith("NINEROUTER_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    return None


def main():
    started = _now()
    try:
        import build_benchmark
        from live_eval_harness import LANE_TASKS, call_model, OUT_PATH

        key = load_key()
        if not key:
            write_status(state="error", error="NINEROUTER_KEY not found", started_at=started)
            return

        with open(build_benchmark.OUTPUT_PATH, encoding="utf-8") as fh:
            bench = json.load(fh)
        models = [
            m["router"] for m in bench["models"]
            if m.get("router") and m["output_authority"] != "none"
            and m["family"] != "openrouter" and m["available"]
        ]
        prompt = LANE_TASKS["default_chat"]
        total = len(models)
        write_status(state="running", started_at=started, total=total, done=0, ok=0, failed=0)

        results, ok, failed = {}, 0, 0
        for i, rid in enumerate(models):
            r = call_model(rid, prompt, key, timeout=60)
            results[rid] = [r]
            if r.get("ok"):
                ok += 1
            else:
                failed += 1
            write_status(state="running", started_at=started, total=total,
                         done=i + 1, ok=ok, failed=failed, current=rid)

        # Aggregate per model.
        measured = {}
        for rid, runs in results.items():
            okr = [x for x in runs if x.get("ok")]
            if not okr:
                measured[rid] = {"runs": len(runs), "ok": 0}
                continue
            lats = sorted(x["latency_s"] for x in okr)
            tps = [x["throughput_tps"] for x in okr if x.get("throughput_tps")]
            measured[rid] = {
                "runs": len(runs), "ok": len(okr),
                "latency_median_s": lats[len(lats) // 2],
                "throughput_mean_tps": round(sum(tps) / len(tps), 1) if tps else None,
            }

        # Merge with prior measurements so this run only updates what it swept.
        prev = {}
        if os.path.exists(OUT_PATH):
            try:
                with open(OUT_PATH, encoding="utf-8") as fh:
                    prev = json.load(fh).get("measured", {}) or {}
            except (OSError, ValueError):
                prev = {}
        prev.update(measured)
        with open(OUT_PATH, "w", encoding="utf-8") as fh:
            json.dump({"meta": {"generated_at": _now(), "note": "live sweep via webui"},
                       "measured": prev, "raw": results}, fh, indent=2)

        build_benchmark.build(verify=False)  # rebuild → merges measured speed
        write_status(state="done", started_at=started, finished_at=_now(),
                     total=total, done=total, ok=ok, failed=failed)
    except Exception as exc:  # noqa: BLE001
        write_status(state="error", started_at=started, error=str(exc)[:300])


if __name__ == "__main__":
    main()
