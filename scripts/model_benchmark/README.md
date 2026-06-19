# Hermes Model Benchmark

A data-driven (+ optional live) benchmark of **every model the fleet can reach**
scored against **every Hermes lane/task** (the main agent, all `auxiliary:`
slots, and the kanban worker roles). It produces:

* `web/public/data/model-benchmark.json` — the dataset the webui **Benchmark**
  page renders (per-model dimensions + per-lane recommendations & fallbacks).
* per-lane **best model + fallback** recommendations, ranked by a transparent,
  editable scoring model.

It is the analysis behind the model-routing recommendations in
`vault/runs/2026-06-16-434-model-benchmark-optimization.md`. **Nothing here is
auto-applied to routing** — recommendations only.

## Layout

| File | Role |
|---|---|
| `roster.py` | Authoritative model roster, transcribed from the conductor-vault `wiki/models/matrices/*.md` (the fleet source of truth), with source notes + estimate flags. |
| `lanes.py` | Every Hermes lane: current wiring, output authority, per-lane scoring **weights** + **hard constraints** + `sensitive` policy flag. This is the benchmark's opinion — tune it here. |
| `build_benchmark.py` | Normalises dims 0..100, scores each model per lane after hard constraints, applies the policy penalty, emits the JSON. `--verify` drift-checks the vault. |
| `live_eval_harness.py` | **Gated** live eval — fires lane-representative prompts through the 9Router and records measured latency/throughput/cost. Dry-run by default. |

## Methodology (data-driven layer)

Seven dimensions, each normalised to 0..100:

`intelligence` (Artificial Analysis Intelligence Index where available, else a
documented capability estimate) · `cost` (inverse log of output-weighted
$/MTok) · `context` (log-scaled window) · `tool_use` (function-calling/agentic
tier) · `speed` (tier/family estimate until the live harness measures it) ·
`vision` (image-input quality, 0 if not capable) · `privacy` (data-handling
posture; OpenRouter passthrough of clean vendors is down-weighted).

Per lane: `score = weighted_mean(dims, lane.weights)` over models that pass the
lane's **hard constraints** (e.g. `require_vision`, `long_context_min`,
`require_authority: binding` for the reviewer, `exclude_trains` for the
privacy-critical km-agent).

**Policy layer.** Lanes flagged `sensitive` (maker / review / knowledge / lanes
that touch customer code or durable memory) apply a documented multiplicative
penalty to `policy_score`: PRC providers ×0.88, providers that train on inputs
×0.55. This mirrors the fleet posture of leading sensitive lanes with
clean-vendor models while keeping PRC models as ranked fallbacks. Recommendations
rank by `policy_score`; the raw `score` is also exposed so you can see the
data-optimal pick vs the policy-adjusted one.

## Run it

```bash
cd ~/.hermes/hermes-agent
python3 scripts/model_benchmark/build_benchmark.py --verify
```

`--verify` re-parses the live vault matrices and warns on any drift between them
and `roster.py`, so the cache stays honest as the vault evolves. Re-run after
editing `roster.py` or `lanes.py`; the webui page picks up the new JSON on
reload (after a `web` build for production).

## Live eval (gated — spends real tokens)

The live harness is **manually gated** per fleet standing rules. It is dry-run by
default and refuses to make calls without explicit confirmation:

```bash
# dry-run: prints the call plan + estimated spend, ZERO network calls
python3 scripts/model_benchmark/live_eval_harness.py --lanes title_generation

# execute (requires BOTH flags AND NINEROUTER_KEY in env)
python3 scripts/model_benchmark/live_eval_harness.py \
    --lanes title_generation --models "vertex/gemini-2.5-flash-lite,openrouter/deepseek/deepseek-v4-flash" \
    --run --confirm-spend
```

It writes `web/public/data/model-benchmark-measured.json`; a future
`build_benchmark.py` pass can merge those measured latency/throughput numbers in
as a `measured` block (replacing the `speed` estimate with real data).

## Editing the benchmark's opinion

* **Re-weight a lane** → edit `lanes.py` `weights`, re-run the builder.
* **Add/remove a hard rule** → `lanes.py` `hard` (e.g. raise `long_context_min`).
* **Change a model's facts** → `roster.py` (then `--verify` against the vault).
* **Tune the policy penalty** → `PRC_PENALTY` / `TRAINS_PENALTY` in
  `build_benchmark.py` and the `_SENSITIVE` set in `lanes.py`.

Every knob is local and transparent; there is no hidden model.
