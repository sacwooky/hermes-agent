#!/usr/bin/env python3
"""Build the Hermes model-benchmark dataset (data-driven scoring).

Reads the authoritative roster (``roster.py``, transcribed from the vault
matrices) and the lane profiles (``lanes.py``), normalises every model onto a
common 0..100 dimension scale, scores each model against each lane (after hard
constraints), and emits a single JSON the webui page renders.

This is the DATA-DRIVEN half of the hybrid benchmark: every number traces to a
published model card / cost sheet / the vault matrices. The companion
``live_eval_harness.py`` (gated) measures real latency + task quality and merges
``measured`` dims into the same JSON — until then ``speed`` is a documented
estimate and quality == capability.

Usage
-----
    python scripts/model_benchmark/build_benchmark.py            # write JSON
    python scripts/model_benchmark/build_benchmark.py --verify   # + drift-check vault
    python scripts/model_benchmark/build_benchmark.py --stamp 2026-06-16T20:40:00Z

Output: web/public/data/model-benchmark.json (served at /data/model-benchmark.json)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from roster import ROSTER  # noqa: E402
from lanes import LANES, DIMENSIONS, WEIGHT_MODE  # noqa: E402
import aa_baseline  # noqa: E402  (pinned AA v4.1 intelligence baseline)
import arena_overlay  # noqa: E402  (Hermes Arena Elo overlay, Layer 3)

# Roster models with no aa_baseline entry (filled during dim computation).
AA_MISSING = []

REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
OUTPUT_PATH = os.path.join(REPO_ROOT, "web", "public", "data", "model-benchmark.json")
MEASURED_PATH = os.path.join(REPO_ROOT, "web", "public", "data", "model-benchmark-measured.json")
# Throughput (tok/s) that maps to a full speed score of 100.
SPEED_REF_TPS = 110.0
HERMES_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
# Dated snapshots live in HERMES_HOME (runtime-writable; survive web rebuilds) so
# the webui's date/time filter can load any past run. Newest kept; old pruned.
HISTORY_DIR = os.path.join(HERMES_HOME, "benchmark-history")
HISTORY_KEEP = 60
VAULT_MATRIX_DIR = os.environ.get(
    "HERMES_VAULT_MATRICES",
    "/srv/fluxlabs/vault/conductor-vault/wiki/models/matrices",
)

# ── Fixed categorical → numeric scales (transparent, editable) ─────────────
TOOL_SCORE = {
    "full": 95,      # Claude full tool use + server tools
    "native": 92,    # MiniMax / Kimi / DeepSeek / Qwen native function calling
    "gpt": 90,       # GPT function calling + structured outputs + hosted tools
    "gemini": 88,    # Gemini FC + structured outputs + code exec + grounding
    "strong": 88,    # explicitly "strong at reliable multi-tool calling"
    "glm": 80,       # GLM tool use via Claude-compat harness
    "tooluse": 76,   # generic "tool use"
    "none": 0,       # classifier / ASR
}

# Speed estimate (0..100, higher = faster). Data-driven from tier/family until
# the live harness supplies measured latency. Refined by name heuristics below.
SPEED_BASE = {
    "anthropic": 62, "openai": 60, "google": 78, "zai-glm": 64,
    "moonshot-kimi": 64, "minimax": 60, "deepseek": 66, "mistral": 64,
    "qwen": 66, "xiaomi": 70, "meta": 75, "nvidia": 70, "openrouter": 35,
}
SPEED_FAST_HINTS = ("flash-lite", "flash", "lite", "mini", "air", "haiku", "highspeed", "v4 flash")
SPEED_SLOW_HINTS = ("pro", "opus", "fusion", "deepseek v4 pro", "5.5-pro")

# Vision quality (0..100) by family, applied only when vision==True.
VISION_SCORE = {
    "google": 90, "minimax": 88, "xiaomi": 86, "zai-glm": 82, "qwen": 80,
    "anthropic": 80, "openai": 76, "mistral": 75, "moonshot-kimi": 74, "meta": 60,
}

# Privacy posture (0..100, higher = cleaner data handling) + label.
PRIVACY = {
    "anthropic": (95, "commercial terms"),
    "openai": (92, "API no-training default; abuse logs ≤30d"),
    "google": (90, "paid tier: no training, transient logging"),
    "mistral": (66, "EU (France); via OpenRouter — terms unverified"),
    "zai-glm": (55, "PRC; DPA: content not stored"),
    "minimax": (45, "PRC; carve-out (MR-6); retention unverified"),
    "deepseek": (35, "PRC; via OpenRouter — unverified"),
    "qwen": (35, "PRC (Alibaba); via OpenRouter — unverified"),
    "xiaomi": (35, "PRC; via OpenRouter — unverified"),
    "moonshot-kimi": (15, "⚠ trains on API inputs; no DPA"),
    "nvidia": (72, "US (NVIDIA); via OpenRouter — no training default"),
    "xai": (80, "US (xAI); via OpenRouter — no training default"),
    "amazon": (82, "US (AWS); no training default"),
    "cohere": (80, "Canada; no training default"),
    "meta": (78, "US (Meta open weights); via OpenRouter"),
    "openrouter": (50, "mixed-jurisdiction panel; unverified"),
}
# OpenRouter-passthrough Anthropic/OpenAI/Google routes lose the direct-vendor
# clearance — nudged down in compute_dims().


# Access / billing channel — how the fleet actually reaches + pays for a model.
# (name, billing-nature tooltip). Derived from the 9Router id prefix, else family.
def channel_for(router, family):
    r = router or ""
    if r.startswith("ag/"):
        return ("Antigravity", "Subscription · Google Antigravity (free public preview; paid tier TBD)")
    if r.startswith("openrouter/"):
        return ("OpenRouter", "Pay-per-use · OpenRouter prepaid credit (aggregator)")
    if r.startswith("vertex/"):
        return ("Vertex", "Pay-per-use · Google Vertex (jake-vertex proxy / paid key)")
    if r.startswith("cx/"):
        return ("ChatGPT/Codex", "Subscription · ChatGPT/Codex seat pool (or OpenAI API PPU)")
    if r.startswith("cc/"):
        return ("Claude Code", "Subscription · Claude Code / Max seat (Anthropic). PPU only if routed via OpenRouter.")
    if r.startswith("glm/"):
        return ("z.ai", "Subscription · GLM Coding Plan quota")
    if r.startswith("kimi/"):
        return ("Kimi", "Subscription · Kimi Code pool")
    if r.startswith("minimax/"):
        return ("MiniMax", "Pay-per-use · MiniMax API")
    if r.startswith("nvidia/"):
        return ("NIM", "Credits · NVIDIA NIM (overflow only)")
    # Not 9Router-wired — fall back to provider family.
    fam = {
        "anthropic": ("Claude Code", "Subscription · Claude Code / Max seat (Anthropic)"),
        "openai": ("ChatGPT/Codex", "Subscription · ChatGPT/Codex seat"),
        "google": ("Vertex", "Pay-per-use · Google Vertex"),
        "zai-glm": ("z.ai", "Subscription · GLM Coding Plan"),
        "moonshot-kimi": ("Kimi", "Subscription · Kimi Code pool"),
        "minimax": ("MiniMax", "Pay-per-use · MiniMax API"),
    }
    return fam.get(family, ("—", "Access channel not recorded"))


def blended_cost(cin, cout):
    """Output-weighted blended $/MTok (output dominates agent workloads)."""
    if cin is None or cout is None:
        return None
    return round(cin * 0.3 + cout * 0.7, 4)


# Cost tier — an informational $ magnitude, NOT folded into the benchmark score.
# Non-overlapping half-open bands on the blended $/MTok rate.
def cost_tier(blend):
    if blend is None:
        return None
    if blend < 0.50:
        return "$"
    if blend < 2.00:
        return "$$"
    if blend < 8.00:
        return "$$$"
    if blend < 30.00:
        return "$$$$"
    return "$$$$$"


def _family_cost_neighbour(m):
    """Estimate cost for models with unpublished pricing from a family median."""
    fam = [x for x in ROSTER if x["family"] == m["family"] and x.get("cin") is not None]
    if not fam:
        return (1.0, 4.0)
    cin = sorted(x["cin"] for x in fam)[len(fam) // 2]
    cout = sorted(x["cout"] for x in fam)[len(fam) // 2]
    return (cin, cout)


def log_norm(value, lo, hi):
    """Log-scaled 0..100 normalisation (compresses wide ranges)."""
    if value is None or value <= 0:
        return 0.0
    value = max(lo, min(hi, value))
    return round((math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo)) * 100, 1)


def compute_dims(models):
    """Attach the normalised 0..100 dimension dict to every model in place."""
    # Cost range across models that have (or can estimate) a blended cost.
    blends = []
    for m in models:
        cin, cout = m.get("cin"), m.get("cout")
        if cin is None or cout is None:
            cin, cout = _family_cost_neighbour(m)
            m["cost_estimated"] = True
        m["_cin"], m["_cout"] = cin, cout
        m["cost_blend"] = blended_cost(cin, cout)
        if m["cost_blend"]:
            blends.append(m["cost_blend"])
    cost_lo, cost_hi = min(blends), max(blends)

    for m in models:
        fam = m["family"]
        # intelligence: PINNED Artificial Analysis Intelligence Index baseline
        # (aa_baseline.py, AA v4.1, single scale). Roster aa/intel_est are no longer
        # used for scoring — aa_baseline is the single source of truth.
        aa_score, aa_src = aa_baseline.lookup(m["display"])
        if aa_src == "missing":
            AA_MISSING.append(m["display"])
        m["intelligence_estimated"] = aa_src != "AA"   # AA~ / est / None count as estimated
        m["intelligence_source"] = aa_src
        m["intelligence_raw"] = aa_score                # the v4.1 index value
        intel = aa_baseline.normalize(aa_score)

        # cost: cheaper → higher (inverse log). cost_blend==0 means free at point
        # of use (e.g. Antigravity preview) → cheapest (100).
        # Subscription / seat-billed channels (Antigravity, ChatGPT-Codex, Claude
        # Code, z.ai/GLM, Kimi) are flat-rate sunk cost → ZERO marginal $/token,
        # so they score cost=100 regardless of the model's list per-MTok price.
        # Only genuinely metered channels (Vertex, OpenRouter, MiniMax, NIM) carry
        # a real cost score. Without this, a sunk-cost flagship (Opus via Claude
        # Code, GPT-5.5 via Codex) is wrongly penalised against $0 Antigravity.
        _, _ch_bill = channel_for(m.get("router"), fam)
        if _ch_bill.startswith("Subscription"):
            cost = 100.0
        elif m["cost_blend"]:
            cost = round(100 - log_norm(m["cost_blend"], cost_lo, cost_hi), 1)
        else:
            cost = 100.0

        # context: log-normalised 100K..1.05M
        context = log_norm(m.get("ctx") or 0, 100_000, 1_050_000)

        # tool use
        tool = float(TOOL_SCORE.get(m.get("tool", "tooluse"), 60))

        # speed: explicit roster estimate (speed_est) wins; else family base ± hints
        nm = m["display"].lower()
        if m.get("speed_est") is not None:
            speed = float(m["speed_est"])
        else:
            speed = float(SPEED_BASE.get(fam, 60))
            if any(h in nm for h in SPEED_FAST_HINTS):
                speed = min(98, speed + 22)
            if any(h in nm for h in SPEED_SLOW_HINTS):
                speed = max(28, speed - 18)
        m["speed_estimated"] = True

        # vision: 0 if not capable
        vision = float(VISION_SCORE.get(fam, 70)) if m.get("vision") else 0.0

        # privacy: family posture, nudged for OpenRouter passthrough of clean vendors
        pscore, plabel = PRIVACY.get(fam, (50, "unverified"))
        router = m.get("router") or ""
        if router.startswith("openrouter/") and fam in ("anthropic", "openai", "google"):
            pscore = min(pscore, 60)
            plabel = "via OpenRouter passthrough — vendor clearance not retained"
        m["privacy_label"] = plabel

        # coding: DERIVED estimate = AA intelligence index ± per-class delta
        # (aa_baseline.coding_lookup). Always `est`; falls back to the intelligence
        # dim for non-chat utilities with no AA score. See aa_baseline CODING_*.
        code_raw, _code_src = aa_baseline.coding_lookup(m["display"], fam, aa_score)
        coding = aa_baseline.normalize(code_raw) if code_raw is not None else intel
        m["coding_estimated"] = True

        m["dims"] = {
            "intelligence": round(intel, 1),
            "coding": round(coding, 1),
            "cost": cost,
            "context": context,
            "tool_use": tool,
            "speed": round(speed, 1),
            "vision": vision,
            "privacy": float(pscore),
        }
    return models


def passes_hard(m, hard):
    """True if model m satisfies every hard constraint for a lane."""
    if hard.get("require_chat") and m["authority"] == "none":
        return False
    if hard.get("require_vision") and not m.get("vision"):
        return False
    if hard.get("must_be_available") and not m.get("avail"):
        return False
    if hard.get("exclude_prc") and m.get("prc"):
        return False
    if hard.get("exclude_trains") and m.get("trains"):
        return False
    if hard.get("no_claude_reviewer") and m["family"] == "anthropic":
        return False
    if m["family"] in hard.get("exclude_families", ()):
        return False
    if "only_models" in hard and m["display"] not in hard["only_models"]:
        return False
    if "require_authority" in hard and m["authority"] != hard["require_authority"]:
        return False
    if "long_context_min" in hard and (m.get("ctx") or 0) < hard["long_context_min"]:
        return False
    # Meta-models / classifiers never recommended as a lane primary.
    if m["family"] == "openrouter" and hard.get("require_chat"):
        return False
    return True


# Documented policy penalties on sensitive lanes (see lanes.py).
PRC_PENALTY = 0.88
TRAINS_PENALTY = 0.55


def lane_score(m, weights):
    num = sum(weights[d] * m["dims"][d] for d in DIMENSIONS)
    den = sum(weights[d] for d in DIMENSIONS)
    return round(num / den, 1) if den else 0.0


def policy_factor(m, sensitive):
    if not sensitive:
        return 1.0
    f = 1.0
    if m.get("trains"):
        f *= TRAINS_PENALTY
    elif m.get("prc"):
        f *= PRC_PENALTY
    return f


def router_index(models):
    idx = {}
    for m in models:
        if m.get("router"):
            idx[m["router"]] = m["display"]
    return idx


def resolve_assignment(rid, ridx):
    """Map a config router id (or alias / 'auto') to a model display name."""
    if not rid or rid == "auto":
        return None
    if rid in ridx:
        return ridx[rid]
    # alias: glm-5-turbo → GLM 5.1 (config model_aliases)
    alias = {"glm/glm-5-turbo": "glm/glm-5.1"}
    if rid in alias and alias[rid] in ridx:
        return ridx[alias[rid]]
    return rid  # unknown — surface the raw id


# When the current primary is eligible but not #1, treat it as co-optimal if its
# policy_score is within this many points of the top — avoids "data prefers X"
# churn on hair-thin, cost-driven margins (the verdict still flags real gaps).
# Keep in sync with NEAR_OPTIMAL_TOL in web/src/pages/ModelBenchmarkPage.tsx.
NEAR_OPTIMAL_TOL = 2.5


def score_lanes(models):
    ridx = router_index(models)
    out = []
    for lane in LANES:
        eligible = [m for m in models if passes_hard(m, lane["hard"])]
        sensitive = lane.get("sensitive", False)
        floor = lane.get("primary_min_intel", 0)

        def _row(m):
            raw = lane_score(m, lane["weights"])
            fac = policy_factor(m, sensitive)
            below = floor > 0 and m["dims"]["intelligence"] < floor
            return {
                "model": m["display"], "router": m.get("router"), "family": m["family"],
                "score": raw, "policy_score": round(raw * fac, 1),
                "policy_penalty": round(fac, 2) if fac != 1.0 else None,
                "available": m.get("avail", True),
                "estimated": m.get("intelligence_estimated") or m.get("speed_estimated"),
                "below_primary_floor": below,
                "dims": m["dims"],
            }

        ranked = sorted((_row(m) for m in eligible), key=lambda r: r["policy_score"], reverse=True)
        cur_disp = resolve_assignment(lane["current_primary"], ridx)
        cur_rank = next((i + 1 for i, r in enumerate(ranked) if r["model"] == cur_disp), None)
        # Recommended primary + fallbacks must clear the knowledge floor (if any);
        # the full ranking still lists every eligible model for transparency.
        primary_pool = [r for r in ranked if not r.get("below_primary_floor")]
        rec = primary_pool[0] if primary_pool else (ranked[0] if ranked else None)
        fbs = primary_pool[1:4] if primary_pool else ranked[1:4]
        # Verdict: does the data agree with today's wiring?
        near = False
        if cur_disp is None:
            verdict = "unpinned (resolves to main model)"
        elif rec and rec["model"] == cur_disp:
            verdict = "current primary is top-ranked ✓"
        elif cur_rank and rec:
            cur_row = next((r for r in ranked if r["model"] == cur_disp), None)
            gap = round(rec["policy_score"] - cur_row["policy_score"], 1) if cur_row else None
            if gap is not None and gap <= NEAR_OPTIMAL_TOL:
                near = True
                verdict = f"current primary ✓ near-optimal (#{cur_rank}, within {gap:.1f} of {rec['model']})"
            else:
                verdict = f"current primary ranks #{cur_rank}; data prefers {rec['model']}"
        else:
            verdict = f"current primary not eligible under hard constraints; data prefers {rec['model'] if rec else 'n/a'}"
        out.append({
            "key": lane["key"], "label": lane["label"], "category": lane["category"],
            "blurb": lane["blurb"], "output_authority": lane["output_authority"],
            "sensitive": sensitive,
            "current_primary": lane["current_primary"],
            "current_primary_display": cur_disp,
            "current_fallbacks": lane["current_fallbacks"],
            "current_fallbacks_display": [resolve_assignment(x, ridx) for x in lane["current_fallbacks"]],
            "current_rank": cur_rank,
            "weights": {k: v for k, v in lane["weights"].items() if v},
            "primary_min_intel": floor or None,
            "hard": lane["hard"],
            "recommended": rec,
            "fallbacks": fbs,
            "verdict": verdict,
            "near_optimal": near,
            "ranking": ranked,
        })
    return out


# ── Optional: drift-check against the live vault matrices ──────────────────
def _parse_md_table(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip().startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 2 or set(cells[0]) <= {"-", ":"} or cells[0].lower() == "model":
                continue
            rows[cells[0].lower()] = cells
    return rows


def verify_against_vault(models):
    warnings = []
    cap = _parse_md_table(os.path.join(VAULT_MATRIX_DIR, "model-capability-matrix.md"))
    if not cap:
        return [f"vault matrices not found at {VAULT_MATRIX_DIR} — skipped drift check"]
    # AA snapshot lives under a heading in the capability file; parse loosely.
    aa = {}
    capfile = os.path.join(VAULT_MATRIX_DIR, "model-capability-matrix.md")
    with open(capfile, encoding="utf-8") as fh:
        for line in fh:
            mobj = re.match(r"\|\s*([^|]+?)\s*\|\s*~?(\d+)\s*\|", line)
            if mobj and "intelligence" not in line.lower():
                aa[mobj.group(1).strip().lower()] = int(mobj.group(2))
    for m in models:
        if m.get("aa") is not None:
            key = m["display"].lower().replace("claude ", "").replace("-", "-")
            found = aa.get(m["display"].lower()) or aa.get(key)
            if found is not None and found != m["aa"]:
                warnings.append(f"AA drift: {m['display']} roster={m['aa']} vault={found}")
    return warnings or ["no drift detected vs vault capability matrix"]


def merge_measured(models):
    """Overlay live-harness measurements (latency/throughput) onto the roster.

    For each model with an OK measured run, replace the `speed` dimension
    estimate with a throughput-derived score and attach the raw numbers. Models
    that failed the sweep (or weren't swept) keep their estimate.
    """
    if not os.path.exists(MEASURED_PATH):
        return None
    try:
        with open(MEASURED_PATH, encoding="utf-8") as fh:
            measured = json.load(fh).get("measured", {})
    except (OSError, ValueError):
        return None
    by_router = {m.get("router"): m for m in models if m.get("router")}
    applied = 0
    for rid, meas in measured.items():
        m = by_router.get(rid)
        if not m or not meas.get("ok"):
            continue
        tps = meas.get("throughput_mean_tps")  # gen-only where truly streamed, else effective
        # Defensive: reject implausible rates (buffered-stream artifacts) — keep the
        # estimate rather than trust a garbage number.
        if tps and tps > 300:
            tps = meas.get("effective_tps_mean")
        m["measured"] = {
            "latency_median_s": meas.get("latency_median_s"),
            "throughput_tps": tps,
            "effective_tps": meas.get("effective_tps_mean"),
            "ttft_median_s": meas.get("ttft_median_s"),
            "runs": meas.get("runs"),
        }
        if tps and tps <= 300:
            m["dims"]["speed"] = round(min(100.0, tps / SPEED_REF_TPS * 100), 1)
            m["speed_estimated"] = False
            applied += 1
    return applied


# FluxLabs security policy (2026-06-17): no PRC-jurisdiction or train-on-input
# model may appear in the benchmark or be recommended for any lane. This gate
# enforces it even if such a model is re-added to roster.py.
BLOCK_PRC_TRAINS = True

# Intelligence is now sourced from the pinned aa_baseline (AA Index v4.1) — the old
# in-file VERIFIED_AA table is retired (it had mixed two AA index scales). See
# aa_baseline.py + vault decision 2026-06-17.


def build(stamp=None, verify=False):
    AA_MISSING.clear()
    models = [dict(m) for m in ROSTER]
    if BLOCK_PRC_TRAINS:
        models = [m for m in models if not (m.get("prc") or m.get("trains"))]
    compute_dims(models)              # intelligence dim ← aa_baseline (AA v4.1)
    measured_applied = merge_measured(models)
    arena_meta = arena_overlay.attach(models)   # Layer 3: our own Arena Elo, if present
    lanes = score_lanes(models)
    if AA_MISSING:
        print(f"[aa_baseline] WARNING: {len(AA_MISSING)} models missing an AA entry: {', '.join(AA_MISSING)}")

    # Strip internal scratch fields from the model rows.
    clean_models = []
    for m in models:
        ch_name, ch_bill = channel_for(m.get("router"), m["family"])
        # Subscription/seat models bill against quota, not per-token — no $ tier.
        is_subscription = ch_bill.startswith("Subscription")
        clean_models.append({
            "id": m["display"], "family": m["family"], "router": m.get("router"),
            "channel": ch_name, "billing": ch_bill,
            "cost_tier": None if is_subscription else cost_tier(m["cost_blend"]),
            "available": m.get("avail", True), "availability_note": m.get("note", ""),
            "output_authority": m["authority"], "prc": m.get("prc", False),
            "trains_on_input": m.get("trains", False),
            "vision": m.get("vision", False), "context_window": m.get("ctx"),
            "max_output": m.get("maxout"), "cost_in": m["_cin"], "cost_out": m["_cout"],
            "cost_blend": m["cost_blend"], "cost_estimated": m.get("cost_estimated", False),
            "intelligence_raw": m.get("intelligence_raw"),
            "intelligence_estimated": m["intelligence_estimated"],
            "intelligence_source": m.get("intelligence_source"),
            "speed_estimated": m.get("speed_estimated", True),
            "measured": m.get("measured"),
            "arena": m.get("arena"),
            "privacy_label": m["privacy_label"], "tool_tier": m.get("tool"),
            "strong": m.get("strong", ""), "weak": m.get("weak", ""),
            "dims": m["dims"],
        })

    payload = {
        "meta": {
            "schema_version": 1,
            "generated_at": stamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generator": "scripts/model_benchmark/build_benchmark.py",
            "aa_index": {
                "version": aa_baseline.AA_VERSION,
                "as_of": aa_baseline.AA_ASOF,
                "source": aa_baseline.AA_SOURCE,
                "methodology": aa_baseline.AA_METHODOLOGY,
                "dim_norm": {"lo": aa_baseline.AA_DIM_LO, "hi": aa_baseline.AA_DIM_HI},
            },
            "arena": arena_meta,
            "source": "Intelligence: pinned Artificial Analysis Index " + aa_baseline.AA_VERSION + " (aa_baseline.py). Speed/cost: measured (live_eval_harness e2e) + model cards. Agent Elo: arena_eval.py (Bradley-Terry, seeded from LMArena).",
            "method": "Data-driven hybrid benchmark. Dimensions normalised 0..100; per-lane composite = weighted mean over dimensions after hard constraints. Sensitive lanes apply a documented policy penalty (PRC ×0.88, trains-on-input ×0.55) → policy_score; recommendations rank by policy_score, raw score also shown. `speed` is a documented estimate and quality==capability until live_eval_harness.py merges measured dims.",
            "model_count": len(clean_models),
            "lane_count": len(lanes),
            "measured_models": measured_applied or 0,
            "dimensions": DIMENSIONS,
            "weight_mode": WEIGHT_MODE,
            "disclaimer": "Recommendations are advisory — nothing here is auto-applied to routing. AA Intelligence Index is a point-in-time third-party snapshot, not a model-card fact.",
        },
        "models": clean_models,
        "lanes": lanes,
    }

    if verify:
        payload["meta"]["vault_drift_check"] = verify_against_vault(models)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    # Write a dated snapshot for the webui date/time filter, then prune.
    try:
        os.makedirs(HISTORY_DIR, exist_ok=True)
        safe = payload["meta"]["generated_at"].replace(":", "").replace("-", "")
        snap = os.path.join(HISTORY_DIR, f"model-benchmark-{safe}.json")
        with open(snap, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        snaps = sorted(
            (f for f in os.listdir(HISTORY_DIR) if f.startswith("model-benchmark-") and f.endswith(".json"))
        )
        for old in snaps[:-HISTORY_KEEP]:
            try:
                os.remove(os.path.join(HISTORY_DIR, old))
            except OSError:
                pass
    except OSError:
        pass  # snapshot is best-effort; canonical write already succeeded
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="drift-check against live vault matrices")
    ap.add_argument("--stamp", help="ISO timestamp to record as generated_at (else now)")
    args = ap.parse_args()
    payload = build(stamp=args.stamp, verify=args.verify)
    print(f"wrote {OUTPUT_PATH}")
    print(f"  models={payload['meta']['model_count']} lanes={payload['meta']['lane_count']}")
    if args.verify:
        for w in payload["meta"]["vault_drift_check"]:
            print(f"  [verify] {w}")
    # Quick console summary of per-lane recommendations.
    print("\n  lane → recommended (verdict):")
    for ln in payload["lanes"]:
        rec = ln["recommended"]["model"] if ln["recommended"] else "—"
        print(f"    {ln['key']:<20} → {rec:<26} {ln['verdict']}")


if __name__ == "__main__":
    main()
