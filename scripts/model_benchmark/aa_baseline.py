#!/usr/bin/env python3
"""Artificial Analysis Intelligence Index — pinned, VERSIONED baseline.

This is the single source of truth for the `intelligence` dimension. It replaces
the old hand-maintained `roster.aa` / `intel_est` / `VERIFIED_AA` mix, which had
silently blended TWO incompatible AA index scales (the pre-v4 "top≈73" scale and
the v4 "top≈50" scale) → producing the Gemini-Pro-57-vs-46.5 contradictions.

Industry-standard choice (see vault decision 2026-06-17 + run record): the
**Artificial Analysis Intelligence Index** is the de-facto composite standard —
a weighted mean of 9 production benchmarks across 4 categories, ±1% CI, recalibrated
per major version. We PIN a version so the scale can never drift again.

Every score here is on ONE scale (AA Index v4.1, max/high-effort per model). Each
carries a `src`:
    "AA"   captured directly from the AA v4.1 leaderboard (2026-06 snapshot)
    "AA~"  AA-derived: an effort-variant / interpolation of a sourced AA model
    "est"  no AA datapoint — reasoned estimate placed on the v4.1 scale by neighbours
To refresh: re-pull the AA leaderboard, bump AA_VERSION/AA_ASOF, update SCORES.
"""

import re

AA_VERSION = "v4.1"
AA_ASOF = "2026-06"           # snapshot date of the captured numbers
AA_SOURCE = "Artificial Analysis Intelligence Index (artificialanalysis.ai), v4.1 leaderboard"

# v4.1 composition (for the UI / methodology panel). Weights sum to 100.
AA_METHODOLOGY = {
    "scale": "0-100 weighted mean; v4.1 tops out ~56-60 (recalibrated from the v3 ~73 scale)",
    "ci": "±1%",
    "categories": {
        "Agents (34%)": {"GDPval-AA v2": 20, "tau3-Banking": 14},
        "Coding (24%)": {"Terminal-Bench v2.1": 16, "SciCode": 8},
        "Scientific Reasoning (24%)": {"HLE": 12, "GPQA Diamond": 6, "CritPt": 6},
        "General (18%)": {"AA-Omniscience": 12, "AA-LCR": 6},
    },
}

# Normalisation of a v4.1 index score → the benchmark's 0..100 `intelligence` dim.
# Linear: aa=10 → 0, aa=35 → 50, aa=60 → 100. Keeps spread + ordering on v4.1.
AA_DIM_LO = 10.0
AA_DIM_HI = 60.0


def normalize(score):
    """v4.1 index score → 0..100 intelligence dimension."""
    if score is None:
        return 0.0
    return max(0.0, min(100.0, (score - AA_DIM_LO) / (AA_DIM_HI - AA_DIM_LO) * 100.0))


# display name → (AA v4.1 index score, src). None score = not an intelligence model.
SCORES = {
    # ── Anthropic (sourced where on the leaderboard; ladder-anchored estimates else) ──
    "Claude Fable 5":            (59.9, "AA"),
    "Claude Opus 4.8":           (55.7, "AA"),
    "Claude Opus 4.7":           (53.5, "AA"),
    "Claude Opus 4.6":           (51.0, "est"),   # between 4.5 and 4.7 (53.5)
    "Claude Opus 4.5":           (48.0, "est"),
    "Claude Sonnet 4.6":         (47.2, "AA"),
    "Claude Sonnet 4.5":         (40.0, "est"),
    "Claude Haiku 4.5":          (29.6, "AA"),
    # ── OpenAI ──
    "GPT-5.5-pro":               (57.0, "est"),   # above GPT-5.5
    "GPT-5.5":                   (54.8, "AA"),
    "GPT-5.4":                   (51.4, "AA"),
    "GPT-5.3-Codex":             (40.1, "AA~"),   # = GPT-5.2 Codex proxy (unavailable on host)
    "GPT-5.4-mini":              (40.0, "AA"),
    "GPT-5.3-Codex Spark":       (36.0, "est"),   # lighter codex
    "GPT-chat-latest":           (33.0, "est"),   # non-reasoning chat (~GPT-5.5 non-reasoning 32.7)
    "GPT-OSS-120B":              (20.0, "est"),
    # ── Google ──
    "Gemini 3.5 Flash":          (50.2, "AA"),
    "Gemini 3.1 Pro":            (46.5, "AA"),
    "Gemini 3 Flash":            (37.8, "AA"),    # reasoning
    "Gemini 2.5 Pro":            (30.0, "est"),
    "Gemini 3.1 Flash-Lite":     (28.0, "est"),
    "Gemma 4 26B":               (25.0, "est"),
    "Gemini 2.5 Flash":          (18.0, "est"),
    "Gemini 2.5 Flash-Lite":     (13.0, "est"),
    # ── xAI ──
    "Grok 4.3":                  (37.6, "AA"),    # high effort
    "Grok Build 0.1":            (34.0, "est"),
    # ── NVIDIA ──
    "Nemotron 3 Ultra":          (40.0, "est"),
    "Nemotron 3 Super":          (31.0, "est"),
    # ── Mistral ──
    "Mistral Medium 3.5":        (26.0, "est"),
    "Mistral Devstral 2":        (24.0, "est"),
    "Mistral Codestral 2508":    (22.0, "est"),
    "Mistral Small 4":           (19.0, "est"),
    "Mistral Large 3":           (16.2, "AA"),
    # ── Amazon ──
    "Amazon Nova Pro":           (12.0, "est"),
    "Amazon Nova Lite":          (9.0, "est"),
    "Amazon Nova Micro":         (6.0, "est"),
    # ── Cohere ──
    "Cohere Command A":          (28.0, "est"),
    "Cohere Command R":          (18.0, "est"),
    # ── Router / meta ──
    "Fusion":                    (35.0, "est"),
    # ── Antigravity channel variants → base-model AA at the variant's effort ──
    "Gemini 3.1 Pro Agent (AG)":     (46.5, "AA~"),   # = Gemini 3.1 Pro (agent variant)
    "Gemini 3.1 Pro Low (AG)":       (38.0, "AA~"),   # 3.1 Pro, low effort
    "Gemini 3.5 Flash Low (AG)":     (42.0, "AA~"),   # 3.5 Flash, between med 45.4 / min 34.9
    "Gemini 3.5 Flash X-Low (AG)":   (34.9, "AA~"),   # 3.5 Flash minimal
    "Gemini 3 Flash (AG)":           (37.8, "AA~"),   # = Gemini 3 Flash reasoning
    "Gemini 3 Flash Agent (AG)":     (37.8, "AA~"),
    "Claude Opus 4.6 Thinking (AG)": (51.0, "AA~"),   # = Opus 4.6 est
    "Claude Sonnet 4.6 (AG)":        (47.2, "AA~"),   # = Sonnet 4.6
    "GPT-OSS 120B Med (AG)":         (18.0, "est"),
    # ── PRC models (BLOCKED from recommendations, but carried for completeness) ──
    "MiniMax-M3":                (44.4, "AA"),
    "DeepSeek V4 Pro":           (44.3, "AA"),
    "Kimi K2.6":                 (42.8, "AA"),
    "Qwen3.7 Plus":              (44.0, "est"),   # ~ Qwen3.7 Max 46
    "GLM 5.3":                   (45.0, "est"),
    "GLM 5.1":                   (43.0, "est"),
    "GLM 5.2":                   (44.0, "est"),
    "GLM 5":                     (42.0, "est"),
    "Kimi K2.7":                 (43.0, "est"),
    "Kimi K2.7 Code":            (43.0, "est"),
    "Kimi K2.7 Code HighSpeed":  (40.0, "est"),
    "GLM 4.7":                   (40.0, "est"),
    "MiniMax M2.7":              (38.0, "est"),
    "GLM 4.6V":                  (38.0, "est"),
    "GLM 4.5-Air":               (33.0, "est"),
    "DeepSeek V4 Flash":         (31.0, "est"),
    "MiMo V2.5":                 (30.0, "est"),
    # ── Non-chat utilities (no intelligence score) ──
    # ── Morgan-only models (its 9router exposes these; AA estimates on the v4.1 scale) ──
    "Gemma 4 31B":               (27.0, "est"),
    "Claude Opus 4.8 Fast":      (50.0, "est"),
    "Grok 4.20 Multi Agent":     (35.0, "est"),
    "Grok 4.20":                 (33.0, "est"),
    "Nex N2 Pro":                (28.0, "est"),
    "Owl Alpha":                 (25.0, "est"),
    "Laguna M.1":                (25.0, "est"),
    "Nemotron 3 Nano Omni 30B A3B Reasoning": (24.0, "est"),
    "Nemotron 3 Nano 30B A3B":   (22.0, "est"),
    "Laguna Xs.2":               (16.0, "est"),
    "GPT OSS 20B":               (14.0, "est"),
    "Llama 3.1 8B":              (10.0, "est"),
    "Llama Guard 4 12B":         (None, None),
    "Parakeet CTC 1.1B ASR":     (None, None),
}


def lookup(display):
    """Return (v4.1 score|None, src|None) for a model display name. Tolerates a
    trailing channel/variant suffix like " (OR)", " (cx)", " (AG)" by falling back
    to the base name (a review/openrouter variant scores as its base model)."""
    hit = SCORES.get(display)
    if hit is not None:
        return hit
    base = re.sub(r"\s*\([^)]*\)\s*$", "", display)
    if base != display and base in SCORES:
        return SCORES[base]
    return (None, "missing")


# ── Coding-specific capability (ESTIMATED) ──────────────────────────────────
# There is no per-model coding leaderboard in the repo yet, so the `coding`
# dimension is DERIVED: the model's AA v4.1 intelligence index (which already
# folds in 24% coding) plus a documented per-class delta on the same scale —
# coding specialists up, chat/vision-only down. Every value is therefore an
# `est`. Replace with real Terminal-Bench v2.1 / SciCode per-model numbers when
# available (bump a CODING_VERSION then). The intelligence dim still exists
# independently; this is an ADDITIONAL coding-weighted signal so builder-type
# lanes can separate from evidence/skeptic lanes. Deltas are tunable.
CODING_VERSION = "est-v1"

# Explicit per-model overrides (AA-scale points added to the intelligence index).
CODING_DELTA = {
    "GPT-5.3-Codex":              +4,
    "GPT-5.3-Codex Spark":        +4,
    "Mistral Codestral 2508":     +5,
    "Mistral Devstral 2":         +4,
    "GLM 5.3":                    +3,
    "GLM 5.1":                    +2,
    "GLM 5":                      +2,
    "Kimi K2.7 Code":             +4,
    "Kimi K2.7 Code HighSpeed":   +4,
    "Qwen3.7 Plus":               +2,
    # coding-weak / general-only → negative on top of family default
    "Mistral Large 3":            -2,
    "Gemma 4 26B":                -3,
    "Gemma 4 31B":                -3,
    "Cohere Command R":           -2,
    "GPT-OSS-120B":               -2,
    "GPT OSS 20B":                -2,
}

# Fallback delta by provider family when no explicit override above.
CODING_FAMILY_DELTA = {
    "anthropic":   +3,   # consistently strong agentic coders
    "openai":      +2,
    "zai-glm":     +2,   # GLM coding plans
    "minimax":     +1,
    "moonshot-kimi": +2,
    "mistral":     +1,
    "google":       0,
    "xai":          0,
    "qwen":        +1,
    "cohere":      -2,
    "meta":        -2,
    "amazon":      -3,
    "nvidia":       0,
}


def coding_lookup(display, family, aa_score):
    """(coding_raw_on_v4.1_scale | None, src). Returns None when the model has no
    AA intelligence score (non-chat utilities) → caller falls back to intel."""
    if aa_score is None:
        return (None, "missing")
    delta = CODING_DELTA.get(display)
    if delta is None:
        base = re.sub(r"\s*\([^)]*\)\s*$", "", display)
        delta = CODING_DELTA.get(base)
    if delta is None:
        delta = CODING_FAMILY_DELTA.get(family, 0)
    return (max(0.0, aa_score + delta), "est")
