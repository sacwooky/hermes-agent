#!/usr/bin/env python3
"""Hermes lane (task) definitions + per-lane scoring profiles.

A *lane* is any task in the Hermes/fleet stack that consumes a model: the main
chat agent, every ``auxiliary:`` slot in ``config.yaml``, and every kanban
worker role (builder, qa, reviewer, ...). For each lane we record:

* ``category``          — grouping for the UI.
* ``current_primary``   — the model wired in today (config.yaml / lane contract).
* ``current_fallbacks`` — the recorded fallback chain (trimmed to the head).
* ``output_authority``  — advisory | proposed | scoped-write | binding | none.
* ``weights``           — how much each scoring dimension matters for THIS lane
                          (values are relative; they are normalised at scoring
                          time, so they need not sum to 1).
* ``hard``              — hard constraints applied BEFORE scoring. A model that
                          fails any hard constraint is excluded from the lane's
                          ranking (it can never be a recommendation).

Dimensions (all normalised 0..100 in build_benchmark.py):
    intelligence  capability / AA Intelligence Index
    cost          cheaper == higher score (inverse of blended $/MTok)
    context       usable context window
    tool_use      function-calling / agentic tool reliability
    speed         latency / throughput (data-driven estimate; harness refines)
    vision        multimodal image/video input quality
    privacy       data-handling cleanliness (commercial-clean best)

Hard-constraint keys:
    require_vision        model must accept image input
    require_chat          exclude classifiers / ASR (output_authority == none)
    exclude_prc           exclude PRC-jurisdiction providers (sensitive lanes)
    exclude_trains        exclude providers that train on API inputs
    must_be_available     exclude unavailable / gated-unknown models
    no_claude_reviewer    (review lanes) exclude Claude family — independence
    exclude_families      list of model families barred from this lane
    long_context_min      minimum context window in tokens

Per-lane ``sensitive`` flag: when True, models from PRC providers and providers
that train on inputs receive a documented multiplicative penalty on their
``policy_score`` (PRC ×0.88, trains-on-input ×0.55). This encodes the fleet
posture of leading sensitive maker/review/knowledge lanes with clean-vendor
models while keeping PRC models visible as ranked fallbacks. Low-risk lanes
(titles, search, web-extract, ...) are not sensitive — cheapest wins there.
Recommendations rank by ``policy_score``; the raw ``score`` is also exposed.

These profiles are intentionally transparent and editable — they ARE the
benchmark's opinion. Tune the weights here, re-run build_benchmark.py, and the
webui page + recommendations update. Nothing here is auto-applied to routing.
"""
from __future__ import annotations

import os

# Dimensions that exist for every model. Order is the canonical display order.
DIMENSIONS = ["intelligence", "coding", "cost", "context", "tool_use", "speed", "vision", "privacy"]


def _w(**kw):
    """Weight profile helper — unspecified dimensions default to 0."""
    base = {d: 0.0 for d in DIMENSIONS}
    base.update(kw)
    return base


LANES = [
    # ───────────────────────── Core agent / chat ─────────────────────────
    {
        "key": "default_chat",
        "label": "Default user chat",
        "category": "core",
        "blurb": "The primary conversational agent (config.yaml model.default). Balanced general assistant; advisory output.",
        "current_primary": "cx/gpt-5.5",
        "current_fallbacks": ["glm/glm-5-turbo", "vertex/gemini-2.5-flash"],
        "output_authority": "advisory",
        "weights": _w(intelligence=5, coding=1, cost=2, context=2, tool_use=4, speed=3, vision=1, privacy=2),
        "hard": {"require_chat": True, "must_be_available": True},
    },
    {
        "key": "orchestrator",
        "label": "Orchestrator / conductor (PM)",
        "category": "core",
        "blurb": "Plans work, routes specialists, drives gates. Never self-approves a gate. Reliable structured output + large context.",
        "current_primary": "cx/gpt-5.5",
        "current_fallbacks": ["glm/glm-5-turbo", "vertex/gemini-2.5-flash"],
        "output_authority": "advisory",
        "weights": _w(intelligence=5, coding=2, cost=1, context=4, tool_use=5, speed=2, privacy=2),
        "hard": {"require_chat": True, "must_be_available": True},
    },
    # ───────────────────────── Kanban worker roles ───────────────────────
    {
        "key": "builder",
        "label": "Builder",
        "category": "kanban-worker",
        "blurb": "Writes small reviewable diffs. Proposed change only — no merge/ship. Balanced cost/latency, strong coding + tool use.",
        "current_primary": "cx/gpt-5.5",
        "current_fallbacks": ["glm/glm-5-turbo", "vertex/gemini-2.5-flash"],
        "output_authority": "proposed",
        "weights": _w(intelligence=4, coding=6, cost=2, context=3, tool_use=5, speed=3, privacy=2),
        "hard": {"require_chat": True, "must_be_available": True},
    },
    {
        "key": "integrator",
        "label": "Integrator",
        "category": "kanban-worker",
        "blurb": "The only scoped-write lane — may merge+push to the integration branch on a verified out-of-band verdict. Deterministic, reliable git tool use; cost secondary.",
        "current_primary": "cx/gpt-5.5",
        "current_fallbacks": ["glm/glm-5-turbo", "vertex/gemini-2.5-flash"],
        "output_authority": "scoped-write",
        "weights": _w(intelligence=4, coding=4, cost=1, context=2, tool_use=6, speed=2, privacy=3),
        "hard": {"require_chat": True, "must_be_available": True},
    },
    {
        "key": "maintainer",
        "label": "Maintainer",
        "category": "kanban-worker",
        "blurb": "Refactor / debug. Proposed change only. Reliable, cost-efficient, low-variance.",
        "current_primary": "cx/gpt-5.5",
        "current_fallbacks": ["glm/glm-5-turbo", "vertex/gemini-2.5-flash"],
        "output_authority": "proposed",
        "weights": _w(intelligence=4, coding=6, cost=3, context=3, tool_use=5, speed=2, privacy=2),
        "hard": {"require_chat": True, "must_be_available": True},
    },
    {
        "key": "researcher",
        "label": "Researcher",
        "category": "kanban-worker",
        "blurb": "Web/research + long-context synthesis with citation discipline. Advisory evidence.",
        "current_primary": "cx/gpt-5.5",
        "current_fallbacks": ["glm/glm-5-turbo", "vertex/gemini-2.5-flash"],
        "output_authority": "advisory",
        "weights": _w(intelligence=5, cost=2, context=6, tool_use=3, speed=1, privacy=2),
        "hard": {"require_chat": True, "must_be_available": True, "long_context_min": 400_000},
    },
    {
        "key": "km_agent",
        "label": "Knowledge-management agent",
        "category": "kanban-worker",
        "blurb": "Faithful summarisation into the vault (system of record). Long context + schema conformance. PRIVACY-CRITICAL.",
        "current_primary": "cx/gpt-5.5",
        "current_fallbacks": ["glm/glm-5-turbo", "vertex/gemini-2.5-flash"],
        "output_authority": "advisory",
        "weights": _w(intelligence=4, cost=2, context=6, tool_use=2, speed=1, privacy=5),
        "hard": {"require_chat": True, "must_be_available": True, "exclude_trains": True},
    },
    {
        "key": "ops_watch",
        "label": "Ops-watch",
        "category": "kanban-worker",
        "blurb": "Cheap, fast, reliable alert-only signal. Low-judgment. The hollow-all-clear trap is acute here — pin an explicitly-available model.",
        "current_primary": "cx/gpt-5.5",
        "current_fallbacks": ["glm/glm-5-turbo", "vertex/gemini-2.5-flash"],
        "output_authority": "advisory",
        "weights": _w(intelligence=2, cost=6, context=2, tool_use=3, speed=6, privacy=2),
        "hard": {"require_chat": True, "must_be_available": True},
    },
    {
        "key": "qa_functional",
        "label": "QA — functional",
        "category": "kanban-worker",
        "blurb": "Functional test dispatch / evidence gathering. Skeptical, evidence-first. Advisory.",
        "current_primary": "cx/gpt-5.5",
        "current_fallbacks": ["glm/glm-5-turbo", "vertex/gemini-2.5-flash"],
        "output_authority": "advisory",
        "weights": _w(intelligence=5, coding=3, cost=2, context=3, tool_use=5, speed=2, privacy=2),
        "hard": {"require_chat": True, "must_be_available": True},
    },
    {
        "key": "qa_vision",
        "label": "QA — vision (screenshot↔wireframe)",
        "category": "kanban-worker",
        "blurb": "Compares rendered screenshots to wireframes (Phase 14). REQUIRES image input. Must not be the builder model.",
        "current_primary": "minimax/MiniMax-M3",
        "current_fallbacks": ["vertex/gemini-2.5-flash"],
        "output_authority": "advisory",
        "weights": _w(intelligence=4, cost=3, context=3, tool_use=2, speed=2, vision=8, privacy=2),
        "hard": {"require_vision": True, "must_be_available": True},
    },
    {
        "key": "reviewer",
        "label": "Reviewer (binding, Claude-authored code)",
        "category": "review",
        "blurb": "Binding independent review of Claude-authored code. Must be independent of the builder — NO Claude self-review. INTERIM (2026-06-17): Gemini 2.5 Pro after PRC reviewers (MiniMax/GLM/Kimi) were removed.",
        "current_primary": "vertex/gemini-2.5-pro",
        "current_fallbacks": [],
        "output_authority": "binding",
        "weights": _w(intelligence=6, coding=4, cost=1, context=4, tool_use=4, speed=2, privacy=3),
        "hard": {"require_chat": True, "must_be_available": True, "no_claude_reviewer": True},
    },
    # ───────────────────────── Auxiliary slots (config.yaml) ─────────────
    {
        "key": "vision",
        "label": "Vision (aux)",
        "category": "auxiliary",
        "blurb": "General image analysis aux slot. REQUIRES image input. 18-deep fallback chain today.",
        "current_primary": "vertex/gemini-2.5-flash",
        "current_fallbacks": ["vertex/gemini-2.5-pro", "cx/gpt-5.5", "cc/claude-sonnet-4-6"],
        "output_authority": "advisory",
        "weights": _w(intelligence=3, cost=4, context=2, tool_use=1, speed=4, vision=8, privacy=2),
        "hard": {"require_vision": True, "must_be_available": True},
    },
    {
        "key": "compression",
        "label": "Context compression",
        "category": "auxiliary",
        "blurb": "Compacts long conversation context. Needs a large window + cheap/fast throughput; faithful summarisation.",
        "current_primary": "glm/glm-5.1",
        "current_fallbacks": ["vertex/gemini-2.5-flash-lite"],
        "output_authority": "advisory",
        "weights": _w(intelligence=3, cost=5, context=7, tool_use=0, speed=4, privacy=2),
        "hard": {"require_chat": True, "must_be_available": True, "long_context_min": 200_000},
    },
    {
        "key": "triage_specifier",
        "label": "Triage / spec fleshing",
        "category": "auxiliary",
        "blurb": "Fleshes out kanban specs. Reliable structured output, cheap, fast.",
        "current_primary": "vertex/gemini-2.5-flash",
        "current_fallbacks": ["vertex/gemini-2.5-flash-lite"],
        "output_authority": "advisory",
        "weights": _w(intelligence=4, cost=5, context=3, tool_use=2, speed=5, privacy=2),
        "hard": {"require_chat": True, "must_be_available": True},
    },
    {
        "key": "kanban_decomposer",
        "label": "Kanban decomposer",
        "category": "auxiliary",
        "blurb": "Decomposes epics into tasks. Reliable structured output + large ctx (2026-06-12 bake-off winner: gemini-2.5-flash).",
        "current_primary": "vertex/gemini-2.5-flash",
        "current_fallbacks": ["vertex/gemini-2.5-flash-lite"],
        "output_authority": "advisory",
        "weights": _w(intelligence=5, cost=4, context=4, tool_use=3, speed=4, privacy=2),
        "hard": {"require_chat": True, "must_be_available": True},
    },
    {
        "key": "profile_describer",
        "label": "Profile describer",
        "category": "auxiliary",
        "blurb": "Auto-writes short profile descriptions. Cheap, fast, low-judgment.",
        "current_primary": "vertex/gemini-2.5-flash",
        "current_fallbacks": ["cx/gpt-5.5"],
        "output_authority": "advisory",
        "weights": _w(intelligence=2, cost=6, context=2, tool_use=1, speed=6, privacy=1),
        "hard": {"require_chat": True, "must_be_available": True},
    },
    {
        "key": "web_extract",
        "label": "Web extract / summarise",
        "category": "auxiliary",
        "blurb": "Summarises fetched web pages. Cheap, fast, large-ish context.",
        "current_primary": "auto",
        "current_fallbacks": [],
        "output_authority": "advisory",
        "weights": _w(intelligence=3, cost=6, context=4, tool_use=0, speed=5, privacy=1),
        "hard": {"require_chat": True, "must_be_available": True},
    },
    {
        "key": "title_generation",
        "label": "Title generation",
        "category": "auxiliary",
        "blurb": "Generates short session titles. The cheapest/fastest tier wins.",
        "current_primary": "auto",
        "current_fallbacks": [],
        "output_authority": "advisory",
        "weights": _w(intelligence=1, cost=8, context=1, tool_use=0, speed=8, privacy=1),
        "hard": {"require_chat": True, "must_be_available": True},
    },
    {
        "key": "approval",
        "label": "Smart auto-approve",
        "category": "auxiliary",
        "blurb": "Decides whether a tool call is safe to auto-approve. Fast, reliable, low-judgment classification.",
        "current_primary": "auto",
        "current_fallbacks": [],
        "output_authority": "advisory",
        "weights": _w(intelligence=3, cost=5, context=1, tool_use=2, speed=7, privacy=2),
        "hard": {"require_chat": True, "must_be_available": True},
    },
    {
        "key": "mcp",
        "label": "MCP tool routing",
        "category": "auxiliary",
        "blurb": "Routes among MCP tools. Strong tool-use, fast.",
        "current_primary": "auto",
        "current_fallbacks": [],
        "output_authority": "advisory",
        "weights": _w(intelligence=3, cost=4, context=2, tool_use=6, speed=5, privacy=2),
        "hard": {"require_chat": True, "must_be_available": True},
    },
    {
        "key": "skills_hub",
        "label": "Skills hub search",
        "category": "auxiliary",
        "blurb": "Searches/ranks skills. Cheap, fast.",
        "current_primary": "auto",
        "current_fallbacks": [],
        "output_authority": "advisory",
        "weights": _w(intelligence=2, cost=6, context=2, tool_use=2, speed=7, privacy=1),
        "hard": {"require_chat": True, "must_be_available": True},
    },
    {
        "key": "curator",
        "label": "Skill-usage curator",
        "category": "auxiliary",
        "blurb": "Reviews skill usage over long transcripts. Long context, cost-disciplined.",
        "current_primary": "auto",
        "current_fallbacks": [],
        "output_authority": "advisory",
        "weights": _w(intelligence=4, cost=4, context=6, tool_use=1, speed=1, privacy=2),
        "hard": {"require_chat": True, "must_be_available": True, "long_context_min": 200_000},
    },
    {
        "key": "session_search",
        "label": "Session search",
        "category": "auxiliary",
        "blurb": "Semantic search across sessions. Cheap, fast, concurrent.",
        "current_primary": "auto",
        "current_fallbacks": [],
        "output_authority": "advisory",
        "weights": _w(intelligence=2, cost=6, context=3, tool_use=1, speed=7, privacy=2),
        "hard": {"require_chat": True, "must_be_available": True},
    },
    {
        "key": "flush_memories",
        "label": "Flush memories",
        "category": "auxiliary",
        "blurb": "Distils durable memories from sessions. Faithful, cheap, large context. Privacy-aware (touches durable memory).",
        "current_primary": "glm/glm-5.1",
        "current_fallbacks": ["vertex/gemini-2.5-flash-lite"],
        "output_authority": "advisory",
        "weights": _w(intelligence=3, cost=5, context=5, tool_use=0, speed=3, privacy=4),
        "hard": {"require_chat": True, "must_be_available": True},
    },
    {
        "key": "delegation",
        "label": "Sub-agent delegation",
        "category": "auxiliary",
        "blurb": "Drives delegated sub-agents. Strong tool-use + reasoning, cost-disciplined.",
        "current_primary": "glm/glm-5.1",
        "current_fallbacks": [],
        "output_authority": "advisory",
        "weights": _w(intelligence=4, cost=4, context=3, tool_use=5, speed=3, privacy=2),
        "hard": {"require_chat": True, "must_be_available": True},
    },
]


# ── Policy annotations ─────────────────────────────────────────────────────
# Sensitive lanes touch fleet/customer code, durable knowledge, binding output,
# or real conversation content → PRC / trains-on-input penalty applies.
_SENSITIVE = {
    "default_chat", "orchestrator", "builder", "integrator", "maintainer",
    "researcher", "km_agent", "qa_functional", "reviewer", "compression",
    "flush_memories", "delegation", "curator",
    # process customer project content → policy penalty applies
    "kanban_decomposer", "triage_specifier",
}
for _l in LANES:
    _l.setdefault("sensitive", _l["key"] in _SENSITIVE)

# ── Weight mode: per-lane (default) vs global flat override (optional) ───────
# By default each lane scores with its OWN per-role weights defined above, so a
# Builder rewards tool_use, Ops-watch rewards cost/speed, QA-vision rewards
# vision, etc. — the benchmark's recommendation is then role-aware.
#
# The operator "I want good models" GLOBAL flat vector (2026-06-17, rev2) is kept
# as an OPT-IN toggle. When enabled it overwrites every lane with one
# capability-led vector (cost removed, intelligence dominant) so the highest-
# intelligence eligible model is the standard #1 on every lane; lanes then differ
# ONLY by hard constraints (require_vision, require_authority, ...).
#
# Enable the flat override with either:
#     env  HERMES_BENCHMARK_FLAT_WEIGHTS=1   (1/true/yes/on)
#     or set USE_FLAT_WEIGHTS = True below.
_GLOBAL_WEIGHTS = {
    "intelligence": 7,  # "knowledge" — dominant. Bumped 5→7 2026-06-17 to restore
                        # Opus 4.8 as a durable #1 after honest e2e speed lifted Gemini
                        # 3.5 Flash; 6× was a dead tie (raw −0.02), 7× gives +0.68 cushion.
    "coding": 0,        # flat mode: intelligence already embeds coding → keep 0 so
                        # the global-flat override stays identical to pre-coding behaviour
    "cost": 0,          # removed per operator (budget; subscription-billed)
    "context": 2,
    "tool_use": 3,      # "tools"
    "speed": 2,         # light — breaks ties among strong models, never leads
    "vision": 1,        # unspecified by operator — kept low
    "privacy": 2,
}

USE_FLAT_WEIGHTS = os.environ.get(
    "HERMES_BENCHMARK_FLAT_WEIGHTS", ""
).strip().lower() in ("1", "true", "yes", "on")

# Cost handling: the operator originally zeroed cost globally (2026-06-17) because
# the cost DIMENSION was mis-modelled — it penalised sunk-cost subscription seats
# (Opus via Claude Code, GPT-5.5 via Codex) against $0 Antigravity, flooding every
# lane with cheap variants. That root cause is now fixed in build_benchmark.py
# (subscription/seat channels score cost=100), so per-lane cost weights are once
# again meaningful: cost now only discriminates against genuinely metered
# pay-per-token channels, never against a seat already paid for. Cost is therefore
# kept IN by default. Set HERMES_BENCHMARK_DROP_COST=1 to zero it per lane anyway.
DROP_COST = os.environ.get(
    "HERMES_BENCHMARK_DROP_COST", ""
).strip().lower() in ("1", "true", "yes", "on")

WEIGHT_MODE = "global-flat" if USE_FLAT_WEIGHTS else (
    "per-lane (cost-neutral)" if DROP_COST else "per-lane"
)

if USE_FLAT_WEIGHTS:
    for _l in LANES:
        _l["weights"] = dict(_GLOBAL_WEIGHTS)
elif DROP_COST:
    for _l in LANES:
        _l["weights"] = {**_l["weights"], "cost": 0}

# Knowledge floor (2026-06-17): with speed weighted as heavily as knowledge, a
# fast-but-weak model can out-score a flagship. On high-judgment lanes a model
# below this knowledge dim (≈ AA 35) may still appear in the full ranking but is
# NOT eligible as the recommended primary or fallback. Stops a knowledge-16 model
# being wired as a thinking-lane default while still rewarding speed above the bar.
_HIGH_JUDGMENT = {
    "default_chat", "orchestrator", "builder", "integrator", "maintainer",
    "researcher", "km_agent", "qa_functional", "reviewer",
}
for _l in LANES:
    if _l["key"] in _HIGH_JUDGMENT:
        _l["primary_min_intel"] = 30

# The binding reviewer of Claude-authored code must be an APPROVED independent
# reviewer. Per the review-independence matrix, OpenAI/Codex has no review lane
# (retired on Robin) and Claude may never self-review.
LANES_BY_KEY = {l["key"]: l for l in LANES}
# The binding reviewer of Claude-authored code must be a model the fleet trusts
# with a *binding* verdict (review-independence matrix: MiniMax-M3 binding
# primary; GLM 5.2 preferred primary on z.ai reset). Kimi K2.7 Code is the
# 2nd-opinion pair (never solo-binding) and is surfaced in the recommendations
# doc, not as the lane primary.
LANES_BY_KEY["reviewer"]["hard"]["require_authority"] = "binding"
LANES_BY_KEY["reviewer"]["hard"]["long_context_min"] = 200_000
