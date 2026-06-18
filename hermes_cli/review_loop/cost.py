"""Encoded cost-routing / tiering — flat-rate-first (WI-C7, ADD-ON C v2 Phase 7).

Cost is policy, not agent discretion. This module makes the loop's cost posture explicit and
*assertable*: which lanes are free / flat / metered, what a Fusion run costs, the rule that Grok
(the only always-metered lane) fires on high-risk only, the context-size routing rule, and the
per-project budget gate. Pure (stdlib-only) — the Robin wrapper and `fusion.py` call it.

The cost-routing rules (WI-C7) are mostly already enforced structurally upstream:
- L0 deterministic = free (no model); L1 screen = flat (Antigravity).
- ``routing.select_jury`` puts Grok (xAI) only as the 3rd seat ⇒ high-risk only (rules 3 & 5).
- routine jury = flat Codex/Antigravity lanes; judge = flat Codex (cx/gpt-5.5-review).
This module adds the explicit tier table, the cost estimate, the defense-in-depth Grok guard,
the >context routing rule, and the budget-gate signal.
"""

from __future__ import annotations

from typing import Optional, Tuple

from hermes_cli.review_loop import routing

FREE = "free"        # no model (L0 deterministic)
FLAT = "flat"        # subscription / flat-quota — ≈$0 marginal
METERED = "metered"  # per-token billed

# Per-million-token rates for METERED lanes (USD). Flat/free lanes contribute $0 marginal.
# Grok numbers per the v2 spec; Vertex/openrouter are representative metered estimates used
# only for budget signalling (the binding budget source is budgets/<project>.yaml).
_METERED_RATES = {
    "grok": (1.25, 2.50),          # xAI grok-4.3 — the one always-metered lane (high-risk only)
    "vertex": (1.25, 5.00),        # Vertex Gemini 3.x pro (representative)
    "openrouter": (1.25, 5.00),    # openrouter passthrough (representative)
}

# Default context-window limits (tokens) for the A.2 lanes; the catalogue (config.yaml
# context_length) is the source of truth — these are a safe floor for routing decisions.
_CONTEXT_LIMITS = {
    "ag/gemini-3.1-pro-low": 1048576,
    "vertex/gemini-3.1-pro-preview": 1048576,
    "cx/gpt-5.4-review": 400000,
    "cx/gpt-5.5-review": 400000,
    "openrouter/x-ai/grok-4.3": 1000000,
    "openrouter/openai/gpt-5.5-pro": 400000,
    "ag/gemini-3-flash": 1048576,
}
_DEFAULT_CONTEXT_LIMIT = 131072  # conservative floor for unknown lanes
_CHARS_PER_TOKEN = 4             # coarse estimate for routing/cost


def lane_cost_tier(model_id: str) -> str:
    """Classify a lane's cost posture (flat-rate-first policy)."""
    s = (model_id or "").lower()
    if s.startswith(("cx/", "cc/", "ag/")):
        return FLAT  # Codex / Claude / Antigravity subscriptions
    if "grok" in s or "x-ai" in s:
        return METERED
    if s.startswith("vertex/") or s.startswith("openrouter/"):
        return METERED
    return METERED  # unknown → treat as metered (conservative for budgeting)


def is_grok(model_id: str) -> bool:
    s = (model_id or "").lower()
    return "grok" in s or "x-ai" in s


def assert_grok_high_risk_only(lane_ids, *, high_risk: bool) -> None:
    """Rule 5 defense-in-depth: Grok (the only always-metered lane) may appear ONLY on a
    high-risk run. Raise if a routine run somehow carries a Grok lane."""
    if high_risk:
        return
    for lane in lane_ids or []:
        if is_grok(lane):
            raise ValueError(f"cost rule: Grok lane {lane!r} not permitted on a routine run "
                             "(metered — high-risk only)")


def lane_context_limit(model_id: str) -> int:
    return _CONTEXT_LIMITS.get(model_id, _DEFAULT_CONTEXT_LIMIT)


def context_fits(artifact_chars: int, model_id: str) -> bool:
    """Rule 7: does the artifact fit the lane's context (coarse chars→tokens)?"""
    est_tokens = max(1, artifact_chars // _CHARS_PER_TOKEN)
    return est_tokens <= lane_context_limit(model_id)


def estimate_fusion_cost(lane_ids, *, est_in_tokens: int = 4000, est_out_tokens: int = 800) -> dict:
    """Estimate marginal USD for a Fusion run. Flat/free lanes contribute $0; metered lanes
    are billed by the representative per-Mtok rate. Returns a telemetry dict."""
    flat, metered, est_usd = [], [], 0.0
    for lane in lane_ids or []:
        tier = lane_cost_tier(lane)
        if tier in (FLAT, FREE):
            flat.append(lane)
            continue
        metered.append(lane)
        rate = next((r for k, r in _METERED_RATES.items() if k in (lane or "").lower()),
                    (1.25, 5.00))
        est_usd += (est_in_tokens / 1_000_000) * rate[0] + (est_out_tokens / 1_000_000) * rate[1]
    return {
        "flat_lanes": flat,
        "metered_lanes": metered,
        "est_usd": round(est_usd, 4),
        "tier": "flat" if not metered else "metered",
    }


def cost_preflight(
    lane_ids,
    *,
    high_risk: bool,
    est_in_tokens: int = 4000,
    est_out_tokens: int = 800,
    budget_remaining_usd: Optional[float] = None,
) -> Tuple[dict, Optional[str]]:
    """Pre-run cost check. Returns ``(estimate, pause_reason)``.

    ``pause_reason`` is set (rule 8) when the estimated metered spend would exceed the
    project's remaining budget — the caller routes that to ``pause_for_approval`` instead of
    spending. Flat-only runs (the routine common case) estimate ~$0 and never pause.
    """
    # Rule 5 defense-in-depth.
    assert_grok_high_risk_only(lane_ids, high_risk=high_risk)
    est = estimate_fusion_cost(lane_ids, est_in_tokens=est_in_tokens, est_out_tokens=est_out_tokens)
    if budget_remaining_usd is not None and est["est_usd"] > budget_remaining_usd:
        return est, (f"budget: estimated ${est['est_usd']} metered spend exceeds remaining "
                     f"${budget_remaining_usd} — pause_for_approval")
    return est, None


def planned_lanes(author_provider: str, *, high_risk: bool) -> list:
    """The lane ids a Fusion run WOULD use (jury primaries + judge) — for a pre-run estimate."""
    lanes = [primary for _fam, primary, _fb in routing.select_jury(author_provider, high_risk=high_risk)]
    judge_primary, _ = routing.select_judge(author_provider)
    lanes.append(judge_primary)
    return lanes
