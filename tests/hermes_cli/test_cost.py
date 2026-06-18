"""Tests for ADD-ON C v2 Phase 7 — encoded cost-routing (WI-C7). Pure, LLM-free."""

from __future__ import annotations

import pytest

from hermes_cli.review_loop import cost as C
from hermes_cli.review_loop import routing as R


# --- tiers ---
@pytest.mark.parametrize("lane,tier", [
    ("cx/gpt-5.4-review", C.FLAT),
    ("cx/gpt-5.5-review", C.FLAT),
    ("cc/claude-opus-4-8", C.FLAT),
    ("ag/gemini-3.1-pro-low", C.FLAT),
    ("ag/gemini-3-flash", C.FLAT),
    ("openrouter/x-ai/grok-4.3", C.METERED),
    ("vertex/gemini-3.1-pro-preview", C.METERED),
    ("openrouter/openai/gpt-5.5-pro", C.METERED),
])
def test_lane_cost_tier(lane, tier):
    assert C.lane_cost_tier(lane) == tier


# --- grok high-risk-only guard (rule 5) ---
def test_grok_allowed_on_high_risk():
    C.assert_grok_high_risk_only(["cx/gpt-5.4-review", "openrouter/x-ai/grok-4.3"], high_risk=True)


def test_grok_barred_on_routine():
    with pytest.raises(ValueError):
        C.assert_grok_high_risk_only(["cx/gpt-5.4-review", "openrouter/x-ai/grok-4.3"], high_risk=False)


def test_routine_jury_has_no_grok():
    # The structural guarantee: select_jury routine never includes Grok.
    lanes = [p for _f, p, _fb in R.select_jury(R.CLAUDE, high_risk=False)]
    C.assert_grok_high_risk_only(lanes, high_risk=False)  # must not raise
    assert not any(C.is_grok(l) for l in lanes)


def test_high_risk_jury_includes_grok():
    lanes = [p for _f, p, _fb in R.select_jury(R.CLAUDE, high_risk=True)]
    assert any(C.is_grok(l) for l in lanes)


# --- estimate (flat-first) ---
def test_routine_estimate_is_zero_flat():
    # routine Claude-authored jury = flat lanes + flat judge ⇒ ~$0 marginal
    lanes = C.planned_lanes(R.CLAUDE, high_risk=False)
    est = C.estimate_fusion_cost(lanes)
    assert est["est_usd"] == 0.0 and est["metered_lanes"] == [] and est["tier"] == "flat"


def test_high_risk_estimate_has_metered_grok():
    lanes = C.planned_lanes(R.CLAUDE, high_risk=True)
    est = C.estimate_fusion_cost(lanes)
    assert est["est_usd"] > 0.0
    assert any(C.is_grok(l) for l in est["metered_lanes"])
    assert est["tier"] == "metered"


# --- context routing (rule 7) ---
def test_context_fits_small_artifact():
    assert C.context_fits(4000, "cx/gpt-5.4-review") is True


def test_context_overflow_flagged():
    # > cx 400k-token limit (~1.6M chars)
    assert C.context_fits(2_000_000, "cx/gpt-5.4-review") is False


# --- budget gate (rule 8) ---
def test_preflight_flat_never_pauses():
    lanes = C.planned_lanes(R.CLAUDE, high_risk=False)
    est, pause = C.cost_preflight(lanes, high_risk=False, budget_remaining_usd=0.0)
    assert pause is None and est["est_usd"] == 0.0  # flat-first: $0 ⇒ no pause even at $0 budget


def test_preflight_over_cap_pauses():
    lanes = C.planned_lanes(R.CLAUDE, high_risk=True)  # metered Grok
    est, pause = C.cost_preflight(lanes, high_risk=True, budget_remaining_usd=0.0001)
    assert pause is not None and "budget" in pause


def test_preflight_grok_on_routine_raises():
    with pytest.raises(ValueError):
        C.cost_preflight(["openrouter/x-ai/grok-4.3"], high_risk=False)
