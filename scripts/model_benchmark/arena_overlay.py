#!/usr/bin/env python3
"""Hermes Arena Elo overlay (Layer 3, read side).

Loads the Bradley-Terry ratings produced by `arena_eval.py` (our own blind A/B
agent evals on real lanes, seeded from public LMArena Elo) and exposes them to
build_benchmark. This is the "inject our real agent benchmark scores" layer:
where we have enough of our own matches for a model, its Hermes Arena Elo is the
authoritative agent-quality signal; otherwise the AA v4.1 baseline stands alone.

Write side: arena_eval.py → model-benchmark-arena.json.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
ARENA_PATH = os.path.join(REPO_ROOT, "web", "public", "data", "model-benchmark-arena.json")

# Minimum of OUR own matches before an Elo is trusted enough to surface as "real".
MIN_MATCHES = 6


def load():
    """Return (meta, {router_id: rating_record}) or ({}, {}) if absent."""
    if not os.path.exists(ARENA_PATH):
        return {}, {}
    try:
        with open(ARENA_PATH, encoding="utf-8") as fh:
            d = json.load(fh)
        return d.get("meta", {}), d.get("ratings", {}) or {}
    except (OSError, ValueError):
        return {}, {}


def attach(models):
    """Attach an `arena` block to each model that has ratings. Returns the meta."""
    meta, ratings = load()
    if not ratings:
        return None
    for m in models:
        rec = ratings.get(m.get("router") or "")
        if not rec:
            continue
        m["arena"] = {
            "elo": rec.get("elo"),
            "seed_elo": rec.get("seed_elo"),
            "matches": rec.get("matches", 0),
            "wins": rec.get("wins", 0),
            # "real" once we've run enough of our OWN matches (not just the seed prior)
            "settled": rec.get("matches", 0) >= MIN_MATCHES,
        }
    return meta
