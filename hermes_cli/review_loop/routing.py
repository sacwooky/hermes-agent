"""Author-aware review routing + binding-lane exclusions (WI-C6, ADD-ON C v2 Phase 5).

Pure (no network). Encodes who may sit on a binding review seat for a given author, and
the hard exclusions from Addendum A.3. Phase 6's Robin Fusion gate calls
:func:`assert_lane_allowed` before a lane is allowed to produce a binding verdict, and
:func:`select_jury` / :func:`select_judge` to compose the panel.

Independence rule (WI-C6): a binding verdict whose lane family == the work's author family
is invalid. Claude authors the bulk, so its jury is {Gemini, OpenAI, xAI} and its judge is
GPT-5.5. The judge is always author-disjoint.

Exclusions (A.3), never on a binding/review lane:
- ``:free`` models (free = training tier)
- ``fusion:*`` meta-routers (non-deterministic — breaks model pinning / verdict integrity)
- Nemotron (open-weight; PRC-lineage reward model)
- PRC families (deepseek / qwen / minimax / kimi / glm / mimo / ...)
- the author's own family (self-review)
"""

from __future__ import annotations

from typing import Tuple

# Provider families.
CLAUDE = "claude"
GEMINI = "gemini"
OPENAI = "openai"
XAI = "xai"
PRC = "prc"
OTHER = "other"

_PRC_MARKERS = ("deepseek", "qwen", "minimax", "kimi", "glm", "moonshot", "mimo", "zhipu", "z-ai", "ernie", "baidu")
_NEMOTRON_MARKERS = ("nemotron",)  # PRC-lineage reward/distillation


def provider_of(model_id: str) -> str:
    """Map a robin-9router model id (the path) to a provider family."""
    s = (model_id or "").lower()
    # PRC and PRC-lineage first (so they're never misread as their host prefix).
    if any(m in s for m in _PRC_MARKERS):
        return PRC
    if any(m in s for m in _NEMOTRON_MARKERS):
        return PRC  # treat PRC-lineage as PRC for exclusion purposes
    if "x-ai" in s or "grok" in s:
        return XAI
    if "claude" in s or "anthropic" in s or s.startswith("cc/"):
        return CLAUDE
    if "gpt" in s or "openai" in s or s.startswith("cx/"):
        return OPENAI
    if "gemini" in s or "gemma" in s or s.startswith("vertex/") or s.startswith("ag/gemini"):
        return GEMINI
    return OTHER


def is_excluded_binding_lane(model_id: str, *, author_provider: str) -> Tuple[bool, str]:
    """Return ``(excluded, reason)`` for a *binding/verdict* lane per A.3 + WI-C6."""
    s = (model_id or "").lower()
    if not s:
        return True, "empty model id"
    if ":free" in s:
        return True, "free tier (training-on-data) — barred from review/verdict lanes"
    if "fusion:" in s:
        return True, "fusion:* meta-router — non-deterministic, breaks model pinning"
    fam = provider_of(model_id)
    if fam == PRC:
        return True, "PRC (or PRC-lineage) model — barred everywhere"
    if author_provider and fam == author_provider and fam != OTHER:
        return True, f"self-family ({fam}) cannot review {author_provider}-authored work"
    return False, ""


def assert_lane_allowed(model_id: str, *, author_provider: str) -> None:
    """Raise ``ValueError`` if ``model_id`` may not sit on a binding lane for this author.

    Phase 6's Fusion gate calls this before counting any lane toward a verdict.
    """
    excluded, reason = is_excluded_binding_lane(model_id, author_provider=author_provider)
    if excluded:
        raise ValueError(f"lane {model_id!r} barred from binding review: {reason}")


# --- A.2 concrete bindings (verified live, runs 478/479) ---------------------
# Jury seat → (primary lane, fallback lane). Antigravity-first → Vertex fallback for Google.
JURY_SEATS = {
    GEMINI: ("ag/gemini-3.1-pro-low", "vertex/gemini-3.1-pro-preview"),
    OPENAI: ("cx/gpt-5.4-review", None),
    XAI: ("openrouter/x-ai/grok-4.3", None),
}

# Author family → (judge primary, judge fallback). Judge is author-disjoint.
JUDGE_FOR_AUTHOR = {
    CLAUDE: ("cx/gpt-5.5-review", "openrouter/openai/gpt-5.5-pro"),
    OPENAI: ("cc/claude-opus-4-8", None),
    GEMINI: ("cx/gpt-5.5-review", "openrouter/openai/gpt-5.5-pro"),
    OTHER: ("cx/gpt-5.5-review", "openrouter/openai/gpt-5.5-pro"),
}


def select_jury(author_provider: str, *, high_risk: bool) -> list:
    """Compose the author-disjoint, non-PRC jury.

    Routine = 2 substantive seats; high-risk / gating / sensitive = all 3 (true 3/3).
    Drops any seat whose family == author (independence) — for the Claude-authored bulk
    all three of {Gemini, OpenAI, xAI} qualify.
    """
    seats = [
        (fam, primary, fallback)
        for fam, (primary, fallback) in JURY_SEATS.items()
        if fam != author_provider
    ]
    n = 3 if high_risk else 2
    return seats[:n]


def select_judge(author_provider: str) -> Tuple[str, str]:
    """Return ``(judge_primary, judge_fallback)`` — always author-disjoint."""
    primary, fallback = JUDGE_FOR_AUTHOR.get(author_provider, JUDGE_FOR_AUTHOR[OTHER])
    # Safety: never return a same-family judge.
    if provider_of(primary) == author_provider:
        return JUDGE_FOR_AUTHOR[OTHER]
    return primary, fallback
