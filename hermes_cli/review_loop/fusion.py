"""L2 Robin Fusion — binding review orchestration (WI-C5 / WI-F1–F7, ADD-ON C v2 Phase 6).

Composes an independent, author-disjoint, non-PRC **jury** → checks **quorum** → an
author-disjoint **judge** synthesises a verdict candidate → a **confidence × risk** gate.
This module is PURE: stdlib-only, no kanban import, no signing, no network except through the
injected ``chat`` callable. Robin (the signing wrapper, ``scripts/review-fusion.py``) is the sole
signer; this module only decides ``pass`` / ``block`` / ``rejected``.

Fail-closed everywhere on the verdict path: below quorum, empty/unavailable lanes, a lane that
fails the independence check, an oversize artifact, or a degraded-HIGH run all resolve to
``rejected`` — **never** a pass. ``rejected`` is terminal; the wrapper must not sign it.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from hermes_cli.review_loop import ninerouter, routing

_log = logging.getLogger(__name__)

DEFAULT_MAX_ARTIFACT_CHARS = 120000

PASS = "pass"
BLOCK = "block"
REJECTED = "rejected"

_JUROR_SYSTEM = (
    "You are an INDEPENDENT code reviewer on a review panel. Review ONLY the artifact below "
    "against its stated acceptance criteria. Do not use web tools or outside context. Respond "
    "with ONLY a JSON object, no prose:\n"
    '{"verdict": "pass"|"block", "severity": "none"|"low"|"medium"|"high"|"critical", '
    '"findings": ["..."], "summary": "<one short line>"}'
)
_JUDGE_SYSTEM = (
    "You are the author-disjoint JUDGE. Synthesise the independent jury reviews of the artifact "
    "into a single verdict. Weigh consensus, contradictions, and coverage gaps. A credible blocking "
    "finding from any juror should not be waved through. Respond with ONLY a JSON object, the verdict "
    "FIRST, no prose before it:\n"
    '{"verdict": "pass"|"block", "confidence": 0.0-1.0, "severity": "none|low|medium|high|critical", '
    '"consensus": "<line>", "contradictions": ["..."], "coverage_gaps": ["..."], "rationale": "<short>"}'
)


@dataclass(frozen=True)
class FusionResult:
    verdict: str                       # "pass" | "block" | "rejected"
    confidence: dict
    jury: list = field(default_factory=list)
    judge: dict = field(default_factory=dict)
    fusion_run_id: str = ""
    lanes_run: list = field(default_factory=list)
    degraded: bool = False
    reject_reason: Optional[str] = None


def mint_fusion_run_id(task_id: str, commit: str, recorded_at) -> str:
    """Deterministic id (the verdict lane id). Minted by the Robin wrapper, not run_fusion."""
    h = hashlib.sha256(f"{task_id}\n{commit}\n{recorded_at}".encode("utf-8")).hexdigest()
    return "fusion-" + h[:16]


def _rejected(reason: str, fusion_run_id: str, jury=None, judge=None, lanes_run=None) -> FusionResult:
    return FusionResult(
        verdict=REJECTED,
        confidence={"capacity": "none", "judge_confidence": 0.0, "composite": 0.0,
                    "missing": [], "divergence": 0.0},
        jury=jury or [], judge=judge or {}, fusion_run_id=fusion_run_id,
        lanes_run=lanes_run or [], degraded=True, reject_reason=reason,
    )


def _run_lane(chat, lane, messages, *, base_url, key_env, timeout_s, max_tokens, backoff_s):
    """Call one lane with bounded backoff on transient (ok=False) results."""
    last = None
    for i, delay in enumerate(backoff_s or (0.0,)):
        if delay and i:
            time.sleep(delay)
        res = chat(lane, messages, base_url=base_url, key_env=key_env,
                   timeout_s=timeout_s, max_tokens=max_tokens, json_mode=True,
                   reasoning_effort="low")
        last = res
        if res.ok:
            return res
    return last


def _juror_review(chat, family, primary, fallback, artifact, *, author_provider,
                  base_url, key_env, timeout_s, max_tokens, backoff_s):
    """Run one jury seat (primary → fallback). Returns a seat dict.

    Raises ValueError (via assert_lane_allowed) if a configured lane is independence-barred —
    the caller treats that as a run-level rejection, never a 'juror down'.
    """
    messages = [{"role": "system", "content": _JUROR_SYSTEM},
                {"role": "user", "content": artifact}]
    used_fallback = False
    routing.assert_lane_allowed(primary, author_provider=author_provider)
    res = _run_lane(chat, primary, messages, base_url=base_url, key_env=key_env,
                    timeout_s=timeout_s, max_tokens=max_tokens, backoff_s=backoff_s)
    lane_used = primary
    if (res is None or not res.ok) and fallback:
        routing.assert_lane_allowed(fallback, author_provider=author_provider)
        res = _run_lane(chat, fallback, messages, base_url=base_url, key_env=key_env,
                        timeout_s=timeout_s, max_tokens=max_tokens, backoff_s=backoff_s)
        used_fallback = True
        lane_used = fallback

    seat = {"family": family, "lane": lane_used, "used_fallback": used_fallback,
            "ok": False, "verdict": None, "severity": None, "findings": [], "error": None}
    if res is None or not res.ok:
        seat["error"] = (res.error if res else "no result")
        return seat
    obj = res.as_json()
    if not isinstance(obj, dict) or obj.get("verdict") not in (PASS, BLOCK):
        seat["error"] = "unparseable juror output"
        return seat
    seat.update(ok=True, verdict=obj.get("verdict"), severity=obj.get("severity"),
                findings=obj.get("findings") or [])
    return seat


def run_fusion(
    artifact: str,
    *,
    author_provider: str,
    fusion_run_id: str,
    risk: str = "routine",
    high_risk: bool = False,
    chat: Callable = ninerouter.chat,
    base_url: str = ninerouter.DEFAULT_BASE_URL,
    key_env: str = ninerouter.DEFAULT_KEY_ENV,
    juror_timeout_s: int = 120,
    judge_timeout_s: int = 120,
    juror_max_tokens: int = 1200,
    judge_max_tokens: int = 1500,
    max_artifact_chars: int = DEFAULT_MAX_ARTIFACT_CHARS,
    backoff_s=(0.0, 1.0, 3.0),
) -> FusionResult:
    """Run the binding Fusion review. Returns a :class:`FusionResult`; never raises."""
    hr = bool(high_risk) or risk == "high"

    if not artifact or not artifact.strip():
        return _rejected("empty artifact", fusion_run_id)
    if len(artifact) > max_artifact_chars:
        return _rejected(f"artifact_exceeds_context: {len(artifact)} > {max_artifact_chars}", fusion_run_id)

    # --- Jury ---
    try:
        seats = routing.select_jury(author_provider, high_risk=hr)
    except Exception as exc:
        return _rejected(f"jury selection failed: {exc}", fusion_run_id)

    jury = []
    try:
        for family, primary, fallback in seats:
            jury.append(_juror_review(
                chat, family, primary, fallback, artifact,
                author_provider=author_provider, base_url=base_url, key_env=key_env,
                timeout_s=juror_timeout_s, max_tokens=juror_max_tokens, backoff_s=backoff_s,
            ))
    except ValueError as exc:
        return _rejected(f"lane_independence_violation: {exc}", fusion_run_id, jury=jury)

    needed = 3 if hr else 2
    responded = [j for j in jury if j["ok"]]
    lanes_run = [j["lane"] for j in responded]
    if len(responded) < needed:
        return _rejected(f"below_quorum: {len(responded)}/{needed}", fusion_run_id,
                         jury=jury, lanes_run=lanes_run)

    degraded = any(j["used_fallback"] for j in responded) or len(responded) < len(seats)
    missing = [j["family"] for j in jury if not j["ok"]]

    # --- Judge (author-disjoint) ---
    judge_primary, judge_fallback = routing.select_judge(author_provider)
    judge_msgs = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": "ARTIFACT (summary):\n" + artifact[:8000]
         + "\n\nJURY REVIEWS:\n" + _format_jury(responded)},
    ]
    try:
        routing.assert_lane_allowed(judge_primary, author_provider=author_provider)
    except ValueError as exc:
        return _rejected(f"lane_independence_violation: judge {exc}", fusion_run_id,
                         jury=jury, lanes_run=lanes_run)
    jres = _run_lane(chat, judge_primary, judge_msgs, base_url=base_url, key_env=key_env,
                     timeout_s=judge_timeout_s, max_tokens=judge_max_tokens, backoff_s=backoff_s)
    judge_lane = judge_primary
    judge_on_fallback = False
    if (jres is None or not jres.ok) and judge_fallback:
        try:
            routing.assert_lane_allowed(judge_fallback, author_provider=author_provider)
        except ValueError as exc:
            return _rejected(f"lane_independence_violation: judge_fallback {exc}", fusion_run_id,
                             jury=jury, lanes_run=lanes_run)
        jres = _run_lane(chat, judge_fallback, judge_msgs, base_url=base_url, key_env=key_env,
                         timeout_s=judge_timeout_s, max_tokens=judge_max_tokens, backoff_s=backoff_s)
        judge_lane = judge_fallback
        judge_on_fallback = True

    if jres is None or not jres.ok:
        return _rejected("judge_unavailable", fusion_run_id, jury=jury, lanes_run=lanes_run)
    jobj = jres.as_json()
    if not isinstance(jobj, dict) or jobj.get("verdict") not in (PASS, BLOCK):
        return _rejected("judge_unparseable", fusion_run_id, jury=jury, lanes_run=lanes_run)

    judge = {"lane": judge_lane, "ok": True, "used_fallback": judge_on_fallback,
             "verdict": jobj.get("verdict"),
             "confidence": _as_float(jobj.get("confidence"), 0.5),
             "consensus": jobj.get("consensus", ""),
             "contradictions": jobj.get("contradictions") or []}
    lanes_run = lanes_run + [judge_lane]

    judge_verdict = judge["verdict"]
    # Downgrade-only same-family fallback: a fallback judge may FAIL, not PASS over a juror block.
    if judge_on_fallback and judge_verdict == PASS and any(j["verdict"] == BLOCK for j in responded):
        judge_verdict = BLOCK
        judge["downgraded_on_fallback"] = True

    # --- Confidence × risk ---
    divergence = (sum(1 for j in responded if j["verdict"] != judge_verdict) / len(responded)
                  if responded else 0.0)
    capacity = "full" if not degraded else "degraded"
    cap_score = 1.0 if capacity == "full" else 0.5
    confidence = {
        "capacity": capacity,
        "judge_confidence": judge["confidence"],
        "composite": round(min(cap_score, judge["confidence"]), 4),
        "missing": missing,
        "divergence": round(divergence, 4),
    }

    if judge_verdict == BLOCK:
        verdict = BLOCK
    elif capacity == "full":
        verdict = PASS
    elif not hr:
        verdict = PASS  # degraded + routine → proceed + flag
    else:
        return _rejected("degraded_high_no_autopass", fusion_run_id,
                         jury=jury, judge=judge, lanes_run=lanes_run)

    return FusionResult(
        verdict=verdict, confidence=confidence, jury=jury, judge=judge,
        fusion_run_id=fusion_run_id, lanes_run=lanes_run, degraded=degraded, reject_reason=None,
    )


def _format_jury(responded: list) -> str:
    lines = []
    for j in responded:
        lines.append(f"- [{j['family']} / {j['lane']}] verdict={j['verdict']} "
                     f"severity={j['severity']} findings={j['findings']}")
    return "\n".join(lines)


def _as_float(v, default: float) -> float:
    try:
        f = float(v)
        return f if 0.0 <= f <= 1.0 else default
    except (TypeError, ValueError):
        return default
