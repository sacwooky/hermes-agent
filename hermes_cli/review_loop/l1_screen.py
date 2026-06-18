"""L1 cheap screen — non-binding first-pass triage (WI-C4, ADD-ON C v2 Phase 4).

A single fast, cheap, **non-Claude** model (default ``ag/gemini-3-flash`` via the
robin-9router OpenAI-compatible API) triages an artifact into *routine* vs *risky* and
decides whether to escalate to the full L2 review. **Non-binding**: L1 can clear routine
work and flag risky work, but it **never** issues or influences the signed verdict — that
is L2 (Phase 6). The signal is recorded as a non-binding ``l1_screen`` event and feeds the
loop-state ``l1`` slot.

Hard rules:
- **Fail-open to ESCALATE.** Any error (network, timeout, unparseable output) → ``escalate=True``
  so uncertain work always gets full review. L1 never silently clears on failure.
- **Deterministic risk floor.** If the caller marks the work high-risk (auth/payment/etc.),
  L1 escalates regardless of what the model says — the model can only *add* caution.
- Self-contained: stdlib ``urllib`` so it is trivially mockable and adds no dependency.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

_log = logging.getLogger(__name__)

DEFAULT_MODEL = "ag/gemini-3-flash"
DEFAULT_BASE_URL = "http://127.0.0.1:20128/v1"
DEFAULT_KEY_ENV = "NINEROUTER_KEY"
DEFAULT_TIMEOUT_S = 45
DEFAULT_MAX_TOKENS = 400

_TRIAGE_SYSTEM = (
    "You are a fast, cheap code-review TRIAGE screen. You do NOT review or decide — you "
    "only classify. Read the artifact and decide whether it is routine (low-risk, no obvious "
    "issues) or risky (security/auth/payment/data, large/complex, or anything you'd want a full "
    "review to scrutinise). Respond with ONLY a JSON object, no prose:\n"
    '{"risk": "routine"|"risky", "escalate": true|false, "findings_count": <int>, '
    '"summary": "<one short line>"}'
)


@dataclass(frozen=True)
class L1Result:
    risk: str            # "routine" | "risky"
    escalate: bool       # True ⇒ must get full L2 review
    findings_count: int
    summary: str
    model: str
    ok: bool             # True if the model answered and parsed; False ⇒ failed-open
    error: Optional[str] = None


def _failed_open(model: str, error: str) -> L1Result:
    # Fail-open: uncertain ⇒ escalate to full review, never clear.
    return L1Result(
        risk="risky", escalate=True, findings_count=0,
        summary=f"l1 screen unavailable — escalating ({error})",
        model=model, ok=False, error=error,
    )


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)  # first JSON-looking object
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _post_chat(base_url: str, api_key: str, model: str, artifact: str,
               *, timeout_s: int, max_tokens: int) -> str:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _TRIAGE_SYSTEM},
            {"role": "user", "content": artifact[:24000]},  # cheap screen: cap input
        ],
        "max_tokens": max_tokens,
        "stream": False,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 — operator-config URL
        raw = resp.read().decode("utf-8", "replace")
    # Tolerate SSE-ish or multi-object bodies: pull the assistant content.
    try:
        data = json.loads(raw)
        return data["choices"][0]["message"]["content"]
    except Exception:
        m = re.findall(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        joined = "".join(m)
        try:
            return json.loads(f'"{joined}"')  # unescape
        except Exception:
            return joined


def run_l1_screen(
    artifact: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    key_env: str = DEFAULT_KEY_ENV,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    force_escalate: bool = False,
) -> L1Result:
    """Triage ``artifact`` with the cheap screen. Non-binding.

    :param force_escalate: deterministic risk floor — when True the result escalates
        regardless of the model (the model may only add caution, never remove it).
    """
    api_key = os.environ.get(key_env, "")
    if not api_key:
        return _failed_open(model, f"missing {key_env}")
    try:
        content = _post_chat(base_url, api_key, model, artifact,
                             timeout_s=timeout_s, max_tokens=max_tokens)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return _failed_open(model, str(exc)[:120])

    obj = _extract_json(content)
    if not isinstance(obj, dict):
        return _failed_open(model, "unparseable model output")

    risk = str(obj.get("risk", "risky")).lower()
    escalate = bool(obj.get("escalate", True)) or risk == "risky"
    try:
        findings = int(obj.get("findings_count", 0) or 0)
    except Exception:
        findings = 0
    summary = str(obj.get("summary", ""))[:200]

    if force_escalate:
        risk, escalate = "risky", True
        if not summary:
            summary = "deterministic high-risk floor"

    return L1Result(
        risk=("risky" if escalate else "routine"),
        escalate=escalate, findings_count=findings, summary=summary,
        model=model, ok=True,
    )
