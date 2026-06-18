"""robin-9router OpenAI-compatible client for the review path (WI-C11, ADD-ON C v2 Phase 5).

Addendum A.1: every review/verdict call goes through the robin-9router API (no CLI prompts),
which lets the caller pass the exact model id and pin the judge contract — **JSON verdict
before prose** (``json_mode``) with **bounded reasoning** (``reasoning_effort``), both verified
live in run 478 on ``cx/gpt-5.5-review``.

Stdlib ``urllib`` only (trivially mockable). Returns ``ChatResult`` — never raises on a
transport/parse failure; ``ok=False`` carries the error so the caller (the Fusion gate) can
treat an empty/failed lane as ``verdict_rejected`` rather than a pass.
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

DEFAULT_BASE_URL = "http://127.0.0.1:20128/v1"
DEFAULT_KEY_ENV = "NINEROUTER_KEY"
DEFAULT_TIMEOUT_S = 120


@dataclass(frozen=True)
class ChatResult:
    content: str
    model: str
    ok: bool
    error: Optional[str] = None

    def as_json(self) -> Optional[dict]:
        """Parse ``content`` as JSON (json_mode), tolerating prose-wrapped output."""
        if not self.content:
            return None
        try:
            return json.loads(self.content)
        except Exception:
            m = re.search(r"\{.*\}", self.content, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return None
        return None


def _content_from_body(raw: str) -> str:
    try:
        data = json.loads(raw)
        return data["choices"][0]["message"]["content"]
    except Exception:
        # SSE / multi-object body: stitch assistant content fragments.
        frags = re.findall(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        joined = "".join(frags)
        try:
            return json.loads(f'"{joined}"')
        except Exception:
            return joined


def chat(
    model: str,
    messages: list,
    *,
    base_url: str = DEFAULT_BASE_URL,
    key_env: str = DEFAULT_KEY_ENV,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    max_tokens: int = 1024,
    json_mode: bool = False,
    reasoning_effort: Optional[str] = None,
) -> ChatResult:
    """Call robin-9router ``/chat/completions``. Never raises; returns ``ChatResult``.

    :param json_mode: set ``response_format={"type":"json_object"}`` (judge verdict-before-prose).
    :param reasoning_effort: ``"low"|"medium"|"high"`` — bounded reasoning (cost + determinism).
    """
    api_key = os.environ.get(key_env, "")
    if not api_key:
        return ChatResult(content="", model=model, ok=False, error=f"missing {key_env}")

    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 — operator-config URL
            raw = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return ChatResult(content="", model=model, ok=False, error=str(exc)[:160])

    content = _content_from_body(raw)
    if not content:
        return ChatResult(content="", model=model, ok=False, error="empty content from model")
    return ChatResult(content=content, model=model, ok=True)
