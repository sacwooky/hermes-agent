"""Tests for ADD-ON C v2 Phase 5 — author-aware routing + A.3 exclusions (WI-C6) and the
robin-9router client (WI-C11). Pure + mocked-HTTP; LLM-free."""

from __future__ import annotations

import pytest

from hermes_cli.review_loop import ninerouter as NR
from hermes_cli.review_loop import routing as R


# ---------------------------------------------------------------------------
# provider_of
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mid,fam", [
    ("cx/gpt-5.5-review", R.OPENAI),
    ("cc/claude-opus-4-8", R.CLAUDE),
    ("ag/gemini-3.1-pro-low", R.GEMINI),
    ("vertex/gemini-3.1-pro-preview", R.GEMINI),
    ("openrouter/x-ai/grok-4.3", R.XAI),
    ("openrouter/anthropic/claude-fable-5", R.CLAUDE),
    ("openrouter/openai/gpt-5.5-pro", R.OPENAI),
    ("openrouter/google/gemini-3.5-flash", R.GEMINI),
    ("openrouter/deepseek/deepseek-v4-pro", R.PRC),
    ("openrouter/qwen/qwen3.7-plus", R.PRC),
    ("openrouter/nvidia/nemotron-3-ultra-550b-a55b:free", R.PRC),  # PRC-lineage
])
def test_provider_of(mid, fam):
    assert R.provider_of(mid) == fam


# ---------------------------------------------------------------------------
# exclusions (A.3)
# ---------------------------------------------------------------------------


def test_exclude_free_tier():
    ex, _ = R.is_excluded_binding_lane("openrouter/openai/gpt-oss-120b:free", author_provider=R.CLAUDE)
    assert ex is True


def test_exclude_fusion_meta_router():
    ex, _ = R.is_excluded_binding_lane("openrouter/openrouter/fusion:general-high", author_provider=R.CLAUDE)
    assert ex is True


def test_exclude_nemotron_lineage():
    ex, _ = R.is_excluded_binding_lane("openrouter/nvidia/nemotron-3-super-120b-a12b", author_provider=R.CLAUDE)
    assert ex is True


def test_exclude_prc():
    ex, _ = R.is_excluded_binding_lane("openrouter/deepseek/deepseek-v4-pro", author_provider=R.CLAUDE)
    assert ex is True


def test_exclude_self_family():
    # A Claude lane cannot review Claude-authored work.
    ex, reason = R.is_excluded_binding_lane("cc/claude-opus-4-8", author_provider=R.CLAUDE)
    assert ex is True and "self-family" in reason


def test_allow_cross_family():
    ex, _ = R.is_excluded_binding_lane("cx/gpt-5.4-review", author_provider=R.CLAUDE)
    assert ex is False


def test_assert_lane_allowed_raises_on_excluded():
    with pytest.raises(ValueError):
        R.assert_lane_allowed("openrouter/deepseek/deepseek-v4-pro", author_provider=R.CLAUDE)


def test_assert_lane_allowed_ok_on_cross_family():
    R.assert_lane_allowed("openrouter/x-ai/grok-4.3", author_provider=R.CLAUDE)  # no raise


# ---------------------------------------------------------------------------
# jury / judge selection
# ---------------------------------------------------------------------------


def test_jury_routine_is_two_disjoint(author=R.CLAUDE):
    jury = R.select_jury(R.CLAUDE, high_risk=False)
    assert len(jury) == 2
    fams = {f for f, _, _ in jury}
    assert R.CLAUDE not in fams  # author-disjoint


def test_jury_high_risk_is_three():
    jury = R.select_jury(R.CLAUDE, high_risk=True)
    assert len(jury) == 3
    assert {f for f, _, _ in jury} == {R.GEMINI, R.OPENAI, R.XAI}


def test_jury_excludes_author_family_openai():
    jury = R.select_jury(R.OPENAI, high_risk=True)
    assert R.OPENAI not in {f for f, _, _ in jury}


def test_judge_for_claude_is_openai():
    primary, fallback = R.select_judge(R.CLAUDE)
    assert primary == "cx/gpt-5.5-review" and R.provider_of(primary) == R.OPENAI


def test_judge_is_author_disjoint_for_openai_author():
    primary, _ = R.select_judge(R.OPENAI)
    assert R.provider_of(primary) != R.OPENAI


def test_jury_lanes_are_not_excluded_for_claude_author():
    for _fam, primary, _fb in R.select_jury(R.CLAUDE, high_risk=True):
        R.assert_lane_allowed(primary, author_provider=R.CLAUDE)  # none raise


# ---------------------------------------------------------------------------
# ninerouter client (mocked HTTP)
# ---------------------------------------------------------------------------


def test_chat_missing_key_fails_safe(monkeypatch):
    monkeypatch.delenv("NINEROUTER_KEY", raising=False)
    r = NR.chat("cx/gpt-5.5-review", [{"role": "user", "content": "hi"}])
    assert r.ok is False and "missing" in r.error


def test_chat_parses_content(monkeypatch):
    monkeypatch.setenv("NINEROUTER_KEY", "k")
    body = '{"choices":[{"message":{"role":"assistant","content":"{\\"verdict\\":\\"PASS\\"}"}}]}'

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return body.encode()

    monkeypatch.setattr(NR.urllib.request, "urlopen", lambda *a, **k: _Resp())
    r = NR.chat("cx/gpt-5.5-review", [{"role": "user", "content": "x"}], json_mode=True)
    assert r.ok is True
    assert r.as_json() == {"verdict": "PASS"}


def test_chat_network_error_fails_safe(monkeypatch):
    monkeypatch.setenv("NINEROUTER_KEY", "k")

    def _boom(*a, **k):
        raise NR.urllib.error.URLError("refused")

    monkeypatch.setattr(NR.urllib.request, "urlopen", _boom)
    r = NR.chat("cx/gpt-5.5-review", [{"role": "user", "content": "x"}])
    assert r.ok is False and r.content == ""


def test_chat_empty_content_fails_safe(monkeypatch):
    monkeypatch.setenv("NINEROUTER_KEY", "k")

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"choices":[{"message":{"content":""}}]}'

    monkeypatch.setattr(NR.urllib.request, "urlopen", lambda *a, **k: _Resp())
    r = NR.chat("cx/gpt-5.5-review", [{"role": "user", "content": "x"}])
    assert r.ok is False and "empty" in r.error


def test_chat_sets_json_and_reasoning_payload(monkeypatch):
    monkeypatch.setenv("NINEROUTER_KEY", "k")
    seen = {}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"choices":[{"message":{"content":"ok"}}]}'

    def _capture(req, *a, **k):
        import json as _j
        seen.update(_j.loads(req.data.decode()))
        return _Resp()

    monkeypatch.setattr(NR.urllib.request, "urlopen", _capture)
    NR.chat("cx/gpt-5.5-review", [{"role": "user", "content": "x"}],
            json_mode=True, reasoning_effort="low")
    assert seen["response_format"] == {"type": "json_object"}
    assert seen["reasoning_effort"] == "low"
    assert seen["stream"] is False
