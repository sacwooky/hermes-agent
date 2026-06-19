#!/usr/bin/env python3
"""Authoritative model roster for the Hermes model benchmark.

Every value here is transcribed from the conductor-vault model matrices
(``wiki/models/matrices/*.md``, ``last_verified: 2026-06-16``) — the fleet's
source of truth. ``build_benchmark.py --verify`` re-parses those matrices and
warns on any drift, so this file is a fast, structured cache of the vault, not
a competing source.

Field notes
-----------
* ``router``        — the id used to reach the model via ``custom:9router-codex``
                      (``config.yaml``). ``None`` = reachable only off-9router or
                      not wired. Lane assignments are matched back to models by
                      this id.
* ``aa``            — Artificial Analysis Intelligence Index (capability matrix
                      snapshot, 2026-06-15). ``None`` where not scored.
* ``intel_est``     — capability estimate (0..70 AA-equivalent) used ONLY when
                      ``aa`` is ``None``. Flagged ``intelligence_estimated`` in
                      output so the UI can mark it.
* ``cin``/``cout``  — USD per MTok in/out (cost matrix). ``None`` = not published;
                      ``cost_estimated`` is set and a family neighbour is used.
* ``ctx``           — usable context window (tokens).
* ``vision``        — accepts image input (modalities matrix).
* ``tool``          — tool-use tier label → numeric in build_benchmark.py.
* ``prc``           — PRC-jurisdiction provider.
* ``trains``        — provider trains on API inputs (hard-excluded from
                      privacy-critical lanes).
* ``avail``         — usable on the fleet today (availability matrix / roster).
* ``authority``     — max output authority the model is trusted with:
                      chat | binding | none (none = classifier/ASR, not a chat LLM).
"""
from __future__ import annotations

# fmt: off
ROSTER = [
    # display, family, router id, aa, intel_est, cin, cout, ctx, maxout, vision, tool, prc, trains, avail, authority, strong, weak, note
    dict(display="Claude Opus 4.8", family="anthropic", router="cc/claude-opus-4-8", aa=61, cin=5, cout=25, ctx=1_000_000, maxout=128_000, vision=True, tool="full", prc=False, trains=False, avail=True, authority="chat",
         strong="agentic, review, knowledge work, planning", weak="cost-bulk", note="Effective top Claude; review primary for non-Claude work + host default."),
    dict(display="Claude Sonnet 4.6", family="anthropic", router="cc/claude-sonnet-4-6", aa=43, cin=3, cout=15, ctx=1_000_000, maxout=64_000, vision=True, tool="full", prc=False, trains=False, avail=True, authority="chat",
         strong="balanced general/coding", weak="hardest long-horizon. AA 43 is an estimate corrected 2026-06-17 (was 52, which was impossibly above the verified Opus 4.6=47); anchored to fit the verified Claude ladder Haiku 37 < Sonnet 4.6 < Opus 4.6 47"),
    dict(display="Claude Haiku 4.5", family="anthropic", router="cc/claude-haiku-4-5-20251001", aa=None, intel_est=42, cin=1, cout=5, ctx=200_000, maxout=64_000, vision=True, tool="tooluse", prc=False, trains=False, avail=True, authority="chat",
         strong="fast triage/classification", weak="high-judgment review", note="Own quota pool; no effort:max."),
    dict(display="Claude Fable 5", family="anthropic", router="openrouter/anthropic/claude-fable-5", aa=65, cin=10, cout=50, ctx=1_000_000, maxout=128_000, vision=True, tool="full", prc=False, trains=False, avail=False, authority="chat",
         strong="nominal flagship", weak="30-day retention, no ZDR; DO-NOT-PIN", note="UNAVAILABLE to everyone (MR-1)."),
    dict(display="GPT-5.5", family="openai", router="cx/gpt-5.5", aa=60, cin=5, cout=30, ctx=1_050_000, maxout=128_000, vision=True, tool="gpt", prc=False, trains=False, avail=True, authority="chat",
         strong="orchestration, long-horizon agentic, tool use", weak="text out only", note="Hermes PRIMARY agent/chat (config.yaml model.default)."),
    dict(display="GPT-5.3-Codex", family="openai", router="cx/gpt-5.3-codex", aa=54, cin=1.75, cout=14, ctx=400_000, maxout=128_000, vision=True, tool="gpt", prc=False, trains=False, avail=False, authority="chat",
         strong="coding-specialized agentic coding", weak="NOT AVAILABLE on this host's auth: 'gpt-5.3-codex is not supported when using Codex with a ChatGPT account' (verified 2026-06-17). Would need an OpenAI API-key path, not the ChatGPT-account Codex auth."),
    dict(display="GPT-5.3-Codex Spark", family="openai", router="cx/gpt-5.3-codex-spark", aa=None, intel_est=48, cin=1.0, cout=8, ctx=400_000, maxout=128_000, vision=True, tool="gpt", prc=False, trains=False, avail=True, authority="chat", cost_estimated=True, speed_est=80,
         strong="lighter/FAST coding-specialized Codex variant; WORKS on this host's ChatGPT-account Codex auth (verified 2026-06-17) — unlike plain gpt-5.3-codex; sub-3s substantial output", weak="lower capability than full Codex (AA est); coding-focused. Speed is an estimate — the e2e harness returned an implausible 586 tps (endpoint ignores max_tokens / buffers), rejected by the >300 sanity cap."),
    dict(display="GPT-5.4", family="openai", router="cx/gpt-5.4", aa=None, intel_est=57, cin=2.5, cout=15, ctx=1_050_000, maxout=128_000, vision=True, tool="gpt", prc=False, trains=False, avail=True, authority="chat",
         strong="coding + tool use, doc understanding", weak=">272K input doubles cost"),
    dict(display="GPT-5.4-mini", family="openai", router="cx/gpt-5.4-mini", aa=None, intel_est=48, cin=0.75, cout=4.5, ctx=400_000, maxout=128_000, vision=True, tool="gpt", prc=False, trains=False, avail=True, authority="chat",
         strong="high-volume coding/subagents/computer use", weak="hardest reasoning"),
    dict(display="GPT-5.5-pro", family="openai", router="openrouter/openai/gpt-5.5-pro", aa=None, intel_est=63, cin=30, cout=180, ctx=1_050_000, maxout=128_000, vision=True, tool="gpt", prc=False, trains=False, avail=True, authority="chat",
         strong="deep reasoning, consistency", weak="no streaming; very expensive", note="Via OpenRouter passthrough."),
    dict(display="GPT-chat-latest", family="openai", router="openrouter/openai/gpt-chat-latest", aa=None, intel_est=50, cin=5, cout=30, ctx=400_000, maxout=128_000, vision=True, tool="gpt", prc=False, trains=False, avail=True, authority="chat",
         strong="fast non-reasoning chat", weak="floating alias, not reproducible", note="Via OpenRouter passthrough."),
    dict(display="Gemini 2.5 Pro", family="google", router="vertex/gemini-2.5-pro", aa=35, cin=1.25, cout=10, ctx=1_048_576, maxout=65_536, vision=True, tool="gemini", prc=False, trains=False, avail=True, authority="binding",
         strong="long-context reasoning; interim binding reviewer (non-PRC)", weak="lower AA index; promoted to binding by interim policy 2026-06-17 pending a permanent reviewer",
         note="INTERIM binding reviewer after PRC models removed (decision 2026-06-17-no-prc-or-training-models). Was advisory-only."),
    dict(display="Gemini 2.5 Flash", family="google", router="vertex/gemini-2.5-flash", aa=None, intel_est=34, cin=0.30, cout=2.5, ctx=1_048_576, maxout=65_536, vision=True, tool="gemini", prc=False, trains=False, avail=True, authority="chat",
         strong="cost-disciplined fallback + vision + session driver", weak="not a review model; 429-prone as driver", note="Kanban decomposition bake-off winner (2026-06-12)."),
    dict(display="Gemini 2.5 Flash-Lite", family="google", router="vertex/gemini-2.5-flash-lite", aa=None, intel_est=28, cin=0.10, cout=0.40, ctx=1_048_576, maxout=65_536, vision=True, tool="gemini", prc=False, trains=False, avail=True, authority="chat",
         strong="cheapest aux/vision tier", weak="not for reasoning or review"),
    dict(display="Gemini 3.5 Flash", family="google", router="vertex/gemini-3.5-flash", aa=55, cin=1.5, cout=9, ctx=1_048_576, maxout=65_536, vision=True, tool="gemini", prc=False, trains=False, avail=True, authority="chat", cost_estimated=True,
         strong="STANDOUT clean value — verified AA 55 (near GPT-5.5/Opus tier), fast, multimodal, 1M ctx, direct Vertex", weak="pricing est. pending Google list price", note="Direct Vertex (fluxlabs-499103, global endpoint). Best clean cost-effective pick from the 2026-06-17 sweep."),
    dict(display="Gemini 3.1 Pro", family="google", router="vertex/gemini-3.1-pro-preview", aa=57, cin=2, cout=12, ctx=1_048_576, maxout=65_536, vision=True, tool="gemini", prc=False, trains=False, avail=True, authority="chat", cost_estimated=True,
         strong="Google's current 2026 flagship reasoning model; long-context multimodal; fast Vertex path", weak="reasoning-heavy (modest output tps)", note="Direct Vertex (fluxlabs-499103, global endpoint). AA Intelligence Index ~57 (snapshot). Successor to Gemini 3 Pro."),
    dict(display="Gemini 3.1 Flash-Lite", family="google", router="vertex/gemini-3.1-flash-lite", aa=None, intel_est=33, cin=0.25, cout=1.5, ctx=1_048_576, maxout=65_536, vision=True, tool="gemini", prc=False, trains=False, avail=True, authority="chat", cost_estimated=True,
         strong="low-latency lightweight multimodal", weak="reasoning-heavy work", note="Direct Vertex (fluxlabs-499103, global endpoint). Pricing est."),
    dict(display="GLM 5.2", family="zai-glm", router="glm/glm-5.2", aa=51, cin=1.4, cout=4.4, ctx=1_048_576, maxout=131_072, vision=False, tool="glm", prc=True, trains=False, avail=True, authority="binding",
         strong="independent review of Claude code, agentic coding (max effort), 1M ctx", weak="vision (text-only), self-review of GLM code, binding PRD review", cost_estimated=True, note="Review-independence policy-preferred primary on z.ai quota reset (2026-06-16 21:24 UTC). List price not published — est. from GLM 5.1."),
    dict(display="GLM 4.7", family="zai-glm", router="glm/glm-4.7", aa=None, intel_est=48, cin=0.6, cout=2.2, ctx=200_000, maxout=128_000, vision=False, tool="glm", prc=True, trains=False, avail=True, authority="chat",
         strong="GLM Coding Plan default (Opus/Sonnet tier), review-lane fallback id", weak="vision (text-only), high-judgment binding review", note="Context window not published — est. 200K (glm-5-turbo alias)."),
    dict(display="GLM 4.5-Air", family="zai-glm", router=None, aa=None, intel_est=38, cin=0.2, cout=1.1, ctx=128_000, maxout=None, vision=False, tool="glm", prc=True, trains=False, avail=True, authority="chat",
         strong="small/fast cheap passes (Haiku-tier default)", weak="vision, high-judgment binding review", note="Context window not published — est. 128K. Not 9router-wired."),
    dict(display="GLM 4.6V", family="zai-glm", router="glm/glm-4.6v", aa=None, intel_est=45, cin=0.3, cout=0.9, ctx=131_072, maxout=None, vision=True, tool="glm", prc=True, trains=False, avail=True, authority="chat",
         strong="multimodal reasoning over image/video/file + text", weak="binding review verdicts (advisory only)"),
    dict(display="GLM 5", family="zai-glm", router="glm/glm-5", aa=None, intel_est=50, cin=1, cout=3.2, ctx=204_800, maxout=128_000, vision=False, tool="glm", prc=True, trains=False, avail=True, authority="chat",
         strong="agentic coding + long-horizon agent workflows (text)", weak="vision (text-only); binding review (advisory only)"),
    dict(display="GLM 5.1", family="zai-glm", router="glm/glm-5.1", aa=None, intel_est=51, cin=1.4, cout=4.4, ctx=204_800, maxout=128_000, vision=False, tool="glm", prc=True, trains=False, avail=True, authority="chat",
         strong="long-horizon agentic coding; tool use; general chat (text)", weak="vision (text-only); binding review (advisory only)"),
    dict(display="GLM 5.3", family="zai-glm", router="glm/glm-5.3", aa=None, intel_est=53, cin=1.4, cout=4.4, ctx=204_800, maxout=128_000, vision=False, tool="glm", prc=True, trains=False, avail=True, authority="chat", cost_estimated=True,
         strong="latest GLM agentic coder (text); config.yaml fallback-1", weak="vision (text-only); not in vault matrices yet", note="Live on 9router; not yet in vault matrices — all values ESTIMATED from GLM 5.1."),
    dict(display="Kimi K2.7 Code", family="moonshot-kimi", router="kimi/kimi-k2.7-code", aa=None, intel_est=56, cin=0.95, cout=4, ctx=262_144, maxout=32_768, vision=True, tool="native", prc=True, trains=True, avail=True, authority="chat",
         strong="agentic-coding review, token-efficient long-context 2nd opinion", weak="binding code verdict; confidential/student data (trains on inputs)", note="Robin review 2nd-opinion lane; never solo-binding on code."),
    dict(display="Kimi K2.7 Code HighSpeed", family="moonshot-kimi", router=None, aa=None, intel_est=56, cin=1.9, cout=8, ctx=262_144, maxout=32_768, vision=True, tool="native", prc=True, trains=True, avail=True, authority="chat",
         strong="latency-sensitive quick 2nd-opinion passes (~180-260 tok/s)", weak="cost-bulk (2x rate); binding verdict"),
    dict(display="Kimi K2.6", family="moonshot-kimi", router=None, aa=54, cin=0.95, cout=4, ctx=262_144, maxout=None, vision=True, tool="native", prc=True, trains=True, avail=True, authority="chat",
         strong="prior-gen review 2nd opinion", weak="superseded by K2.7 Code for the fleet lane"),
    dict(display="Kimi K2.7", family="moonshot-kimi", router="kimi/kimi-k2.7", aa=None, intel_est=54, cin=0.95, cout=4, ctx=262_144, maxout=None, vision=False, tool="tooluse", prc=True, trains=True, avail=True, authority="chat", cost_estimated=True,
         strong="advisory general chat / reasoning over text", weak="binding review; confidential/student data (trains on inputs)", note="Base K2.7 (Moonshot publishes Code only); ctx/cost est. from K2.7 Code."),
    dict(display="MiniMax-M3", family="minimax", router="minimax/MiniMax-M3", aa=55, cin=0.30, cout=1.20, ctx=1_000_000, maxout=None, vision=True, tool="native", prc=True, trains=False, avail=True, authority="binding",
         strong="binding independent review of Claude code, native multimodal (img+video), 1M ctx, extended thinking", weak="NOT PRD binding, NOT student/confidential data, NOT self-review", note="Claude-authored review BINDING PRIMARY (lane minimax). Carve-out MR-6."),
    dict(display="MiniMax M2.7", family="minimax", router="nvidia/minimaxai/minimax-m2.7", aa=None, intel_est=50, cin=0.30, cout=1.20, ctx=204_800, maxout=131_072, vision=False, tool="native", prc=True, trains=False, avail=True, authority="chat", cost_estimated=True,
         strong="agentic coding + tool use (NIM overflow, advisory)", weak="NIM fetch-timeout history (unreliable); binding verdicts; sensitive data", note="NIM-hosted copy, NOT api.minimax.io."),
    dict(display="Claude Opus 4.5", family="anthropic", router="cc/claude-opus-4-5-20251101", aa=None, intel_est=56, cin=5, cout=25, ctx=200_000, maxout=64_000, vision=True, tool="full", prc=False, trains=False, avail=True, authority="chat",
         strong="prior-gen agentic coding/reasoning", weak=">200K context (use 4.6/4.7/4.8 at 1M); advisory only"),
    dict(display="Claude Opus 4.6", family="anthropic", router="cc/claude-opus-4-6", aa=None, intel_est=58, cin=5, cout=25, ctx=1_000_000, maxout=128_000, vision=True, tool="full", prc=False, trains=False, avail=True, authority="chat",
         strong="prior-gen, 1M long-document work", weak="cost-sensitive bulk; advisory only"),
    dict(display="Claude Opus 4.7", family="anthropic", router="cc/claude-opus-4-7", aa=None, intel_est=59, cin=5, cout=25, ctx=1_000_000, maxout=128_000, vision=True, tool="full", prc=False, trains=False, avail=True, authority="chat",
         strong="prior-gen, 1M long-horizon work", weak="cost-sensitive bulk; advisory only; new tokenizer (~35% more tokens)"),
    dict(display="Claude Sonnet 4.5", family="anthropic", router="cc/claude-sonnet-4-5-20250929", aa=None, intel_est=50, cin=3, cout=15, ctx=200_000, maxout=64_000, vision=True, tool="full", prc=False, trains=False, avail=True, authority="chat",
         strong="balanced prior-gen general/coding", weak=">200K context (use Sonnet 4.6 at 1M); advisory only"),
    dict(display="DeepSeek V4 Pro", family="deepseek", router="openrouter/deepseek/deepseek-v4-pro", aa=None, intel_est=58, cin=0.435, cout=0.87, ctx=1_048_576, maxout=None, vision=False, tool="native", prc=True, trains=False, avail=True, authority="chat",
         strong="heavy reasoning/coding, full-codebase analysis (1M, hybrid attn)", weak="latency/cost-sensitive bulk (use Flash); advisory only"),
    dict(display="DeepSeek V4 Flash", family="deepseek", router="openrouter/deepseek/deepseek-v4-flash", aa=None, intel_est=46, cin=0.098, cout=0.196, ctx=1_048_576, maxout=None, vision=False, tool="native", prc=True, trains=False, avail=True, authority="chat",
         strong="fast/high-throughput coding + chat + agent (1M MoE)", weak="heaviest reasoning (use Pro); fabrication risk noted; advisory only"),
    dict(display="Mistral Medium 3.5", family="mistral", router="openrouter/mistralai/mistral-medium-3-5", aa=None, intel_est=47, cin=1.5, cout=7.5, ctx=262_144, maxout=None, vision=True, tool="strong", prc=False, trains=False, avail=True, authority="chat",
         strong="agentic multi-tool calling, vision, EU provider", weak="1M-context bulk (262K window); advisory only"),
    dict(display="Qwen3.7 Plus", family="qwen", router="openrouter/qwen/qwen3.7-plus", aa=None, intel_est=48, cin=0.32, cout=1.28, ctx=1_000_000, maxout=65_536, vision=True, tool="native", prc=True, trains=False, avail=True, authority="chat",
         strong="cost-effective coding/tool use, multimodal GUI/screen agent (1M)", weak="advisory only"),
    dict(display="MiMo V2.5", family="xiaomi", router="openrouter/xiaomi/mimo-v2.5", aa=None, intel_est=44, cin=0.14, cout=0.28, ctx=1_048_576, maxout=None, vision=True, tool="native", prc=True, trains=False, avail=True, authority="chat",
         strong="cost-efficient omnimodal agentic (Pro-level at ~half cost)", weak="advisory only"),
    # ── Clean cost-effective additions 2026-06-17 (fill PRC gaps; all non-PRC,
    # no train-on-input). Pricing from OpenRouter live catalog; ids probed 200.
    # Where `intel_est` remains below, it is a PRE-VERIFICATION estimate; verified
    # Artificial Analysis Intelligence Index scores override it via
    # build_benchmark.VERIFIED_AA (2026-06-17). ──
    dict(display="Mistral Large 3", family="mistral", router="openrouter/mistralai/mistral-large-2512", aa=23, cin=0.5, cout=1.5, ctx=262_144, maxout=None, vision=True, tool="strong", prc=False, trains=False, avail=True, authority="chat",
         strong="cheap EU model, 262K ctx, multi-tool calling", weak="WEAK on verified intelligence (AA 23) — surpassed by the newer Mistral Medium 3.5; not a reviewer-grade model"),
    dict(display="Mistral Small 4", family="mistral", router="openrouter/mistralai/mistral-small-2603", aa=None, intel_est=44, cin=0.15, cout=0.6, ctx=262_144, maxout=None, vision=True, tool="strong", prc=False, trains=False, avail=True, authority="chat",
         strong="cheap clean general/aux (EU), 262K ctx", weak="small model"),
    dict(display="Mistral Codestral 2508", family="mistral", router="openrouter/mistralai/codestral-2508", aa=None, intel_est=48, cin=0.3, cout=0.9, ctx=262_144, maxout=None, vision=False, tool="strong", prc=False, trains=False, avail=True, authority="chat",
         strong="cheap coding specialist (EU)", weak="coding-focused, not general"),
    dict(display="Mistral Devstral 2", family="mistral", router="openrouter/mistralai/devstral-2512", aa=None, intel_est=50, cin=0.4, cout=2.0, ctx=262_144, maxout=None, vision=False, tool="native", prc=False, trains=False, avail=True, authority="chat",
         strong="agentic coding / SWE (EU)", weak="coding-focused"),
    dict(display="Nemotron 3 Ultra", family="nvidia", router="openrouter/nvidia/nemotron-3-ultra-550b-a55b", aa=None, intel_est=56, cin=0.5, cout=2.2, ctx=1_048_576, maxout=None, vision=False, tool="native", prc=False, trains=False, avail=True, authority="chat",
         strong="frontier-reasoning + orchestration (US, NVIDIA); 1M ctx; strong value", weak="newer, text-only"),
    dict(display="Nemotron 3 Super", family="nvidia", router="openrouter/nvidia/nemotron-3-super-120b-a12b", aa=None, intel_est=48, cin=0.09, cout=0.45, ctx=1_048_576, maxout=None, vision=False, tool="native", prc=False, trains=False, avail=True, authority="chat",
         strong="very cheap clean coder (US); 1M ctx", weak="mid capability"),
    dict(display="Grok 4.3", family="xai", router="openrouter/x-ai/grok-4.3", aa=53, cin=1.25, cout=2.5, ctx=1_000_000, maxout=None, vision=True, tool="native", prc=False, trains=False, avail=True, authority="chat",
         strong="fast + clean (US/xAI), 1M ctx, good tool use", weak="mid capability (AA 53 — below GPT-5.5/Opus); less community tooling"),
    dict(display="Grok Build 0.1", family="xai", router="openrouter/x-ai/grok-build-0.1", aa=None, intel_est=50, cin=1.0, cout=2.0, ctx=256_000, maxout=None, vision=False, tool="native", prc=False, trains=False, avail=True, authority="chat",
         strong="fast coding/agentic (US/xAI)", weak="preview"),
    dict(display="Amazon Nova Pro", family="amazon", router="openrouter/amazon/nova-pro-v1", aa=14, cin=0.8, cout=3.2, ctx=300_000, maxout=None, vision=True, tool="strong", prc=False, trains=False, avail=True, authority="chat",
         strong="cheap US multimodal (AWS); fast", weak="WEAK on verified intelligence (AA 14) — only for trivial/aux work, not capability lanes"),
    dict(display="Amazon Nova Lite", family="amazon", router="openrouter/amazon/nova-lite-v1", aa=None, intel_est=36, cin=0.06, cout=0.24, ctx=300_000, maxout=None, vision=True, tool="tooluse", prc=False, trains=False, avail=True, authority="chat",
         strong="ultra-cheap US aux, multimodal", weak="low capability"),
    dict(display="Amazon Nova Micro", family="amazon", router="openrouter/amazon/nova-micro-v1", aa=None, intel_est=30, cin=0.035, cout=0.14, ctx=128_000, maxout=None, vision=False, tool="tooluse", prc=False, trains=False, avail=True, authority="chat",
         strong="cheapest clean aux (titles/triage/approval)", weak="very low capability; text-only"),
    dict(display="Cohere Command A", family="cohere", router="openrouter/cohere/command-a", aa=37, cin=2.5, cout=10, ctx=256_000, maxout=None, vision=False, tool="strong", prc=False, trains=False, avail=True, authority="chat",
         strong="agentic + multilingual + RAG (Canada, 111B); clean jurisdiction", weak="mid intelligence (AA 37 — ~Haiku 4.5 tier) and pricey ($$$) — not the reviewer-grade I'd earlier implied"),
    dict(display="Cohere Command R", family="cohere", router="openrouter/cohere/command-r-08-2024", aa=None, intel_est=38, cin=0.15, cout=0.6, ctx=128_000, maxout=None, vision=False, tool="tooluse", prc=False, trains=False, avail=True, authority="chat",
         strong="cheap clean general/RAG (Canada)", weak="older, mid"),
    dict(display="Gemma 4 26B", family="google", router="openrouter/google/gemma-4-26b-a4b-it", aa=None, intel_est=38, cin=0.1, cout=0.4, ctx=131_072, maxout=None, vision=True, tool="gemini", prc=False, trains=False, avail=True, authority="chat", cost_estimated=True,
         strong="open Google model (clean), cheap", weak="lower tier; pricing est."),
    # ── GPT-OSS + Antigravity (ag/) subscription models, added 2026-06-17 ──
    dict(display="GPT-OSS-120B", family="openai", router="openrouter/openai/gpt-oss-120b", aa=33, cin=0.05, cout=0.25, ctx=131_072, maxout=None, vision=False, tool="gpt", prc=False, trains=False, avail=True, authority="chat", cost_estimated=True,
         strong="open-weights reasoning/agentic (US, OpenAI); cheap", weak="mid capability (AA 33)"),
    # ── Antigravity (Google AI Pro subscription) — operator's PREFERRED channel,
    # esp. for Gemini (added 2026-06-17). The earlier $0-inflation removal reason is
    # MOOT now that cost is weighted 0. Caps: 100 Pro / 300 Thinking / 20 Deep
    # Research per day; 5-hour rolling window; 1,000 monthly credits; COMPUTE-BASED
    # (big-context/codebase prompts burn capacity fast — see run 447). Backups when
    # exhausted/down: Vertex → OpenRouter (Claude → Claude Code; GPT-OSS → OpenRouter).
    # ROUTER IDS verified live from 9Router /v1/models 2026-06-17. Only these tiers are
    # ROUTABLE on this Antigravity account (project valued-portfolio-wqlhg):
    #   * Gemini 3.5 Flash → ONLY -low / -extra-low. The desktop app's "High" and
    #     "Medium" Flash tiers return 404 "Requested entity was not found" from Google's
    #     Antigravity API (live-probed 2026-06-17: -high, -medium, and the base no-suffix
    #     id all 404; 9Router modelLock'd them). They are NOT a 9Router gap — the backend
    #     has no such id for this account. For higher-effort Flash use VERTEX
    #     (vertex/gemini-3.5-flash + a high thinkingLevel via the proxy).
    #   * Gemini 3.1 Pro → ag/gemini-pro-agent (Antigravity's "agent" variant; effort
    #     tier unconfirmed) + ag/gemini-3.1-pro-low.
    # speed_est is an estimate (agent endpoints don't return clean token-throughput).
    # AA scores for scoring now come from aa_baseline.py (the roster `aa` here is legacy);
    # each AG model inherits its base model's AA (channel-independent).
    dict(display="Gemini 3.1 Pro Agent (AG)", family="google", router="ag/gemini-pro-agent", aa=57, cin=0, cout=0, ctx=1_048_576, maxout=65_536, vision=True, tool="gemini", prc=False, trains=False, avail=True, authority="chat", speed_est=14,
         strong="Gemini 3.1 Pro 'agent' variant via Antigravity ($0 marginal); preferred Gemini channel", weak="agent-harness overhead → slow e2e; daily caps → fallback vertex/gemini-3.1-pro-preview → OpenRouter",
         note="ag/gemini-pro-agent — Antigravity's 'agent' Pro variant (effort tier UNCONFIRMED; not a verified 'high'). Same underlying model as Vertex 3.1 Pro."),
    dict(display="Gemini 3.1 Pro Low (AG)", family="google", router="ag/gemini-3.1-pro-low", aa=54, cin=0, cout=0, ctx=1_048_576, maxout=65_536, vision=True, tool="gemini", prc=False, trains=False, avail=True, authority="chat", speed_est=32,
         strong="Gemini 3.1 Pro, low reasoning, via Antigravity ($0 marginal); faster than the Agent variant", weak="lower reasoning ceiling; caps → fallback Vertex → OpenRouter",
         note="ag/gemini-3.1-pro-low."),
    dict(display="Gemini 3.5 Flash Low (AG)", family="google", router="ag/gemini-3.5-flash-low", aa=52, cin=0, cout=0, ctx=1_048_576, maxout=65_536, vision=True, tool="gemini", prc=False, trains=False, avail=True, authority="chat", speed_est=72,
         strong="Gemini 3.5 Flash (low reasoning) via Antigravity ($0 marginal); fast multimodal", weak="caps → fallback vertex/gemini-3.5-flash",
         note="ag/gemini-3.5-flash-low."),
    dict(display="Gemini 3.5 Flash X-Low (AG)", family="google", router="ag/gemini-3.5-flash-extra-low", aa=47, cin=0, cout=0, ctx=1_048_576, maxout=65_536, vision=True, tool="gemini", prc=False, trains=False, avail=True, authority="chat", speed_est=88,
         strong="Gemini 3.5 Flash (extra-low reasoning) via Antigravity ($0 marginal); fastest Flash variant", weak="minimal reasoning; caps → fallback vertex/gemini-3.5-flash",
         note="ag/gemini-3.5-flash-extra-low."),
    dict(display="Gemini 3 Flash (AG)", family="google", router="ag/gemini-3-flash", aa=None, intel_est=46, cin=0, cout=0, ctx=1_048_576, maxout=None, vision=True, tool="gemini", prc=False, trains=False, avail=True, authority="chat", speed_est=60,
         strong="Gemini 3 Flash via Antigravity ($0 marginal); fast multimodal", weak="caps → fallback vertex/gemini-3.5-flash → vertex/gemini-2.5-flash",
         note="ag/gemini-3-flash."),
    dict(display="Gemini 3 Flash Agent (AG)", family="google", router="ag/gemini-3-flash-agent", aa=None, intel_est=46, cin=0, cout=0, ctx=1_048_576, maxout=None, vision=True, tool="gemini", prc=False, trains=False, avail=True, authority="chat", speed_est=28,
         strong="Gemini 3 Flash agentic harness via Antigravity ($0 marginal)", weak="agent orchestration overhead → slower e2e; caps → fallback Vertex Flash",
         note="ag/gemini-3-flash-agent (agent-loop variant of 3 Flash)."),
    dict(display="Claude Opus 4.6 Thinking (AG)", family="anthropic", router="ag/claude-opus-4-6-thinking", aa=47, cin=0, cout=0, ctx=200_000, maxout=64_000, vision=True, tool="strong", prc=False, trains=False, avail=True, authority="chat", speed_est=40,
         strong="Claude Opus 4.6 (extended thinking) via Antigravity ($0 marginal)", weak="caps → fallback Claude Code cc/claude-opus-4-6",
         note="ag/claude-opus-4-6-thinking. Same model as cc/claude-opus-4-6 (AA 47 verified)."),
    dict(display="Claude Sonnet 4.6 (AG)", family="anthropic", router="ag/claude-sonnet-4-6", aa=43, cin=0, cout=0, ctx=200_000, maxout=64_000, vision=True, tool="strong", prc=False, trains=False, avail=True, authority="chat", speed_est=50,
         strong="Claude Sonnet 4.6 (thinking) via Antigravity ($0 marginal); fast capable", weak="caps → fallback Claude Code Sonnet",
         note="ag/claude-sonnet-4-6. AA 43 = same model as base cc/claude-sonnet-4-6 (channel-independent); estimate anchored to verified Claude ladder (< Opus 4.6=47)."),
    dict(display="GPT-OSS 120B Med (AG)", family="openai", router="ag/gpt-oss-120b-medium", aa=33, cin=0, cout=0, ctx=131_072, maxout=None, vision=False, tool="gpt", prc=False, trains=False, avail=True, authority="chat", speed_est=50,
         strong="GPT-OSS 120B (medium reasoning) via Antigravity ($0 marginal)", weak="mid capability (AA 33); caps → fallback OpenRouter gpt-oss-120b",
         note="ag/gpt-oss-120b-medium. Same open-weights model as the OpenRouter GPT-OSS-120B entry."),
    dict(display="Llama Guard 4 12B", family="meta", router="openrouter/meta-llama/llama-guard-4-12b", aa=None, intel_est=0, cin=0.18, cout=0.18, ctx=163_840, maxout=None, vision=True, tool="none", prc=False, trains=False, avail=True, authority="none",
         strong="content-safety moderation classifier (MLCommons S1-S14)", weak="NOT a chat model — emits safe/unsafe labels"),
    dict(display="Parakeet CTC 1.1B ASR", family="nvidia", router="nvidia/parakeet-ctc-1.1b-asr", aa=None, intel_est=0, cin=None, cout=None, ctx=0, maxout=None, vision=False, tool="none", prc=False, trains=False, avail=True, authority="none",
         strong="English speech-to-text transcription", weak="NOT a chat/text LLM; non-English audio", note="ASR utility; output_authority none.", cost_estimated=True),
    dict(display="Fusion", family="openrouter", router="openrouter/openrouter/fusion:general-high", aa=None, intel_est=60, cin=None, cout=None, ctx=200_000, maxout=None, vision=False, tool="native", prc=False, trains=False, avail=True, authority="chat",
         strong="multi-model deliberation / second-opinion synthesis (panel + judge)", weak="priced as sum of all completions; 500-prone; requires stream:true; latency-heavy", note="Meta-model — not a single-call lane primary. Cost varies per call.", cost_estimated=True),
]
# fmt: on
