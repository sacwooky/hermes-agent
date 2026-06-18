"""Focused preflight test: credential-aware failover skips a circuit-open key.

Design: preflight-credential-failover-circuit-breaker §4B / §9.1-§9.2.

When the primary provider raises ``AuthError``, ``_ensure_runtime_credentials``
must walk the fallback chain and select the first candidate that is BOTH
resolvable AND not circuit-open by credential identity — a present-but-dead key
whose breaker is OPEN must be skipped, not selected.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

from hermes_cli.auth import AuthError


def _import_cli():
    for name in list(sys.modules):
        if name == "cli" or name == "run_agent" or name == "tools" or name.startswith("tools."):
            sys.modules.pop(name, None)
    if "firecrawl" not in sys.modules:
        sys.modules["firecrawl"] = types.SimpleNamespace(Firecrawl=object)
    try:
        importlib.import_module("prompt_toolkit")
    except ModuleNotFoundError:
        pytest.skip("prompt_toolkit not available")
    return importlib.import_module("cli")


@pytest.fixture(autouse=True)
def _restore_modules():
    prefixes = ("tools", "cli", "run_agent")
    original = {
        n: m for n, m in sys.modules.items()
        if any(n == p or n.startswith(p + ".") for p in prefixes)
    }
    try:
        yield
    finally:
        for n in list(sys.modules):
            if any(n == p or n.startswith(p + ".") for p in prefixes):
                sys.modules.pop(n, None)
        sys.modules.update(original)


def test_preflight_skips_circuit_open_credential(monkeypatch):
    cli = _import_cli()

    # Primary fails auth; first fallback's CREDENTIAL is circuit-open (must be
    # skipped); second fallback is healthy and must be selected.
    def _runtime_resolve(**kwargs):
        requested = (kwargs.get("requested") or "").strip().lower()
        if requested in ("", "primary-provider"):
            raise AuthError("primary key rejected")
        # Any fallback resolves fine (resolvability != validity — that's D1).
        return {
            "provider": requested,
            "api_mode": "chat_completions",
            "base_url": "https://endpoint.example/v1",
            "api_key": "resolved-key",
            "source": "env/config",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", _runtime_resolve
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.format_runtime_provider_error", lambda exc: str(exc)
    )

    # Open the breaker ONLY for the dead credential's cred_id.
    dead_entry = {"provider": "dead-fallback", "model": "m1",
                  "base_url": "https://dead.example/v1"}
    healthy_entry = {"provider": "healthy-fallback", "model": "m2",
                     "base_url": "https://healthy.example/v1"}

    from hermes_cli import provider_health as ph
    dead_cid = ph.cred_id(dead_entry)

    def _is_open(cid, **kwargs):
        return cid == dead_cid

    monkeypatch.setattr(ph, "is_open", _is_open)

    shell = cli.HermesCLI(model="m0", compact=True, max_turns=1)
    shell.requested_provider = "primary-provider"
    shell._fallback_model = [dead_entry, healthy_entry]

    assert shell._ensure_runtime_credentials() is True
    # The dead-credential fallback was skipped; the healthy one was selected.
    assert shell.requested_provider == "healthy-fallback"
    assert shell.model == "m2"
