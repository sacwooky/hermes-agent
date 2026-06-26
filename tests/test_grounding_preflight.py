"""Hermetic tests for ``agent.grounding_preflight``.

The grounding preflight shells out to the ``conductor-vault`` CLI to fetch
vault + host-skill grounding for the current turn. These tests monkeypatch
``subprocess.run`` (and ``_resolve_cli``) so the real CLI is NEVER invoked —
they pin the skip / gate / failure-isolation contract:

* low-signal prompts are skipped before the subprocess is ever called;
* a successful preflight with grounding yields a ``<grounding-context>`` block;
* an empty (no project/skills/decision) preflight is gated out;
* any CLI failure (nonzero exit, timeout) returns ``""`` without raising;
* a missing CLI returns ``""``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

# Ensure the repo root is importable when running via `pytest tests/...`.
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent import grounding_preflight as gp


def _completed(returncode: int, stdout: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["conductor-vault"], returncode=returncode, stdout=stdout, stderr=""
    )


def _force_cli(monkeypatch) -> None:
    """Pretend the CLI is present so resolution never short-circuits."""
    monkeypatch.setattr(gp, "_resolve_cli", lambda: "/fake/bin/conductor-vault")


def test_low_signal_prompt_skipped_without_subprocess(monkeypatch):
    """A < 3-word prompt is skipped; the subprocess is never invoked."""
    called = {"n": 0}

    def _boom(*args, **kwargs):  # pragma: no cover - must not run
        called["n"] += 1
        raise AssertionError("subprocess.run should not be called for low-signal")

    monkeypatch.setattr(gp.subprocess, "run", _boom)
    _force_cli(monkeypatch)

    assert gp.build_grounding_context_block("yes") == ""
    assert called["n"] == 0


def test_success_returns_grounding_block(monkeypatch):
    """A successful preflight with grounding yields a tagged block."""
    compact = "## Project Foo\nKey facts about Foo from the vault."
    payload = {
        "status": "ok",
        "compact_context": compact,
        "project_context": [],
        "decision_context": [],
        "skills_context": ["skill-a"],
        "source_paths": ["wiki/projects/foo.md"],
    }
    monkeypatch.setattr(
        gp.subprocess, "run", lambda *a, **k: _completed(0, json.dumps(payload))
    )
    _force_cli(monkeypatch)

    block = gp.build_grounding_context_block("tell me about project foo please")
    assert "<grounding-context>" in block
    assert "</grounding-context>" in block
    assert compact in block


def test_empty_grounding_is_gated_out(monkeypatch):
    """No project/skills/decision context -> skip even with compact text."""
    payload = {
        "status": "ok",
        "compact_context": "some text",
        "project_context": [],
        "decision_context": [],
        "skills_context": [],
        "source_paths": [],
    }
    monkeypatch.setattr(
        gp.subprocess, "run", lambda *a, **k: _completed(0, json.dumps(payload))
    )
    _force_cli(monkeypatch)

    assert gp.build_grounding_context_block("tell me about project foo please") == ""


def test_nonzero_returncode_returns_empty(monkeypatch):
    """A CLI failure (nonzero exit) is swallowed -> ''."""
    monkeypatch.setattr(
        gp.subprocess, "run", lambda *a, **k: _completed(1, "")
    )
    _force_cli(monkeypatch)

    assert gp.build_grounding_context_block("tell me about project foo please") == ""


def test_timeout_returns_empty_without_raising(monkeypatch):
    """A subprocess timeout never raises into the turn -> ''."""

    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="conductor-vault", timeout=8)

    monkeypatch.setattr(gp.subprocess, "run", _timeout)
    _force_cli(monkeypatch)

    assert gp.build_grounding_context_block("tell me about project foo please") == ""


def test_missing_cli_returns_empty(monkeypatch):
    """When the CLI cannot be resolved, return '' without subprocessing."""

    def _boom(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("subprocess.run should not be called with no CLI")

    monkeypatch.setattr(gp.subprocess, "run", _boom)
    monkeypatch.setattr(gp, "_resolve_cli", lambda: None)

    assert gp.build_grounding_context_block("tell me about project foo please") == ""
