"""Per-turn conductor-vault grounding preflight.

At the start of each substantive user turn, run the conductor-vault
"grounding preflight" CLI and inject its result into the model's view of the
current turn — so the agent grounds itself in the vault + host skills BEFORE
asking the user for facts or answering generically about named systems.

This mirrors the existing per-turn memory-context injection
(``agent.memory_manager.build_memory_context_block``) but is its own,
distinctly-tagged ``<grounding-context>`` block.

The blessed cross-repo integration is a CLI subprocess: the
``conductor_vault`` package is not importable from this repo, so we shell out
to the ``conductor-vault`` binary (fixed argv, no shell). A preflight failure
must NEVER raise into the turn — the entire function body is guarded and any
exception returns ``""`` (skip injection).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_VAULT_ROOT = "/srv/fluxlabs/vault/conductor-vault"

# Minimum word count for a prompt to be worth grounding. Low-signal
# continuations / acks ("yes", "ok thanks") are skipped without ever paying
# for the subprocess.
_MIN_WORDS = 3

# Hard ceiling on the CLI call so a slow vault never stalls a turn.
_TIMEOUT_SECONDS = 8


def _vault_root() -> str:
    """Resolve the conductor-vault root (env override or fleet default)."""
    return os.environ.get("CONDUCTOR_VAULT_ROOT") or _DEFAULT_VAULT_ROOT


def _resolve_cli() -> Optional[str]:
    """Locate the ``conductor-vault`` CLI binary, or ``None`` if absent."""
    found = shutil.which("conductor-vault")
    if found:
        return found
    fallback = os.path.expanduser("~/.local/bin/conductor-vault")
    if os.path.exists(fallback):
        return fallback
    return None


def build_grounding_context_block(
    user_message: Any,
    conversation_history: Optional[List[Any]] = None,
) -> str:
    """Run the grounding preflight and return a fenced block (or ``""``).

    Returns ``""`` (skip injection) when:
      * ``user_message`` is not a non-empty str, or has < 3 words
        (low-signal continuation/ack) — NOT skipped merely because it is the
        first turn;
      * the CLI cannot be resolved;
      * the CLI fails, times out, returns no output, or returns JSON with no
        project/skills/decision context to ground against;
      * ANY exception occurs (guarded — a preflight failure never raises).
    """
    try:
        # Skip low-signal prompts before paying for the subprocess.
        if not isinstance(user_message, str) or not user_message.strip():
            return ""
        if len(user_message.split()) < _MIN_WORDS:
            return ""

        cli = _resolve_cli()
        if cli is None:
            return ""

        proc = subprocess.run(  # noqa: S603 -- fixed argv, no shell
            [cli, "project-memory", "preflight", user_message, "--format", "json"],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=False,
            cwd=_vault_root(),
        )
        if proc.returncode != 0 or not proc.stdout or not proc.stdout.strip():
            return ""

        data = json.loads(proc.stdout)
        if not isinstance(data, dict):
            return ""

        # Gate: only inject when there is actual grounding to offer. An empty
        # preflight (no project/skills/decision hits) is not worth a block.
        has_grounding = any(
            bool(data.get(key))
            for key in ("project_context", "skills_context", "decision_context")
        )
        if not has_grounding:
            return ""

        compact = data.get("compact_context")
        if not isinstance(compact, str) or not compact.strip():
            return ""

        return (
            "<grounding-context>\n"
            "[System note: The following is grounding context retrieved from "
            "the conductor vault and host skills for systems/projects named in "
            "this turn. It is authoritative reference data, NOT new user input. "
            "Consult it BEFORE asking the user for facts or answering "
            "generically about these systems.]\n\n"
            f"{compact}\n"
            "</grounding-context>"
        )
    except Exception:
        logger.debug("grounding preflight failed; skipping injection", exc_info=True)
        return ""
