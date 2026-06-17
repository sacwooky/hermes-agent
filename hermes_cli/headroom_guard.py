"""EPIC 4 — Headroom guardrails.

Enforces five rules:
1. No Headroom placeholder on any conformance-evidence field before it
   crosses to Robin.
2. Conformance evidence must be materialized (original bytes) before send.
3. compress/retrieve calls are traced (hash + lane + retrieved-before-decision).
4. ``headroom learn`` AGENTS.md/CLAUDE.md auto-write is OFF.
5. No model chain has base_url pointing at ``:8797``.

All checks are best-effort and safe to import even when the headroom
service is not running.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Placeholder detection
# ---------------------------------------------------------------------------

# Matches the headroom_compress MCP tool output: [headroom:<hex8+>]
PLACEHOLDER_RE = re.compile(r'\[headroom:[a-f0-9]{8,}\]')

# 64-char lowercase hex string — the raw hash that headroom_compress embeds.
_HEX64_RE = re.compile(r'^[a-f0-9]{64}$')


def is_headroom_placeholder(value: str) -> bool:
    """Return True if *value* looks like an unexpanded headroom placeholder.

    Conservative: only flags strings that match exactly the patterns
    produced by headroom_compress, not general content.
    """
    if not isinstance(value, str):
        return False
    # Pattern 1: bracket form [headroom:<hex>]
    if PLACEHOLDER_RE.search(value):
        return True
    # Pattern 2: literal marker string
    if "headroom_placeholder" in value:
        return True
    # Pattern 3: exactly a 64-char lowercase hex string (raw hash token)
    stripped = value.strip()
    if _HEX64_RE.match(stripped):
        return True
    return False


class HeadroomPlaceholderError(ValueError):
    """Raised when a conformance-evidence field contains an unexpanded
    Headroom placeholder that would cross to Robin un-materialized."""


def assert_no_placeholders(fields: dict, context: str = "") -> None:
    """Raise :class:`HeadroomPlaceholderError` if any field value is a
    placeholder.

    Only checks str values. Recursively checks nested dicts/lists one
    level deep. Logs a warning for each hit before raising.

    :param fields: mapping of field-name → value to inspect.
    :param context: caller label included in log messages and the exception.
    """
    hits: list[str] = []
    for key, value in fields.items():
        _check_value(key, value, hits, context)
    if hits:
        msg = f"headroom placeholder detected [{context}]: " + "; ".join(hits)
        for h in hits:
            _log.warning("headroom_guard: %s — %s", context, h)
        raise HeadroomPlaceholderError(msg)


def _check_value(key: str, value: Any, hits: list[str], context: str) -> None:
    """Recursively inspect *value* (one level deep for nested containers)."""
    if isinstance(value, str):
        if is_headroom_placeholder(value):
            hits.append(f"field={key!r} is a placeholder")
    elif isinstance(value, dict):
        for sub_key, sub_val in value.items():
            if isinstance(sub_val, str) and is_headroom_placeholder(sub_val):
                hits.append(f"field={key!r}.{sub_key!r} is a placeholder")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            if isinstance(item, str) and is_headroom_placeholder(item):
                hits.append(f"field={key!r}[{i}] is a placeholder")


# ---------------------------------------------------------------------------
# Compress / retrieve tracing
# ---------------------------------------------------------------------------

TRACE_LOG_PATH = os.path.expanduser("~/.hermes/logs/headroom-trace.jsonl")


def _write_trace(entry: dict) -> None:
    """Append *entry* to TRACE_LOG_PATH. Best-effort — never raises."""
    try:
        log_path = Path(TRACE_LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        _log.debug("headroom_guard: trace write failed", exc_info=True)


def trace_compress(hash_val: str, lane: str, action: str = "compress") -> None:
    """Append a compress trace entry to TRACE_LOG_PATH (append-only, best-effort)."""
    _write_trace({
        "event": "compress",
        "hash": hash_val,
        "lane": lane,
        "action": action,
        "retrieved": False,
        "ts": int(time.time()),
    })


def trace_retrieve(hash_val: str, lane: str, action: str = "retrieve") -> None:
    """Append a retrieve trace entry.

    Sets ``retrieved=True`` on the matching compress entry by writing a
    new retrieve event — the JSONL is append-only so we record the
    retrieve separately and consumers correlate by hash.
    """
    _write_trace({
        "event": "retrieve",
        "hash": hash_val,
        "lane": lane,
        "action": action,
        "retrieved": True,
        "ts": int(time.time()),
    })


# ---------------------------------------------------------------------------
# Startup checks (E4-S4)
# ---------------------------------------------------------------------------


def check_headroom_learn_off() -> list[str]:
    """Return a list of warning strings if ``headroom learn`` appears enabled.

    Checks ``~/.hermes/headroom/config.yaml`` (and a few fallback paths)
    for ``learn.enabled`` or ``learn.auto_write`` being truthy.

    Returns ``[]`` if the config is not found (assume OFF). Never raises.
    """
    candidate_paths = [
        Path(os.path.expanduser("~/.hermes/headroom/config.yaml")),
        Path(os.path.expanduser("~/.hermes/headroom/config.yml")),
        Path(os.path.expanduser("~/.hermes/config/headroom.yaml")),
    ]
    warnings: list[str] = []
    for cfg_path in candidate_paths:
        if not cfg_path.exists():
            continue
        try:
            text = cfg_path.read_text(encoding="utf-8")
        except OSError:
            continue
        # Simple line-level scan — avoids a yaml dependency and keeps this
        # module importable without extras.
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            # Skip comments
            if stripped.startswith("#"):
                continue
            # Flag any learn.enabled: true / learn.auto_write: true style lines
            if re.search(r'\blearn\b', stripped, re.IGNORECASE):
                if re.search(r'\b(enabled|auto_write)\s*:\s*(true|yes|1)\b', stripped, re.IGNORECASE):
                    warnings.append(
                        f"{cfg_path}:{lineno}: headroom learn appears enabled "
                        f"({stripped!r}) — E4-S4 requires learn OFF"
                    )
    return warnings


def check_no_8797_base_url() -> list[str]:
    """Scan hermes chain config files for ``base_url`` containing ``:8797``.

    Paths checked:
      * ``~/.hermes/profiles/*/config.yaml``
      * ``~/.hermes/hermes-agent/config/*.yaml``

    Returns a list of ``(path:line)`` warning strings for hits. Never raises.
    """
    warnings: list[str] = []
    search_globs = [
        Path(os.path.expanduser("~/.hermes/profiles")),
        Path(os.path.expanduser("~/.hermes/hermes-agent/config")),
    ]
    config_files: list[Path] = []
    for base in search_globs:
        if not base.exists():
            continue
        try:
            if base.name == "profiles":
                config_files.extend(base.glob("*/config.yaml"))
                config_files.extend(base.glob("*/config.yml"))
            else:
                config_files.extend(base.glob("*.yaml"))
                config_files.extend(base.glob("*.yml"))
        except OSError:
            continue
    for cfg_path in config_files:
        try:
            text = cfg_path.read_text(encoding="utf-8")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if ":8797" in line and "base_url" in line:
                warnings.append(
                    f"{cfg_path}:{lineno}: base_url contains :8797 "
                    f"— E4-S5 prohibits this chain endpoint ({line.strip()!r})"
                )
    return warnings
