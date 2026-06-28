"""Credential-keyed circuit breaker for auth/availability failover.

Implements the v2 design from
``conductor-vault/wiki/projects/preflight-credential-failover-circuit-breaker.md``
(§0a resolutions; supersedes the original §4A/§5 open-policy/cred_id model).

The breaker lets a transient credential blip route to a *different, healthy*
credential instead of committing to a present-but-dead key and aborting.

Core principle (§3):
    Circuit-break by credential identity, select by liveness, and never abort
    while a tier on a different, healthy credential remains.

Design highlights (§0a):
  * Bounded-failure + exponential backoff (#2): a single 401 does NOT open the
    breaker — only N=3 consecutive AUTH failures inside a 120s rolling window.
    State machine CLOSED → OPEN → HALF_OPEN → CLOSED. Cooldown escalates per
    consecutive open (60→120→240…cap 600s); HALF_OPEN allows exactly one trial.
  * Failure-class taxonomy (#3): auth (401/403-auth) → credential breaker;
    403 content_policy_blocked → request-scoped, never marks the credential;
    429/billing → separate shorter-backoff breaker (30s base, same key);
    5xx/connection → AVAILABILITY breaker (endpoint-scoped, see #8).
  * Healthy-clear + race (#4): success closes + zeroes; records carry
    monotonic-ms timestamps; a record supersedes another ONLY with a strictly
    newer timestamp — a slow worker's stale failure cannot re-open a credential
    a faster worker just proved healthy.
  * cred_id vs endpoint_id (#8): AUTH keys on credential fingerprint
    ``(key_env, sha256(resolved_key)[:8])``; AVAILABILITY keys on
    ``(base_url host:port)``. The key value is NEVER stored or logged — only
    the env-var name and an 8-char hash.
  * fleet-shared scope (#5): per-entry ``scope`` of host-local | fleet-shared.
    host-local lives in the local file; the only fleet-shared door is
    ``fleet:nvidia`` (keyless), whose HALF_OPEN trial is an active liveness
    probe supplied by the caller (probe-authoritative, no shared write path).
  * FAIL-OPEN everywhere: a missing/corrupt health file ⇒ behave as if all
    credentials are healthy — the breaker itself never blocks a spawn. Writes
    are multi-writer safe (flock + atomic os.replace).

State file (pinned, resolves the §0a#5 placeholder):
    ~/.hermes/state/provider-auth-health.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ── Tunables (defaults per §0a#2/#3) ────────────────────────────────────────

# AUTH breaker.
AUTH_FAILURE_THRESHOLD = 3          # N consecutive AUTH failures to open
AUTH_WINDOW_S = 120.0              # rolling window for the consecutive count
AUTH_COOLDOWN_BASE_S = 60.0        # first cooldown
AUTH_COOLDOWN_CAP_S = 600.0        # max cooldown after repeated opens

# RATE-LIMIT breaker (429/billing) — shorter backoff, same key (§0a#3).
RATE_FAILURE_THRESHOLD = 3
RATE_WINDOW_S = 120.0
RATE_COOLDOWN_BASE_S = 30.0
RATE_COOLDOWN_CAP_S = 300.0

# AVAILABILITY breaker (5xx/connection) — endpoint-scoped.
AVAIL_FAILURE_THRESHOLD = 3
AVAIL_WINDOW_S = 120.0
AVAIL_COOLDOWN_BASE_S = 30.0
AVAIL_COOLDOWN_CAP_S = 300.0

# Pinned state file (§0a#5 — replaces the original placeholder).
_DEFAULT_STATE_PATH = os.path.expanduser("~/.hermes/state/provider-auth-health.json")

# Optional best-effort fleet hint file (§0a#5) — a HINT only, never
# authoritative; absent/corrupt ⇒ fall back to local probing. Kept fail-open.
_DEFAULT_FLEET_HINT_PATH = os.path.expanduser(
    "~/.hermes/state/fleet-provider-health.json"
)

# State machine labels.
CLOSED = "CLOSED"
OPEN = "OPEN"
HALF_OPEN = "HALF_OPEN"

# Failure classes (§0a#3).
AUTH = "auth"
RATE_LIMIT = "rate_limit"
AVAILABILITY = "availability"
CONTENT_POLICY = "content_policy"  # request-scoped; never marks a credential

# Scope (§0a#5).
SCOPE_HOST_LOCAL = "host-local"
SCOPE_FLEET_SHARED = "fleet-shared"


# ── Time + identity helpers ─────────────────────────────────────────────────

def _now_ms() -> int:
    """Monotonic-ms timestamp (§0a#4 — records carry monotonic-ms stamps).

    ``time.monotonic_ns`` is used so two writers on the same host order
    correctly and a stale failure cannot supersede a newer success.
    """
    return time.monotonic_ns() // 1_000_000


def _config_custom_providers() -> list:
    """Return the host's ``custom_providers`` defs (fail-open to []).

    Used to derive ``key_env`` for a chain entry whose provider is a
    ``custom:<name>`` reference. Never raises.
    """
    try:
        from hermes_cli.config import get_compatible_custom_providers, load_config

        cfg = load_config()
        return list(get_compatible_custom_providers(cfg) or [])
    except Exception:  # pragma: no cover - fail open
        return []


def _normalize_provider_name(value: str) -> str:
    """Strip the ``custom:`` prefix and lowercase, matching runtime_provider."""
    name = (value or "").strip().lower()
    if name.startswith("custom:"):
        name = name[len("custom:"):].strip()
    return name


def _key_env_for_entry(entry: Dict[str, Any]) -> str:
    """Resolve the ``key_env`` (env-var NAME) for a chain entry.

    Precedence:
      1. an explicit ``key_env`` / ``api_key_env`` on the entry itself,
      2. the matching ``custom_providers`` def (by provider name),
      3. a small built-in provider→env map for non-custom providers.

    Returns "" when unknown (caller treats that as keyless/host-local).
    Never reads or returns a key VALUE.
    """
    explicit = str(entry.get("key_env") or entry.get("api_key_env") or "").strip()
    if explicit:
        return explicit

    provider = _normalize_provider_name(str(entry.get("provider") or ""))
    if provider:
        for cp in _config_custom_providers():
            if not isinstance(cp, dict):
                continue
            if _normalize_provider_name(str(cp.get("name") or "")) == provider:
                ke = str(cp.get("key_env") or cp.get("api_key_env") or "").strip()
                if ke:
                    return ke

    # Built-in provider → conventional env var (host-local creds).
    _BUILTIN_ENV = {
        "zai": "ZAI_API_KEY",
        "zai-keith": "ZAI_KEITH_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "ninerouter": "NINEROUTER_KEY",
        "9router-codex": "NINEROUTER_KEY",
    }
    return _BUILTIN_ENV.get(provider, "")


def _resolved_key_fingerprint(key_env: str) -> str:
    """Return ``sha256(os.getenv(key_env))[:8]`` or "" — NEVER the key value.

    Resolves the key only long enough to hash it; the cleartext is never
    stored, logged, or returned. Fail-open to "" so a missing env var still
    yields a stable (env-only) cred_id rather than raising.
    """
    if not key_env:
        return ""
    val = os.getenv(key_env, "") or ""
    if not val:
        return ""
    return hashlib.sha256(val.encode("utf-8", "surrogatepass")).hexdigest()[:8]


def cred_id(entry: Dict[str, Any]) -> str:
    """Credential fingerprint for AUTH breaking (§0a#8).

    ``(key_env, sha256(resolved_key)[:8])`` — a rejected KEY is dead on every
    endpoint that uses it, so gpt-5.5 and gemini-via-9router (both
    ``NINEROUTER_KEY``) correctly collapse to ONE breaker. Two entries with the
    same env but no resolvable value share the env-only id, which is the safe
    grouping (same configured credential).

    For a keyless fleet-shared door (e.g. ``fleet:nvidia``), there is no key to
    fingerprint; the id keys on the scope marker instead.

    NEVER includes the key value — only the env name + 8-char hash.
    """
    scope = str(entry.get("scope") or "").strip().lower()
    if scope == SCOPE_FLEET_SHARED:
        marker = str(entry.get("cred_id") or entry.get("fleet_cred_id") or "").strip()
        if marker:
            return marker
        # Default keyless fleet door identity.
        return "fleet:nvidia"

    key_env = _key_env_for_entry(entry)
    fp = _resolved_key_fingerprint(key_env)
    if key_env and fp:
        return f"{key_env}:{fp}"
    if key_env:
        # Configured but unresolved — still group by env name (the credential
        # the operator intends to use). Distinct from a resolved one.
        return f"{key_env}:unresolved"
    # No identifiable credential — fall back to endpoint so the breaker still
    # has a stable, distinct key rather than colliding everything to "".
    return f"anon:{endpoint_id(entry)}"


def endpoint_id(entry: Dict[str, Any]) -> str:
    """Availability key (§0a#8): ``host:port`` of the entry's base_url.

    A 5xx on one endpoint must NOT blacklist a healthy key used elsewhere, so
    availability keys on the endpoint, not the credential.
    """
    base_url = str(entry.get("base_url") or "").strip()
    if not base_url:
        return "unknown"
    try:
        parsed = urlparse(base_url)
        host = parsed.hostname or ""
        port = parsed.port
        if host and port:
            return f"{host}:{port}"
        if host:
            return host
    except Exception:  # pragma: no cover - fail open
        pass
    return base_url.rstrip("/").lower()


def scope_of(entry: Dict[str, Any]) -> str:
    """Per-entry scope (§0a#5): host-local (default) | fleet-shared."""
    scope = str(entry.get("scope") or "").strip().lower()
    if scope == SCOPE_FLEET_SHARED:
        return SCOPE_FLEET_SHARED
    return SCOPE_HOST_LOCAL


# ── Failure-class taxonomy (§0a#3) ──────────────────────────────────────────

def classify_failure(
    *,
    status_code: Optional[int] = None,
    reason: Optional[str] = None,
    is_auth: Optional[bool] = None,
) -> str:
    """Map a failure to a breaker class (§0a#3).

    Returns one of AUTH | RATE_LIMIT | AVAILABILITY | CONTENT_POLICY.

      * 403 content_policy_blocked → CONTENT_POLICY (request-scoped — caller
        must NOT mark the credential).
      * 401 / 403-auth (key rejected/expired) → AUTH.
      * 429 / billing-exhausted → RATE_LIMIT (separate shorter breaker).
      * 5xx / connection → AVAILABILITY (endpoint-scoped).

    Accepts a free-form ``reason`` string (e.g. the ``FailoverReason`` value)
    so the caller can pass the existing classifier's verdict directly.
    """
    r = (reason or "").strip().lower()
    if r in {"content_policy_blocked", "provider_policy_blocked"}:
        return CONTENT_POLICY
    if r in {"billing", "rate_limit"}:
        return RATE_LIMIT
    if r in {"auth", "auth_permanent"}:
        return AUTH
    if r in {"overloaded", "server_error", "timeout"}:
        return AVAILABILITY

    if is_auth:
        return AUTH

    if status_code is not None:
        if status_code in (401, 403):
            return AUTH
        if status_code == 429 or status_code == 402:
            return RATE_LIMIT
        if status_code >= 500:
            return AVAILABILITY

    # Connection-level errors arrive with no status — treat as availability.
    return AVAILABILITY


def _params_for_class(failure_class: str) -> Tuple[int, float, float, float]:
    """Return (threshold, window_s, cooldown_base_s, cooldown_cap_s)."""
    if failure_class == RATE_LIMIT:
        return (RATE_FAILURE_THRESHOLD, RATE_WINDOW_S, RATE_COOLDOWN_BASE_S, RATE_COOLDOWN_CAP_S)
    if failure_class == AVAILABILITY:
        return (AVAIL_FAILURE_THRESHOLD, AVAIL_WINDOW_S, AVAIL_COOLDOWN_BASE_S, AVAIL_COOLDOWN_CAP_S)
    return (AUTH_FAILURE_THRESHOLD, AUTH_WINDOW_S, AUTH_COOLDOWN_BASE_S, AUTH_COOLDOWN_CAP_S)


# ── Persistence (fail-open, multi-writer safe) ──────────────────────────────

def _empty_state() -> Dict[str, Any]:
    return {"version": 1, "records": {}}


def _load_state(path: str) -> Dict[str, Any]:
    """Load the health file. FAIL-OPEN: any error ⇒ empty (all healthy)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return _empty_state()
        records = data.get("records")
        if not isinstance(records, dict):
            data["records"] = {}
        return data
    except FileNotFoundError:
        return _empty_state()
    except Exception as exc:  # corrupt/unreadable ⇒ behave as all-healthy
        logger.debug("provider_health: load failed (%s) — failing open", exc)
        return _empty_state()


def _commit_record(path: str, key: str, rec: Dict[str, Any], now_ms: int) -> bool:
    """Lock-protected read-modify-write of a single record. Fail-open.

    Holds an exclusive ``flock`` across the whole RMW so concurrent workers
    serialize. Inside the lock it re-reads the on-disk state and applies the
    strict-newer-timestamp merge (§0a#4): if disk already holds a record with a
    STRICTLY-NEWER ``updated_ms`` than ours, we abandon our write (a stale
    worker cannot clobber a fresher record). Otherwise we write ``rec`` and
    atomically ``os.replace`` the file. Any failure ⇒ fail-open (no write).
    """
    import fcntl

    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    except Exception as exc:  # pragma: no cover - fail open
        logger.debug("provider_health: mkdir failed (%s)", exc)
        return False

    lock_path = path + ".lock"
    tmp_path = f"{path}.{os.getpid()}.tmp"
    try:
        with open(lock_path, "a+", encoding="utf-8") as lock_fh:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            except Exception:  # pragma: no cover - non-posix / nfs
                pass

            disk = _load_state(path)
            disk_rec = disk.get("records", {}).get(key)
            if isinstance(disk_rec, dict) and disk_rec.get("updated_ms", 0) > now_ms:
                # Disk has a strictly-newer record — do not clobber it (§0a#4).
                return False

            disk.setdefault("version", 1)
            disk.setdefault("records", {})
            disk["records"][key] = rec

            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(disk, fh, separators=(",", ":"))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, path)
        return True
    except Exception as exc:  # pragma: no cover - fail open
        logger.debug("provider_health: write failed (%s) — failing open", exc)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False


# ── State-machine record helpers ────────────────────────────────────────────

def _default_record() -> Dict[str, Any]:
    return {
        "state": CLOSED,
        "consecutive_failures": 0,
        "open_count": 0,          # number of consecutive opens (for backoff)
        "window_start_ms": 0,
        "opened_at_ms": 0,
        "cooldown_s": 0.0,
        "last_success_ms": 0,
        "last_failure_ms": 0,
        "updated_ms": 0,
    }


def _record_key(failure_class: str, ident: str) -> str:
    """Compose the storage key: class-namespaced so auth/rate/avail are split."""
    return f"{failure_class}::{ident}"


def _get_record(state: Dict[str, Any], key: str) -> Dict[str, Any]:
    rec = state.get("records", {}).get(key)
    if not isinstance(rec, dict):
        return _default_record()
    merged = _default_record()
    merged.update({k: rec.get(k, merged[k]) for k in merged})
    return merged


def _evaluate_state(rec: Dict[str, Any], failure_class: str, now_ms: int) -> str:
    """Return the *effective* state, transitioning OPEN→HALF_OPEN on cooldown."""
    state = rec.get("state", CLOSED)
    if state != OPEN:
        return state
    opened_at = rec.get("opened_at_ms", 0)
    cooldown_ms = float(rec.get("cooldown_s", 0.0)) * 1000.0
    if now_ms - opened_at >= cooldown_ms:
        return HALF_OPEN
    return OPEN


# ── Public state-machine API ────────────────────────────────────────────────

def is_open(
    cid: str,
    *,
    failure_class: str = AUTH,
    path: str = _DEFAULT_STATE_PATH,
    now_ms: Optional[int] = None,
) -> bool:
    """True iff the breaker for ``cid`` is currently OPEN (skip this credential).

    HALF_OPEN returns False — the caller is allowed exactly one trial. FAIL-OPEN:
    a missing/corrupt file ⇒ always False (credential treated as healthy).
    """
    now = now_ms if now_ms is not None else _now_ms()
    state = _load_state(path)
    rec = _get_record(state, _record_key(failure_class, cid))
    return _evaluate_state(rec, failure_class, now) == OPEN


def get_state(
    cid: str,
    *,
    failure_class: str = AUTH,
    path: str = _DEFAULT_STATE_PATH,
    now_ms: Optional[int] = None,
) -> str:
    """Return the effective state (CLOSED|OPEN|HALF_OPEN) for ``cid``."""
    now = now_ms if now_ms is not None else _now_ms()
    state = _load_state(path)
    rec = _get_record(state, _record_key(failure_class, cid))
    return _evaluate_state(rec, failure_class, now)


def record_failure(
    cid: str,
    failure_class: str = AUTH,
    *,
    path: str = _DEFAULT_STATE_PATH,
    now_ms: Optional[int] = None,
) -> str:
    """Record a failure of ``failure_class`` for ``cid``; return new state.

    Bounded-failure + backoff (§0a#2):
      * CONTENT_POLICY is request-scoped and NEVER marks the credential — this
        function no-ops (returns CLOSED) for it.
      * An ":unresolved" AUTH cid (the key env was not injected into this
        process) also no-ops — it is an injection/config bug, not a provider
        rejection, and must not strand an unclearable credential breaker
        (§0a#8 addendum).
      * In CLOSED: increment the consecutive counter inside the rolling window
        (a failure outside the window restarts the count at 1). Open only when
        the count reaches the threshold.
      * In HALF_OPEN: the single trial failed → go OPEN at the next backoff step.
      * Cooldown escalates per consecutive open (base→…cap).

    Race-safe (§0a#4): a write with a strictly-OLDER timestamp than the stored
    record is dropped (a stale failure cannot re-open a freshly-healthy cred).
    """
    if failure_class == CONTENT_POLICY:
        return CLOSED  # never marks the credential (§0a#3)

    if failure_class == AUTH and cid.endswith(":unresolved"):
        # §0a#8 addendum: an ":unresolved" cid means cred_id() could not
        # fingerprint the key — the KEY env var was not injected into THIS
        # process (a caller / boot-order injection bug), NOT a credential the
        # provider rejected (a real 401 always carries a resolved fingerprint).
        # Opening a credential breaker here strands a record that can never
        # clear (a later success uses the resolved-fingerprint id, not
        # ":unresolved") and surfaces in The Bridge as a false "fix it in
        # FluxCreds" alarm. No-op the credential breaker; surface the injection
        # gap loudly instead so the missing-injection bug gets fixed.
        logger.warning(
            "provider-auth: %s not resolved in this process — check secret "
            "injection (Infisical/FluxCreds) for this service; not opening a "
            "credential breaker",
            cid.split(":")[0],
        )
        return CLOSED

    now = now_ms if now_ms is not None else _now_ms()
    threshold, window_s, base_s, cap_s = _params_for_class(failure_class)
    window_ms = window_s * 1000.0

    state = _load_state(path)
    key = _record_key(failure_class, cid)
    rec = _get_record(state, key)

    # Strict-newer-timestamp guard (§0a#4): ignore stale out-of-order writes.
    if rec.get("updated_ms", 0) > now:
        return _evaluate_state(rec, failure_class, now)

    effective = _evaluate_state(rec, failure_class, now)

    if effective == HALF_OPEN or rec.get("state") == OPEN:
        # The single half-open trial (or a failure while open) → escalate.
        open_count = int(rec.get("open_count", 0)) + 1
        cooldown = min(base_s * (2 ** (open_count - 1)), cap_s)
        rec.update({
            "state": OPEN,
            "consecutive_failures": threshold,
            "open_count": open_count,
            "opened_at_ms": now,
            "cooldown_s": cooldown,
            "last_failure_ms": now,
            "updated_ms": now,
        })
        _commit_record(path, key, rec, now)
        return OPEN

    # CLOSED path — bounded consecutive-failure counting in a rolling window.
    win_start = rec.get("window_start_ms", 0)
    count = int(rec.get("consecutive_failures", 0))
    if count == 0 or (now - win_start) > window_ms:
        # Start (or restart) the window.
        win_start = now
        count = 1
    else:
        count += 1

    rec["consecutive_failures"] = count
    rec["window_start_ms"] = win_start
    rec["last_failure_ms"] = now
    rec["updated_ms"] = now

    if count >= threshold:
        open_count = int(rec.get("open_count", 0)) + 1
        cooldown = min(base_s * (2 ** (open_count - 1)), cap_s)
        rec.update({
            "state": OPEN,
            "open_count": open_count,
            "opened_at_ms": now,
            "cooldown_s": cooldown,
        })
        _commit_record(path, key, rec, now)
        return OPEN

    rec["state"] = CLOSED
    _commit_record(path, key, rec, now)
    return CLOSED


def record_success(
    cid: str,
    *,
    failure_class: str = AUTH,
    path: str = _DEFAULT_STATE_PATH,
    now_ms: Optional[int] = None,
) -> str:
    """Healthy-clear (§0a#4): close the breaker and zero the counter.

    Race-safe: a success with a strictly-OLDER timestamp than the stored record
    is dropped. On a winning write the state is CLOSED, counters reset, and
    ``open_count`` is reset to 0 so the NEXT open starts at the base cooldown.
    """
    now = now_ms if now_ms is not None else _now_ms()
    state = _load_state(path)
    key = _record_key(failure_class, cid)
    rec = _get_record(state, key)

    if rec.get("updated_ms", 0) > now:
        return _evaluate_state(rec, failure_class, now)

    rec.update({
        "state": CLOSED,
        "consecutive_failures": 0,
        "open_count": 0,
        "window_start_ms": 0,
        "opened_at_ms": 0,
        "cooldown_s": 0.0,
        "last_success_ms": now,
        "updated_ms": now,
    })
    _commit_record(path, key, rec, now)
    return CLOSED


# ── Fleet-shared probe hook (§0a#5) ─────────────────────────────────────────

def half_open_probe(
    entry: Dict[str, Any],
    *,
    probe: Optional[Callable[[Dict[str, Any]], bool]] = None,
    path: str = _DEFAULT_STATE_PATH,
    now_ms: Optional[int] = None,
) -> bool:
    """Run the HALF_OPEN trial for a fleet-shared credential (§0a#5).

    For ``scope=fleet-shared`` creds (the keyless ``fleet:nvidia`` door), the
    HALF_OPEN trial is an ACTIVE liveness probe against the shared endpoint
    rather than passive failure-counting — each host learns truth from the
    source, so there is no shared write path and no cross-host lock.

    The caller supplies ``probe(entry) -> bool`` (e.g. a 2s ``GET /v1/models``).
    On a truthy probe the breaker is closed via :func:`record_success`; on a
    falsy/raising probe it re-opens via :func:`record_failure`. Returns the
    probe result (True = healthy). FAIL-OPEN: with no probe supplied this
    returns True (do not block the spawn on a missing probe).

    A best-effort shared hint file is intentionally NOT consulted here — it is a
    HINT only and remains optional (see module docstring); local probing is
    authoritative.
    """
    cid = cred_id(entry)
    if probe is None:
        # No probe wired — fail open: treat as healthy, let the real request
        # discover truth and feed the passive counter.
        return True
    try:
        healthy = bool(probe(entry))
    except Exception as exc:
        logger.debug("provider_health: fleet probe raised (%s) — treating as down", exc)
        healthy = False
    if healthy:
        record_success(cid, failure_class=AUTH, path=path, now_ms=now_ms)
    else:
        record_failure(cid, AUTH, path=path, now_ms=now_ms)
    return healthy
