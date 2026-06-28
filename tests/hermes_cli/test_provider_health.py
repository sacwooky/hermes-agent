"""Tests for the credential-keyed circuit breaker (hermes_cli.provider_health).

Covers §9 acceptance cases 1–15 of the v2 design
(conductor-vault/wiki/projects/preflight-credential-failover-circuit-breaker.md).

All time is injected via ``now_ms`` (monotonic-ms) so the state machine is
deterministic without sleeping. The state file is a per-test tmp path so writes
are isolated and the FAIL-OPEN paths are exercised on real files.
"""

from __future__ import annotations

import json
import os

import pytest

from hermes_cli import provider_health as ph


@pytest.fixture()
def state_path(tmp_path):
    return str(tmp_path / "provider-auth-health.json")


# ── §0a#2: bounded-failure + backoff ────────────────────────────────────────

def test_single_401_does_not_open(state_path):
    """§9.7 — a lone transient 401 must NOT open the breaker."""
    assert ph.record_failure("k:abc", ph.AUTH, path=state_path, now_ms=1000) == ph.CLOSED
    assert ph.is_open("k:abc", path=state_path, now_ms=1000) is False


def test_n_consecutive_opens(state_path):
    """§9.7 — N=3 consecutive AUTH failures inside the window opens the breaker."""
    ph.record_failure("k:abc", ph.AUTH, path=state_path, now_ms=1000)
    ph.record_failure("k:abc", ph.AUTH, path=state_path, now_ms=1100)
    st = ph.record_failure("k:abc", ph.AUTH, path=state_path, now_ms=1200)
    assert st == ph.OPEN
    assert ph.is_open("k:abc", path=state_path, now_ms=1300) is True


def test_unresolved_auth_cid_never_opens(state_path):
    """§0a#8 addendum — an ":unresolved" cid means the key env was not injected
    into THIS process (a caller/boot-order bug), not a provider rejection. AUTH
    failures on it must NOT open a credential breaker: such a record could never
    clear (a later success uses the resolved-fingerprint id) and would render in
    The Bridge as a false "fix it in FluxCreds" alarm. A resolved-fingerprint
    cid with the same env still opens normally."""
    cid = "NINEROUTER_KEY:unresolved"
    for t in (1000, 1100, 1200, 1300):
        assert ph.record_failure(cid, ph.AUTH, path=state_path, now_ms=t) == ph.CLOSED
    assert ph.is_open(cid, path=state_path, now_ms=1400) is False
    # Contrast: the SAME env with a resolved fingerprint still opens at N=3.
    resolved = "NINEROUTER_KEY:7e66a8e4"
    ph.record_failure(resolved, ph.AUTH, path=state_path, now_ms=1000)
    ph.record_failure(resolved, ph.AUTH, path=state_path, now_ms=1100)
    assert ph.record_failure(resolved, ph.AUTH, path=state_path, now_ms=1200) == ph.OPEN


def test_failures_outside_window_do_not_accumulate(state_path):
    """A failure outside the 120s rolling window restarts the count at 1."""
    ph.record_failure("k:abc", ph.AUTH, path=state_path, now_ms=1000)
    ph.record_failure("k:abc", ph.AUTH, path=state_path, now_ms=2000)
    # 3rd failure is >120s after window_start (1000) -> window restarts, count=1
    st = ph.record_failure("k:abc", ph.AUTH, path=state_path, now_ms=1000 + 200_000)
    assert st == ph.CLOSED
    assert ph.is_open("k:abc", path=state_path, now_ms=1000 + 200_001) is False


def test_success_resets_consecutive_counter(state_path):
    """A success between failures resets the consecutive counter to 0."""
    ph.record_failure("k:abc", ph.AUTH, path=state_path, now_ms=1000)
    ph.record_failure("k:abc", ph.AUTH, path=state_path, now_ms=1100)
    ph.record_success("k:abc", path=state_path, now_ms=1150)
    # Two more failures should NOT open (counter was reset).
    ph.record_failure("k:abc", ph.AUTH, path=state_path, now_ms=1200)
    st = ph.record_failure("k:abc", ph.AUTH, path=state_path, now_ms=1300)
    assert st == ph.CLOSED


def _open_once(path, cid, *, base_ms):
    ph.record_failure(cid, ph.AUTH, path=path, now_ms=base_ms)
    ph.record_failure(cid, ph.AUTH, path=path, now_ms=base_ms + 10)
    return ph.record_failure(cid, ph.AUTH, path=path, now_ms=base_ms + 20)


def test_backoff_escalation(state_path):
    """§9.8 — repeated opens escalate cooldown 60→120→240…; success resets base."""
    cid = "k:abc"
    assert _open_once(state_path, cid, base_ms=1000) == ph.OPEN
    rec = json.load(open(state_path))["records"]["auth::" + cid]
    assert rec["cooldown_s"] == 60.0
    opened_at = rec["opened_at_ms"]

    # half-open after first cooldown, then a half-open failure escalates to 120s
    assert ph.get_state(cid, path=state_path, now_ms=opened_at + 60_000) == ph.HALF_OPEN
    ph.record_failure(cid, ph.AUTH, path=state_path, now_ms=opened_at + 60_000)
    rec = json.load(open(state_path))["records"]["auth::" + cid]
    assert rec["cooldown_s"] == 120.0
    opened_at2 = rec["opened_at_ms"]

    # next half-open failure escalates to 240s
    ph.record_failure(cid, ph.AUTH, path=state_path, now_ms=opened_at2 + 120_000)
    rec = json.load(open(state_path))["records"]["auth::" + cid]
    assert rec["cooldown_s"] == 240.0

    # a success resets open_count so the NEXT open starts back at base 60s
    ph.record_success(cid, path=state_path, now_ms=opened_at2 + 500_000)
    assert _open_once(state_path, cid, base_ms=opened_at2 + 600_000) == ph.OPEN
    rec = json.load(open(state_path))["records"]["auth::" + cid]
    assert rec["cooldown_s"] == 60.0


def test_backoff_capped_at_600(state_path):
    """Cooldown never exceeds the 600s cap regardless of open count."""
    cid = "k:abc"
    _open_once(state_path, cid, base_ms=1000)
    rec = json.load(open(state_path))["records"]["auth::" + cid]
    # Drive many half-open failures; cooldown should saturate at cap.
    for _ in range(12):
        opened_at = rec["opened_at_ms"]
        cooldown_ms = rec["cooldown_s"] * 1000
        ph.record_failure(cid, ph.AUTH, path=state_path, now_ms=int(opened_at + cooldown_ms))
        rec = json.load(open(state_path))["records"]["auth::" + cid]
    assert rec["cooldown_s"] == ph.AUTH_COOLDOWN_CAP_S == 600.0


# ── §0a#2: half-open single trial ───────────────────────────────────────────

def test_half_open_trial_success_closes(state_path):
    """§9.9 — exactly one trial after cooldown; success → CLOSED + reset."""
    cid = "k:abc"
    _open_once(state_path, cid, base_ms=1000)
    rec = json.load(open(state_path))["records"]["auth::" + cid]
    opened_at = rec["opened_at_ms"]
    assert ph.get_state(cid, path=state_path, now_ms=opened_at + 60_000) == ph.HALF_OPEN
    ph.record_success(cid, path=state_path, now_ms=opened_at + 60_000)
    assert ph.get_state(cid, path=state_path, now_ms=opened_at + 60_001) == ph.CLOSED
    rec = json.load(open(state_path))["records"]["auth::" + cid]
    assert rec["consecutive_failures"] == 0
    assert rec["open_count"] == 0


def test_half_open_trial_failure_reopens(state_path):
    """§9.9 — a half-open failure re-opens at the NEXT backoff step."""
    cid = "k:abc"
    _open_once(state_path, cid, base_ms=1000)
    rec = json.load(open(state_path))["records"]["auth::" + cid]
    opened_at = rec["opened_at_ms"]
    st = ph.record_failure(cid, ph.AUTH, path=state_path, now_ms=opened_at + 60_000)
    assert st == ph.OPEN
    assert ph.is_open(cid, path=state_path, now_ms=opened_at + 60_001) is True


# ── §0a#4: healthy-clear + stale-failure race ───────────────────────────────

def test_healthy_clear_stale_failure_race(state_path):
    """§9.10 — a stale failure (older ts) cannot re-open a freshly-closed cred."""
    cid = "k:abc"
    _open_once(state_path, cid, base_ms=1000)
    # A faster worker proves the key healthy at t=5000.
    ph.record_success(cid, path=state_path, now_ms=5000)
    assert ph.is_open(cid, path=state_path, now_ms=5001) is False
    # A slow worker's STALE failure (older timestamp t=4000) must be dropped.
    ph.record_failure(cid, ph.AUTH, path=state_path, now_ms=4000)
    rec = json.load(open(state_path))["records"]["auth::" + cid]
    assert rec["state"] == ph.CLOSED
    assert rec["updated_ms"] == 5000  # the newer success record stands


def test_strict_newer_merge_drops_stale_success(state_path):
    """A stale success cannot clobber a newer failure record (strict-newer)."""
    cid = "k:abc"
    ph.record_failure(cid, ph.AUTH, path=state_path, now_ms=5000)
    ph.record_success(cid, path=state_path, now_ms=4000)  # older -> dropped
    rec = json.load(open(state_path))["records"]["auth::" + cid]
    assert rec["last_failure_ms"] == 5000
    assert rec["updated_ms"] == 5000


# ── §0a#3: failure-class taxonomy ───────────────────────────────────────────

def test_403_content_policy_does_not_mark(state_path):
    """§9.11 — content_policy_blocked never marks the credential breaker."""
    cls = ph.classify_failure(status_code=403, reason="content_policy_blocked")
    assert cls == ph.CONTENT_POLICY
    # record_failure is a no-op for CONTENT_POLICY.
    for t in (1000, 1100, 1200, 1300):
        assert ph.record_failure("k:abc", ph.CONTENT_POLICY, path=state_path, now_ms=t) == ph.CLOSED
    assert ph.is_open("k:abc", path=state_path, now_ms=1400) is False
    # The file should have no record at all.
    assert not os.path.exists(state_path) or "auth::k:abc" not in json.load(open(state_path)).get("records", {})


def test_429_separate_breaker(state_path):
    """§9.12 — 429/billing routes to a SEPARATE shorter breaker, same key."""
    assert ph.classify_failure(status_code=429) == ph.RATE_LIMIT
    cid = "k:abc"
    # Open the rate-limit breaker (base 30s) without touching the auth breaker.
    ph.record_failure(cid, ph.RATE_LIMIT, path=state_path, now_ms=1000)
    ph.record_failure(cid, ph.RATE_LIMIT, path=state_path, now_ms=1100)
    st = ph.record_failure(cid, ph.RATE_LIMIT, path=state_path, now_ms=1200)
    assert st == ph.OPEN
    assert ph.is_open(cid, failure_class=ph.RATE_LIMIT, path=state_path, now_ms=1300) is True
    # The AUTH breaker for the SAME key is untouched (distinct namespace).
    assert ph.is_open(cid, failure_class=ph.AUTH, path=state_path, now_ms=1300) is False
    rec = json.load(open(state_path))["records"]["rate_limit::" + cid]
    assert rec["cooldown_s"] == ph.RATE_COOLDOWN_BASE_S == 30.0


def test_classify_failure_mappings():
    """Spot-check the §0a#3 taxonomy across status codes + reasons."""
    assert ph.classify_failure(status_code=401) == ph.AUTH
    assert ph.classify_failure(status_code=403, is_auth=True) == ph.AUTH
    assert ph.classify_failure(reason="auth") == ph.AUTH
    assert ph.classify_failure(reason="billing") == ph.RATE_LIMIT
    assert ph.classify_failure(status_code=500) == ph.AVAILABILITY
    assert ph.classify_failure(status_code=503) == ph.AVAILABILITY
    assert ph.classify_failure() == ph.AVAILABILITY  # connection drop, no status


# ── §0a#8: availability vs auth scope (endpoint vs credential) ──────────────

def test_availability_vs_auth_scope(state_path):
    """§9.13 — a 5xx on one endpoint must NOT blacklist the shared key elsewhere."""
    entry_a = {"provider": "custom:9router-codex", "model": "cx/gpt-5.5",
               "base_url": "http://127.0.0.1:20128/v1", "key_env": "NINEROUTER_KEY"}
    entry_b = {"provider": "custom:9router-codex", "model": "vertex/gemini-2.5-flash",
               "base_url": "http://127.0.0.1:9999/v1", "key_env": "NINEROUTER_KEY"}
    # AUTH groups by credential -> same cred_id for both.
    assert ph.cred_id(entry_a) == ph.cred_id(entry_b)
    # AVAILABILITY keys by endpoint -> DIFFERENT ids (different host:port).
    eid_a, eid_b = ph.endpoint_id(entry_a), ph.endpoint_id(entry_b)
    assert eid_a != eid_b
    assert eid_a == "127.0.0.1:20128"
    # Open availability on endpoint A; endpoint B stays healthy.
    for t in (1000, 1100, 1200):
        ph.record_failure(eid_a, ph.AVAILABILITY, path=state_path, now_ms=t)
    assert ph.is_open(eid_a, failure_class=ph.AVAILABILITY, path=state_path, now_ms=1300) is True
    assert ph.is_open(eid_b, failure_class=ph.AVAILABILITY, path=state_path, now_ms=1300) is False


# ── §0a#8: credential grouping ──────────────────────────────────────────────

def test_credential_grouping_drops_both_9router_tiers(monkeypatch, state_path):
    """§9.2 — a NINEROUTER_KEY 401 drops BOTH gpt-5.5 and gemini-via-9router.

    Both routes resolve to the SAME credential fingerprint, so opening the
    breaker on one closes the door on the other.
    """
    monkeypatch.setenv("NINEROUTER_KEY", "secret-value-123")
    gpt = {"provider": "custom:9router-codex", "model": "cx/gpt-5.5",
           "base_url": "http://127.0.0.1:20128/v1", "key_env": "NINEROUTER_KEY"}
    gem = {"provider": "custom:9router-codex", "model": "vertex/gemini-2.5-flash",
           "base_url": "http://127.0.0.1:20128/v1", "key_env": "NINEROUTER_KEY"}
    assert ph.cred_id(gpt) == ph.cred_id(gem)
    # The id includes env name + 8-char hash, NEVER the value.
    assert ph.cred_id(gpt).startswith("NINEROUTER_KEY:")
    assert "secret-value-123" not in ph.cred_id(gpt)
    cid = ph.cred_id(gpt)
    for t in (1000, 1100, 1200):
        ph.record_failure(cid, ph.AUTH, path=state_path, now_ms=t)
    # Opening via gpt's cred_id also opens for gemini (same id).
    assert ph.is_open(ph.cred_id(gem), path=state_path, now_ms=1300) is True


def test_cred_id_never_contains_key_value(monkeypatch):
    """The fingerprint stores only env-name + short hash, never the secret."""
    monkeypatch.setenv("ZAI_API_KEY", "zzz-super-secret")
    entry = {"provider": "zai", "model": "glm-5.1", "base_url": "https://z.ai/paas/v1"}
    cid = ph.cred_id(entry)
    assert "zzz-super-secret" not in cid
    assert cid.startswith("ZAI_API_KEY:")
    # Hash is exactly 8 hex chars.
    suffix = cid.split(":", 1)[1]
    assert len(suffix) == 8 and all(c in "0123456789abcdef" for c in suffix)


def test_distinct_keys_distinct_cred_ids(monkeypatch):
    """Two different env vars / values must produce distinct cred_ids."""
    monkeypatch.setenv("NINEROUTER_KEY", "aaa")
    monkeypatch.setenv("ZAI_API_KEY", "bbb")
    nine = {"provider": "custom:9router-codex", "model": "cx/gpt-5.5",
            "key_env": "NINEROUTER_KEY", "base_url": "http://127.0.0.1:20128/v1"}
    zai = {"provider": "zai", "model": "glm-5.1", "base_url": "https://z.ai/paas/v1"}
    assert ph.cred_id(nine) != ph.cred_id(zai)


# ── §0a#5: fleet-shared probe ───────────────────────────────────────────────

def test_fleet_shared_cred_id():
    """A keyless fleet-shared door keys on its scope marker, not a key hash."""
    entry = {"provider": "custom:9router-nvidia", "model": "nvidia/llama-3.1-8b",
             "base_url": "http://100.121.15.7:20128/v1", "scope": "fleet-shared"}
    assert ph.scope_of(entry) == ph.SCOPE_FLEET_SHARED
    assert ph.cred_id(entry) == "fleet:nvidia"


def test_fleet_shared_probe_healthy_closes(state_path):
    """§9.14 — a successful half-open probe closes the fleet-shared breaker."""
    entry = {"provider": "custom:9router-nvidia", "model": "nvidia/llama-3.1-8b",
             "base_url": "http://100.121.15.7:20128/v1", "scope": "fleet-shared"}
    cid = ph.cred_id(entry)
    # Pre-open it.
    for t in (1000, 1100, 1200):
        ph.record_failure(cid, ph.AUTH, path=state_path, now_ms=t)
    assert ph.is_open(cid, path=state_path, now_ms=1300) is True
    healthy = ph.half_open_probe(entry, probe=lambda e: True, path=state_path, now_ms=2000)
    assert healthy is True
    assert ph.is_open(cid, path=state_path, now_ms=2001) is False


def test_fleet_shared_probe_down_reopens(state_path):
    """A failing probe re-opens the breaker."""
    entry = {"provider": "custom:9router-nvidia", "model": "nvidia/llama-3.1-8b",
             "base_url": "http://100.121.15.7:20128/v1", "scope": "fleet-shared"}
    cid = ph.cred_id(entry)
    ph.record_success(cid, path=state_path, now_ms=1000)
    res = ph.half_open_probe(entry, probe=lambda e: False, path=state_path, now_ms=2000)
    assert res is False
    # One probe failure alone won't open (needs N) but it is recorded.
    rec = json.load(open(state_path))["records"]["auth::" + cid]
    assert rec["consecutive_failures"] >= 1


def test_fleet_shared_probe_failopen_without_probe(state_path):
    """§9.14 — no probe supplied ⇒ fail-open (treat as healthy)."""
    entry = {"provider": "custom:9router-nvidia", "model": "n", "scope": "fleet-shared",
             "base_url": "http://100.121.15.7:20128/v1"}
    assert ph.half_open_probe(entry, probe=None, path=state_path, now_ms=1000) is True


def test_fleet_shared_probe_raises_treated_as_down(state_path):
    """A probe that raises is treated as down, not propagated."""
    entry = {"provider": "custom:9router-nvidia", "model": "n", "scope": "fleet-shared",
             "base_url": "http://100.121.15.7:20128/v1"}
    def _boom(e):
        raise RuntimeError("network")
    assert ph.half_open_probe(entry, probe=_boom, path=state_path, now_ms=1000) is False


# ── §8 / §0a: fail-open on missing or corrupt file ──────────────────────────

def test_fail_open_missing_file(tmp_path):
    """§9.5 — a missing health file ⇒ all credentials healthy (never blocks)."""
    missing = str(tmp_path / "does-not-exist.json")
    assert ph.is_open("k:abc", path=missing, now_ms=1000) is False
    assert ph.get_state("k:abc", path=missing, now_ms=1000) == ph.CLOSED


def test_fail_open_corrupt_file(state_path):
    """§9.5 — a corrupt health file ⇒ treated as all-healthy, no exception."""
    with open(state_path, "w") as fh:
        fh.write("{ this is not valid json ")
    assert ph.is_open("k:abc", path=state_path, now_ms=1000) is False
    # A subsequent write should still succeed (overwrites the corrupt file).
    ph.record_failure("k:abc", ph.AUTH, path=state_path, now_ms=1000)
    assert isinstance(json.load(open(state_path)), dict)


def test_fail_open_unwritable_dir_does_not_raise(tmp_path):
    """A write to an unwritable location fails open (returns, no exception)."""
    # Point at a path under a file (not a dir) so makedirs/replace fail.
    bad_parent = tmp_path / "afile"
    bad_parent.write_text("x")
    bad_path = str(bad_parent / "nested" / "h.json")
    # Should not raise; record_failure swallows the write error.
    st = ph.record_failure("k:abc", ph.AUTH, path=bad_path, now_ms=1000)
    assert st in (ph.CLOSED, ph.OPEN)
    # And reads still fail-open.
    assert ph.is_open("k:abc", path=bad_path, now_ms=1000) is False


# ── §9.1 dead-key skip (cred-level), §9.6 diagnosable (via is_open list) ─────

def test_dead_key_skip_until_cooldown(state_path):
    """§9.1 — once a credential is open, is_open stays True until cooldown."""
    cid = "zai:deadbeef"
    _open_once(state_path, cid, base_ms=1000)
    rec = json.load(open(state_path))["records"]["auth::" + cid]
    opened_at = rec["opened_at_ms"]
    # Open right up to the cooldown boundary.
    assert ph.is_open(cid, path=state_path, now_ms=opened_at + 59_999) is True
    # At/after cooldown it becomes HALF_OPEN (is_open False -> one trial allowed).
    assert ph.is_open(cid, path=state_path, now_ms=opened_at + 60_000) is False


# ── persistence / multi-writer ──────────────────────────────────────────────

def test_records_namespaced_by_class(state_path):
    """auth / rate_limit / availability records are independent namespaces."""
    cid = "k:abc"
    ph.record_failure(cid, ph.AUTH, path=state_path, now_ms=1000)
    ph.record_failure(cid, ph.RATE_LIMIT, path=state_path, now_ms=1000)
    ph.record_failure(cid, ph.AVAILABILITY, path=state_path, now_ms=1000)
    records = json.load(open(state_path))["records"]
    assert "auth::" + cid in records
    assert "rate_limit::" + cid in records
    assert "availability::" + cid in records
