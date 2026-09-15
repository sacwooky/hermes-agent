#!/usr/bin/env python3
"""Deterministic review path for Claude-authored work.

The review DRIVER model (gemini via the vertex proxy, LLM-driven) is unreliable:
it ignores lane guidance and sometimes does not drive the review at all. For
Claude-Code-authored packets that fragility is unacceptable, so this script
runs the mandated INDEPENDENT lane directly and writes the SAME signed verdict
artifact + kanban event that record_verdict produces. No LLM driver, no lane
discretion.

Lanes (2026-06, PRC review models RETIRED per
decision-fleet-model-review-toolchain-policy-v1):
  * primary (binding)   = gemini   -> gemini-2.5-pro via Robin's local Vertex
                          proxy. Independent of Claude (different model/provider),
                          so Claude never self-reviews.
  * second (advisory)   = gemini-flash -> gemini-2.5-flash, a SAME-PROVIDER
                          cross-check (cheap). NOT a truly independent provider;
                          a second non-PRC provider lane is a follow-up.
The retired PRC lanes (minimax / kimi / claude-glm) are kept in LANES only as
dead references for the legacy driver path; the deterministic flow never invokes
them.

Usage: review_deterministic.py <task_id> <commit> <packet_dir> <risk>
"""
import sys, os, re, json, time, hmac, hashlib, shutil, subprocess
from pathlib import Path

# Ensure user-local bin (gemini-review lives in ~/.local/bin) resolves regardless of
# the caller PATH -- keeps the deterministic fallback from silently failing on PATH.
_LOCAL_BIN = str(Path.home() / ".local" / "bin")
if _LOCAL_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _LOCAL_BIN + os.pathsep + os.environ.get("PATH", "")

REPO = "/srv/fluxlabs/repos/hermes-agent"
sys.path.insert(0, REPO)

# fusionkit lives next to this script; the Fusion path and this fallback MUST agree on who the
# author is, or the two paths enforce different independence guards (run 2267).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusionkit import authorship, routing  # noqa: E402

LANE_TIMEOUT = 900
VERDICT_KEY_PATH = Path.home() / ".hermes" / "credentials" / "robin-verdict-key"
VERDICT_DIR = Path.home() / ".hermes" / "verdicts"
ALLOWED = {"PASS", "FAIL", "CHANGES_REQUESTED", "BLOCKED"}

# lane id -> CLI binary. gemini lanes are the active (non-PRC) review lanes; the
# minimax/kimi/claude-glm entries are RETIRED (PRC models removed) and left only
# so the legacy driver path does not KeyError.
LANES = {
    "gemini": "gemini-review",   # Google, via Robin's local Vertex proxy
    "gemini-flash": "gemini-review",
    "claude": "claude",          # Anthropic, Claude Code harness via 9router cc/ lane
    "grok": "grok-review",       # xAI, via 9router openrouter/ lane (OpenAI-shaped shim)
    # retired PRC lanes (not invoked by the deterministic path):
    "minimax": "claude-minimax", "kimi": "kimi", "claude-glm": "claude-glm",
}

# Per-lane model CHAINS: primary first, then SAME-PROVIDER fallbacks so each lane keeps
# its INDEPENDENT provider identity even when its primary is unavailable (Keith, run 2368).
# Never inherit a host default (drift -> hollow verdict, run 357).
LANE_MODEL_CHAINS = {
    "gemini": ["gemini-3.1-pro-preview", "gemini-2.5-pro"],                        # Google (Vertex proxy)
    "claude": ["cc/claude-fable-5", "cc/claude-opus-4-8", "cc/claude-sonnet-5"],   # Anthropic (9router)
    "grok":   ["openrouter/x-ai/grok-4.6", "openrouter/x-ai/grok-4.3"],            # xAI (9router)
    "gemini-flash": ["gemini-2.5-flash"],
    "minimax": ["MiniMax-M3"], "claude-glm": ["glm-4.7"],
}
# First model of each chain = the lane's primary (build_cmd default + the SC path).
LANE_MODELS = {lane: chain[0] for lane, chain in LANE_MODEL_CHAINS.items()}


def _ninerouter_key():
    v = os.environ.get("NINEROUTER_KEY")
    if v:
        return v
    envf = Path.home() / ".hermes" / ".env"
    if envf.is_file():
        for line in envf.read_text(errors="ignore").splitlines():
            m = re.match(r"\s*NINEROUTER_KEY\s*=\s*[\"']?([^\"'\s]+)", line)
            if m:
                return m.group(1)
    return None


def _lane_env(lane):
    """Per-lane env. The `claude` (Opus) lane drives the Claude Code binary, which is
    NOT natively logged in on Robin -> point it at the local 9router lab. gemini/grok
    lanes handle their own endpoint + auth, so they inherit the plain environment."""
    env = dict(os.environ)
    if lane == "claude":
        env["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:20128"
        key = _ninerouter_key()
        if key:
            env["ANTHROPIC_AUTH_TOKEN"] = key
    return env


def build_cmd(lane, bin_path, prompt, cwd, model=None):
    m = model or LANE_MODELS.get(lane)
    if lane == "kimi":
        cmd = [bin_path, "--quiet", "--afk", "-p", prompt]
        if cwd:
            cmd += ["-w", cwd]
        return cmd
    cmd = [bin_path, "-p", prompt, "--output-format", "text"]
    if m:
        cmd += ["--model", m]
    return cmd


def run_lane(lane, instructions, cwd, diff_name):
    """Run a review lane, trying each model in its chain until one returns a parseable
    VERDICT. Returns (review_text, model_used, err). Requiring a VERDICT (not just any
    output) both drives same-provider fallback when a model is unavailable AND defends
    against the gated-model hollow-verdict trap (run 357: 'model unavailable' at exit 0)."""
    binp = shutil.which(LANES[lane])
    if not binp:
        return None, None, f"(lane {lane}: binary {LANES[lane]} not on PATH)"
    prompt = "\n".join([
        "You are an independent code reviewer. Review ONLY the supplied diff for the "
        "stated commit -- do not review unrelated code and do not modify anything.",
        "", instructions.strip(),
        "", f"The diff to review is at: {diff_name}",
        "", "End your response with a clear verdict line in this exact form:",
        "VERDICT: <PASS|FAIL|CHANGES_REQUESTED>",
        "followed by: evidence checked, risks, and what would change your mind.",
    ])
    last_err = None
    for model in LANE_MODEL_CHAINS.get(lane) or [LANE_MODELS.get(lane)]:
        try:
            p = subprocess.run(build_cmd(lane, binp, prompt, cwd, model=model), cwd=cwd,
                               stdin=subprocess.DEVNULL, capture_output=True,
                               text=True, timeout=LANE_TIMEOUT, env=_lane_env(lane))
        except subprocess.TimeoutExpired:
            last_err = f"(lane {lane}/{model}: timed out after {LANE_TIMEOUT}s)"
            continue
        out = (p.stdout or "").strip()
        if out and parse_verdict(out):
            return out, model, None
        last_err = (f"(lane {lane}/{model}: "
                    + ("no VERDICT in output" if out else "empty output")
                    + f"; rc={p.returncode}; stderr={(p.stderr or '')[:200]})")
    return None, None, last_err


def parse_verdict(text):
    m = re.search(r"VERDICT:\s*(PASS|FAIL|CHANGES_REQUESTED|BLOCKED)", text or "", re.I)
    return m.group(1).upper() if m else None


# Single-lane SC types (one Gemini juror over inlined evidence). `code` is an SC
# contract type too, but it keeps the author-disjoint 3-provider PANEL and is
# handled by run_code_panel_review below -- never by the single-lane path.
SC_TYPES = {"wireframe", "prd", "general"}

# v18 Supreme Court contract shape for the CODE panel (kept in lockstep with
# hermes_cli.kanban_autonomy SC_* constants; duplicated so the panel path stays
# runnable/testable without the DB import chain).
CODE_SC_REQUIRED_FIELDS = (
    "verdict",
    "confidence",
    "scorecard",
    "blocking_issues",
    "advisory_issues",
    "missing_skill_findings",
    "required_repair_actions",
    "evidence_reviewed",
    "calibration_substrate_flags",
)
CODE_SC_LIST_FIELDS = (
    "blocking_issues",
    "advisory_issues",
    "missing_skill_findings",
    "required_repair_actions",
    "evidence_reviewed",
    "calibration_substrate_flags",
)
CODE_SC_PASS_VERDICTS = frozenset({"approved", "approved_with_minor_notes"})
CODE_SC_BLOCK_VERDICTS = frozenset({"rejected_for_revision", "rejected_wrong_skill_stack"})


def _norm_sc_verdict(v):
    return str(v or "").strip().lower().replace("-", "_").replace(" ", "_")


def build_code_sc_prompt(instructions, diff_name):
    """Structured-contract prompt for one CODE panel lane. Same review scope as the
    legacy panel prompt (the supplied diff only), but the reply must be the full v18
    SC JSON object -- verdict + confidence + rubric scorecard + the six list fields."""
    return "\n".join([
        "You are an independent code reviewer on a review panel. Review ONLY the "
        "supplied diff for the stated commit -- do not review unrelated code and "
        "do not modify anything.",
        "", instructions.strip(),
        "", f"The diff to review is at: {diff_name}",
        "",
        "Score against templates/rubrics/code-rubric.md: fill a scorecard with that "
        "rubric's categories (0-10 each, or \"insufficient_evidence\"), apply its "
        "thresholds, and emit ONE verdict from the v18 vocabulary. For any criterion "
        "the diff/evidence does not let you judge, score it \"insufficient_evidence\" "
        "rather than guessing. This is a BINDING gate.",
        "",
        "Respond with ONLY a JSON object (no prose before or after) carrying EVERY "
        "field below -- a response missing any field, or with an empty scorecard, is "
        "rejected as a hollow verdict:",
        '{"verdict": "approved"|"approved_with_minor_notes"|"rejected_for_revision"'
        '|"rejected_wrong_skill_stack", '
        '"confidence": 0.0-1.0, '
        '"scorecard": {"<rubric_category>": <0-10|"insufficient_evidence">, ...}, '
        '"blocking_issues": [...], "advisory_issues": [...], '
        '"missing_skill_findings": [...], "required_repair_actions": [...], '
        '"evidence_reviewed": [...], "calibration_substrate_flags": [...]}',
    ])


def _lane_reply_error(parsed):
    """Validate one lane's parsed structured reply. Returns an error string, or None
    when the reply satisfies the per-lane contract shape. A lane that cannot produce
    a conforming structured reply is UNAVAILABLE -- never silently legacy."""
    if not isinstance(parsed, dict):
        return "reply is not a JSON object"
    missing = [f for f in CODE_SC_REQUIRED_FIELDS if f not in parsed]
    if missing:
        return f"missing required structured field(s) {missing}"
    if _norm_sc_verdict(parsed.get("verdict")) not in (CODE_SC_PASS_VERDICTS | CODE_SC_BLOCK_VERDICTS):
        return f"verdict {parsed.get('verdict')!r} not in the v18 vocabulary"
    sc = parsed.get("scorecard")
    if not isinstance(sc, dict) or not sc:
        return "rubric scorecard is missing or empty"
    for f in CODE_SC_LIST_FIELDS:
        if not isinstance(parsed.get(f), list):
            return f"field {f!r} must be a JSON list"
    return None


def run_structured_lane(lane, prompt, cwd, parse_reply):
    """Run one CODE panel lane requiring a full structured SC reply, trying each model
    in the lane's same-provider chain. Returns (parsed, model_used, err). A reply that
    is missing, unparseable (per parse_reply), or contract-invalid does NOT count --
    the chain advances, and if the whole chain fails the provider is UNAVAILABLE
    (fail-closed; never a silent fall back to the legacy freeform verdict)."""
    binp = shutil.which(LANES[lane])
    if not binp:
        return None, None, f"(lane {lane}: binary {LANES[lane]} not on PATH)"
    last_err = None
    for model in LANE_MODEL_CHAINS.get(lane) or [LANE_MODELS.get(lane)]:
        try:
            p = subprocess.run(build_cmd(lane, binp, prompt, cwd, model=model), cwd=cwd,
                               stdin=subprocess.DEVNULL, capture_output=True,
                               text=True, timeout=LANE_TIMEOUT, env=_lane_env(lane))
        except subprocess.TimeoutExpired:
            last_err = f"(lane {lane}/{model}: timed out after {LANE_TIMEOUT}s)"
            continue
        out = (p.stdout or "").strip()
        parsed = parse_reply(out) if out else None
        if parsed is not None:
            err = _lane_reply_error(parsed)
            if err is None:
                return parsed, model, None
            last_err = f"(lane {lane}/{model}: structured reply invalid: {err})"
            continue
        last_err = (f"(lane {lane}/{model}: "
                    + ("no parseable SC JSON in output" if out else "empty output")
                    + f"; rc={p.returncode}; stderr={(p.stderr or '')[:200]})")
    return None, None, last_err


def aggregate_code_panel(lane_results):
    """Aggregate per-lane structured verdicts under the existing UNANIMOUS-among-
    available policy. Any rejection => rejected_for_revision; all approved-family =>
    approved (or approved_with_minor_notes if any lane raised minor notes). List
    fields are unioned + deduped across lanes; scorecard and confidence carry
    per-provider entries so the panel structure survives into the signed payload."""
    verdicts = {l: _norm_sc_verdict(r.get("verdict")) for l, r in lane_results.items()}
    if any(v in CODE_SC_BLOCK_VERDICTS for v in verdicts.values()):
        verdict = "rejected_for_revision"
    elif any(v == "approved_with_minor_notes" for v in verdicts.values()):
        verdict = "approved_with_minor_notes"
    else:
        verdict = "approved"

    def _union(field):
        out, seen = [], set()
        for lane in sorted(lane_results):
            vals = lane_results[lane].get(field)
            if not isinstance(vals, list):
                continue
            for v in vals:
                key = json.dumps(v, sort_keys=True, ensure_ascii=False) \
                    if isinstance(v, (dict, list)) else ("s:" + str(v))
                if key not in seen:
                    seen.add(key)
                    out.append(v)
        return out

    agg = {
        "verdict": verdict,
        "confidence": {l: lane_results[l].get("confidence") for l in sorted(lane_results)},
        "scorecard": {l: (lane_results[l].get("scorecard") or {}) for l in sorted(lane_results)},
        "per_provider_verdicts": {l: verdicts[l] for l in sorted(verdicts)},
    }
    for f in CODE_SC_LIST_FIELDS:
        agg[f] = _union(f)
    return agg


def run_code_panel_review(task_id, commit, pkt, instructions, diff_name):
    """Explicit review-type: code -- author-disjoint 3-provider deterministic panel
    (unchanged routing/independence/fallback policy) that emits the v18 SC structured
    contract instead of the legacy freeform verdict, stamped review_type='code' and
    recorded with expected_review_type so the contract is REQUIRED at record time."""
    from hermes_cli.review_loop import sc_review as scr

    try:
        author_provider, author_source = authorship.parse_author(instructions)
    except authorship.AuthorDeclarationError as e:
        print(f"ABORT: {e}", file=sys.stderr)
        print("No verdict recorded. Stamp the packet and re-send.", file=sys.stderr)
        sys.exit(3)

    LANE_LABEL = {"gemini": "Gemini (Google)", "claude": "Claude (Anthropic)", "grok": "Grok (xAI)"}
    PANEL = [
        lane for lane in ("gemini", "claude", "grok")
        if routing.provider_of(LANE_MODELS[lane]) != author_provider
    ]
    if not PANEL:
        print(f"ABORT: no author-disjoint lane available for author {author_provider!r}.",
              file=sys.stderr)
        sys.exit(3)
    print(f"deterministic CODE-SC panel (author={author_provider} via {author_source}): "
          f"{[LANE_LABEL[l] for l in PANEL]}", file=sys.stderr)

    prompt = build_code_sc_prompt(instructions, diff_name)
    results, models_used, errors = {}, {}, {}
    for lane in PANEL:
        parsed, model, err = run_structured_lane(lane, prompt, str(pkt), scr.parse_sc_reply)
        if parsed is not None:
            results[lane] = parsed
            models_used[lane] = model
        else:
            errors[lane] = err
            print(f"PANEL PROVIDER {lane} UNAVAILABLE (incl. fallbacks): {err}", file=sys.stderr)

    if not results:
        print("ABORT: no reviewer provider (incl. all fallbacks) produced a structured "
              "SC verdict -- no verdict recorded.", file=sys.stderr)
        sys.exit(3)

    # Defense in depth, immediately before the HMAC: never sign a verdict produced by
    # the author's own family (checked against the MODELS actually used).
    try:
        authorship.assert_panel_independent(author_provider, list(models_used.values()))
    except authorship.AuthorDeclarationError as e:
        print(f"ABORT: {e}", file=sys.stderr)
        print(f"(author={author_provider} via {author_source}; models={models_used}) "
              f"No verdict recorded -- a self-review is a rejection, never a pass.",
              file=sys.stderr)
        sys.exit(3)

    unavailable = sorted(l for l in PANEL if l not in results)
    agg = aggregate_code_panel(results)
    payload = scr.build_sc_review_payload(task_id, "code", agg)
    payload.update({
        "commit": (commit.strip() or None) if commit else None,
        "lane": f"panel-unanimous-available({'+'.join(PANEL)})",
        "author_provider": author_provider,
        "author_source": author_source,
        "providers_reviewed": sorted(results),
        "providers_unavailable": unavailable,
        "providers_unavailable_errors": {l: errors.get(l) for l in unavailable},
        "models_used": {l: models_used[l] for l in sorted(models_used)},
        "lanes_run": sorted(results),
        "second_opinion_lane": None,
        "recorded_by": os.environ.get("HERMES_PROFILE") or "robin",
        "recorded_at": int(time.time()),
        "fetched_via": "robin-ssh",
        "review_path": "deterministic-panel3-sc",
    })
    kb_status, fpath, key_id = _sign_and_record(task_id, payload)
    print(json.dumps({"ok": True, "review_type": "code", "verdict": payload.get("verdict"),
                      "lane": payload["lane"], "lanes_run": payload["lanes_run"],
                      "providers_unavailable": unavailable, "verdict_file": str(fpath),
                      "kanban": kb_status, "key_id": key_id}, ensure_ascii=False))


def _sign_and_record(task_id, payload):
    """Canonicalize + HMAC-sign a payload, write the verdict file, and record it
    via record_review_verdict with expected_review_type set (so the SC contract is
    REQUIRED, not opt-in). Shared by the SC path. Returns (kb_status, fpath, key_id)."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    key_raw = VERDICT_KEY_PATH.read_text().strip()
    try:
        key = bytes.fromhex(key_raw)
    except ValueError:
        key = key_raw.encode()
    signature = hmac.new(key, canonical.encode(), hashlib.sha256).hexdigest()
    key_id = hashlib.sha256(key).hexdigest()[:12]
    signed = {"payload": payload, "signature": signature, "key_id": key_id, "alg": "HMAC-SHA256"}
    VERDICT_DIR.mkdir(parents=True, exist_ok=True)
    commit_tag = ((payload.get("commit") or "nocommit").replace("/", "_"))[:40]
    fpath = VERDICT_DIR / f"{task_id}__{commit_tag}__{payload['recorded_at']}.json"
    fpath.write_text(json.dumps(signed, indent=2, ensure_ascii=False) + "\n")
    os.chmod(fpath, 0o600)
    kb_status = "skipped"
    try:
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_autonomy as ka
        conn = kb.connect()
        try:
            res = ka.record_review_verdict(
                conn, task_id, canonical, signature, fetched_via="robin-ssh",
                expected_review_type=payload["review_type"],
            )
            kb_status = ("recorded (%s)" % res.get("verdict")) if res.get("ok") \
                else ("rejected (%s)" % res.get("reason"))
        finally:
            conn.close()
    except Exception:
        import traceback
        kb_status = "error"
        sys.stderr.write("[review-deterministic] SC kanban writeback failed:\n" + traceback.format_exc())
    return kb_status, fpath, key_id


def run_sc_review(task_id, commit, pkt, instructions, review_type):
    """SC-gate (WIREFRAME-SC / PRD-SC) deterministic review (run-557 fix).

    Uses the shared, reviewed building block hermes_cli.review_loop.sc_review to
    prompt for the full v18 SC contract (9 fields + rubric scorecard + v18 verdict
    vocab) over INLINED evidence the tool-less juror can read, then signs a payload
    that STAMPS review_type so record_review_verdict enforces the contract
    fail-closed. The independent lane runs + signs HERE on Robin (independence
    preserved); the maker courier only inlines the evidence.
    """
    from hermes_cli.review_loop import sc_review as scr
    # The maker courier writes the inlined evidence (render_*_evidence output) to
    # evidence.md; fall back to the review-request body if absent.
    ev_file = pkt / "evidence.md"
    evidence = ev_file.read_text(errors="ignore") if ev_file.is_file() else instructions
    messages = scr.build_sc_messages(review_type, evidence)
    prompt = messages[0]["content"] + "\n\n" + messages[1]["content"]

    binp = shutil.which(LANES["gemini"])
    if not binp:
        print("ABORT: gemini lane binary not on PATH; no SC verdict recorded.", file=sys.stderr)
        sys.exit(3)
    try:
        p = subprocess.run(build_cmd("gemini", binp, prompt, str(pkt)), cwd=str(pkt),
                           stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           timeout=LANE_TIMEOUT)
    except subprocess.TimeoutExpired:
        print("ABORT: SC review lane timed out; no verdict recorded.", file=sys.stderr)
        sys.exit(3)
    out = (p.stdout or "").strip()
    parsed = scr.parse_sc_reply(out) if out else None
    if parsed is None:
        # Fail-closed: stamp review_type, no scorecard -> the contract rejects it.
        parsed = {"verdict": "rejected_for_revision",
                  "_sc_dispatch_error": f"unparseable/empty SC review output (rc={p.returncode})"}

    payload = scr.build_sc_review_payload(task_id, review_type, parsed)
    payload.update({
        "commit": (commit.strip() or None) if commit else None,
        "lane": "gemini",
        "recorded_by": os.environ.get("HERMES_PROFILE") or "robin",
        "recorded_at": int(time.time()),
        "fetched_via": "robin-ssh",
        "review_path": "deterministic-sc",
    })
    kb_status, fpath, key_id = _sign_and_record(task_id, payload)
    print(json.dumps({"ok": True, "review_type": review_type, "verdict": payload.get("verdict"),
                      "lane": "gemini", "verdict_file": str(fpath), "kanban": kb_status,
                      "key_id": key_id}, ensure_ascii=False))


def main():
    if len(sys.argv) < 5:
        print("usage: review_deterministic.py <task_id> <commit> <packet_dir> <risk>", file=sys.stderr)
        sys.exit(2)
    task_id, commit, pkt, risk = sys.argv[1:5]
    pkt = Path(pkt).expanduser()
    rr = pkt / "review-request.md"
    instructions = rr.read_text(errors="ignore") if rr.is_file() else "Review the supplied diff against the stated acceptance criteria."
    diff_name = "diff.patch" if (pkt / "diff.patch").is_file() else next(
        (p.name for p in pkt.glob("*.patch")), "diff.patch")

    # Data-sensitivity carve-out. The PRC carve-out used to reroute to claude-glm;
    # with PRC lanes retired there is no cleared automated fallback for confidential
    # customer / Teaching Labs student data (Gemini is not a cleared surface for it
    # per policy), so ABORT and require manual review rather than send it to Gemini.
    # Sensitivity carve-out keys on an EXPLICIT data-handling DECLARATION, never a topic/keyword.
    # Naming a product, customer, or domain (e.g. "Teaching Labs") is NOT sensitive data. A packet
    # opts in to the carve only by declaring that the ARTIFACT itself handles real sensitive data at
    # rest/in transit. Forward: when real data lands (Supabase/Vercel), fire on the code that handles
    # it, not on the topic. A topic-keyword regex here caused false abstentions (recal 2026-06-18).
    carve = re.search(r"^\s*handles-sensitive-data:\s*(yes|true)\b", instructions, re.I | re.M)
    if carve:
        print("UNREVIEWED (NOT a healthy pass): packet declares handles-sensitive-data:yes and no "
              "cleared automated lane exists -> routed to MANUAL review. CONFIRM this trigger is "
              "legitimate (an explicit data-handling declaration, not a topic mention) before relying "
              "on it. No verdict recorded.", file=sys.stderr)
        sys.exit(4)

    # SC-gate (WIREFRAME-SC / PRD-SC) deterministic review (run-557 fix): opt-in
    # when the packet declares review-type. Runs the contract path (review_type +
    # rubric scorecard) and returns; the code-review path below is unchanged.
    rt_m = re.search(r"^\s*review-type:\s*(wireframe|prd|general|code)\b", instructions, re.I | re.M)
    if rt_m:
        rt = rt_m.group(1).lower()
        if rt == "code":
            # Explicit CODE review keeps the author-disjoint 3-provider deterministic
            # panel but emits the structured SC contract (review_type='code' stamped,
            # expected_review_type enforced at record time via _sign_and_record).
            run_code_panel_review(task_id, commit, pkt, instructions, diff_name)
        else:
            run_sc_review(task_id, commit, pkt, instructions, rt)
        return

    # INDEPENDENT provider panel, UNANIMOUS AMONG AVAILABLE (Keith, run 2368). Every panelist
    # must be a DIFFERENT provider from the AUTHOR, so none can self-review. Each lane has a
    # same-provider fallback chain. Ships only if EVERY AVAILABLE provider PASSes; a provider
    # whose whole chain is down is SKIPPED (not a block). The only hard stop is a total outage
    # (no provider at all) -> abort with no verdict, never a silent pass.
    #
    # The panel is DERIVED from the declared author, not hardcoded. It used to be the literal
    # list ["gemini", "claude", "grok"], correct only while the author was GPT — so every
    # Claude-authored packet that fell back to this path (artifact >120k, judge unavailable,
    # below quorum) got a CLAUDE juror grading Claude's own work. Same class as run 2371, and
    # this file's own docstring still called itself "the path for Claude-authored work".
    # A hardcoded panel is a self-review waiting for the author to change (run 2267).
    try:
        author_provider, author_source = authorship.parse_author(instructions)
    except authorship.AuthorDeclarationError as e:
        print(f"ABORT: {e}", file=sys.stderr)
        print("No verdict recorded. Stamp the packet and re-send.", file=sys.stderr)
        sys.exit(3)

    LANE_LABEL = {
        "gemini": "Gemini (Google)",
        "claude": "Claude (Anthropic)",
        "grok": "Grok (xAI)",
    }
    PANEL = [
        lane for lane in ("gemini", "claude", "grok")
        if routing.provider_of(LANE_MODELS[lane]) != author_provider
    ]
    if not PANEL:
        print(f"ABORT: no author-disjoint lane available for author {author_provider!r}.",
              file=sys.stderr)
        sys.exit(3)
    print(f"deterministic panel (author={author_provider} via {author_source}): "
          f"{[LANE_LABEL[l] for l in PANEL]}", file=sys.stderr)

    lanes_run, reviews, verdicts, models_used, errors = [], {}, {}, {}, {}
    for lane in PANEL:
        out, model, err = run_lane(lane, instructions, str(pkt), diff_name)
        if out:
            lanes_run.append(lane); reviews[lane] = out
            verdicts[lane] = parse_verdict(out)
            models_used[lane] = model
        else:
            errors[lane] = err
            print(f"PANEL PROVIDER {lane} UNAVAILABLE (incl. fallbacks): {err}", file=sys.stderr)

    if not lanes_run:
        # Total review outage -- every provider AND its fallbacks are down. Cannot certify
        # anything; abort with NO verdict (never a silent pass). This is the ONLY hard stop:
        # a single unavailable provider no longer blocks the card (Keith, run 2368).
        print("ABORT: no reviewer provider (incl. all fallbacks) produced a verdict "
              "-- no verdict recorded.", file=sys.stderr)
        sys.exit(3)

    # Unanimous among the providers that were AVAILABLE.
    all_pass = all(v == "PASS" for v in verdicts.values())
    if all_pass:
        verdict = "PASS"
    else:
        vs = set(verdicts.values())
        verdict = "FAIL" if (vs & {"FAIL", "BLOCKED"}) else "CHANGES_REQUESTED"

    unavailable = [l for l in PANEL if l not in lanes_run]
    per_prov = "; ".join(
        f"{LANE_LABEL[l]} [{models_used[l]}]: {verdicts[l]}" for l in PANEL if l in lanes_run)
    if unavailable:
        per_prov += " || UNAVAILABLE: " + ", ".join(
            f"{LANE_LABEL[l]} ({errors.get(l, '?')})" for l in unavailable)
    summary = (f"Deterministic INDEPENDENT provider panel, UNANIMOUS among available "
               f"({len(lanes_run)}/{len(PANEL)} providers reviewed). Panel verdict: {verdict}. "
               + ("[DEGRADED: fewer than 3 independent providers were available] " if unavailable else "")
               + f"Per-provider -- {per_prov}. Reviewed against the stated acceptance criteria.")
    evidence = "\n\n".join(
        f"=== {LANE_LABEL[l]} [{models_used[l]}] ===\n{reviews[l][-800:]}"
        for l in PANEL if l in reviews)[-4000:]

    recorded_at = int(time.time())
    author = os.environ.get("HERMES_PROFILE") or "robin"

    # Defense in depth, immediately before the HMAC: never sign a verdict produced by the
    # author's own family. Checked against the MODELS actually used, not the lane labels, so a
    # lane whose chain fell back onto a same-family model is caught too.
    try:
        authorship.assert_panel_independent(author_provider, list(models_used.values()))
    except authorship.AuthorDeclarationError as e:
        print(f"ABORT: {e}", file=sys.stderr)
        print(f"(author={author_provider} via {author_source}; models={models_used}) "
              f"No verdict recorded — a self-review is a rejection, never a pass.", file=sys.stderr)
        sys.exit(3)

    payload = {
        "task_id": task_id,
        "verdict": verdict,
        "commit": (commit.strip() or None) if commit else None,
        "lane": f"panel-unanimous-available({'+'.join(PANEL)})",
        # Independence provenance: a verdict that does not say who it thought the author was
        # cannot be audited for self-review after the fact (runs 2267 / 2371).
        "author_provider": author_provider,
        "author_source": author_source,
        "providers_reviewed": sorted(lanes_run),
        "providers_unavailable": sorted(unavailable),
        "models_used": {l: models_used[l] for l in sorted(models_used)},
        "second_opinion_lane": None,
        "lanes_run": sorted(lanes_run),
        "summary": summary,
        "evidence": evidence,
        "risks": None,
        "what_would_change_my_mind": None,
        "recorded_by": author,
        "recorded_at": recorded_at,
        "fetched_via": "robin-ssh",
        "review_path": "deterministic-panel3",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    key_raw = VERDICT_KEY_PATH.read_text().strip()
    try:
        key = bytes.fromhex(key_raw)
    except ValueError:
        key = key_raw.encode()
    signature = hmac.new(key, canonical.encode(), hashlib.sha256).hexdigest()
    key_id = hashlib.sha256(key).hexdigest()[:12]
    signed = {"payload": payload, "signature": signature, "key_id": key_id, "alg": "HMAC-SHA256"}

    VERDICT_DIR.mkdir(parents=True, exist_ok=True)
    commit_tag = ((payload["commit"] or "nocommit").replace("/", "_"))[:40]
    fpath = VERDICT_DIR / f"{task_id}__{commit_tag}__{recorded_at}.json"
    fpath.write_text(json.dumps(signed, indent=2, ensure_ascii=False) + "\n")
    os.chmod(fpath, 0o600)

    # Durable kanban writeback -- honor the signed verdict via the supported path:
    # record_review_verdict is "the only path out of review". It re-verifies the HMAC
    # + out-of-band channel, appends verdict_recorded, and transitions the card
    # (pass -> complete; block -> back to ready with findings). Best-effort for
    # off-board test tasks; real failures are logged loudly (no silent skip).
    kb_status = "skipped"
    try:
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_autonomy as ka
        conn = kb.connect()
        try:
            res = ka.record_review_verdict(
                conn, task_id, canonical, signature, fetched_via="robin-ssh"
            )
            if res.get("ok"):
                kb_status = "recorded (%s)" % res.get("verdict")
            else:
                kb_status = "rejected (%s)" % res.get("reason")
        finally:
            conn.close()
    except Exception:
        import traceback
        kb_status = "error"
        sys.stderr.write("[review-deterministic] kanban writeback failed:\n" + traceback.format_exc())

    print(json.dumps({"ok": True, "verdict": verdict, "lane": payload["lane"],
                      "second_opinion_lane": payload["second_opinion_lane"],
                      "lanes_run": payload["lanes_run"], "verdict_file": str(fpath),
                      "kanban": kb_status, "key_id": key_id}, ensure_ascii=False))


if __name__ == "__main__":
    main()
