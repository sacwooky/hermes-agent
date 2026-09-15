#!/usr/bin/env bash
# send-review.sh — courier helper. Sends a review packet to Robin and returns
# her signed verdict. Runs ON Robin; a remote agent (e.g. Jake) calls it over ssh.
#
#   # 1) copy the packet to Robin (diff travels as FILES, never as a string):
#   scp -r <local-packet-dir> robin:/tmp/
#   # 2) ask Robin to review it and print the signed verdict:
#   ssh robin '~/.hermes/scripts/send-review.sh <task_id> <commit> /tmp/<packet> <risk> [lane_note]'
#
# Because the diff is read from the packet on disk, it NEVER appears in a command
# line or JSON body — there is no shell/JSON escaping to get wrong, and no need
# for execute_code. Args are simple tokens (task id, sha, path, LOW|MEDIUM|HIGH).
#
# Exit codes: 0 = signed verdict returned (signature VALID); 2 = usage/setup
# error; 3 = verdict produced but signature INVALID; 4 = timed out waiting.
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/bin:$PATH"

# Load 9router credentials (NINEROUTER_KEY) so the Fusion jury lanes can authenticate over ssh.
# Without this the jurors fail with "missing NINEROUTER_KEY" -> silent 0/3 -> deterministic-only
# fallback (fleet-wide quiet degradation; root-caused run 512, fixed run 513). .env is clean KEY=VALUE.
set -a; [ -f "$HOME/.hermes/.env" ] && . "$HOME/.hermes/.env"; set +a

ROBIN_URL="${ROBIN_URL:-http://100.101.182.4:8787}"
VERDICT_DIR="$HOME/.hermes/verdicts"
KEY_FILE="$HOME/.hermes/credentials/robin-verdict-key"
POLL_SECS=6
POLL_MAX=70   # ~7 min ceiling

die(){ echo "send-review: $*" >&2; exit "${2:-2}"; }

[ $# -ge 4 ] || die "usage: send-review.sh <task_id> <commit> <packet_dir> <risk:LOW|MEDIUM|HIGH> [lane_note]"
TASK="$1"; COMMIT="$2"; PKT_IN="$3"; RISK="$(printf '%s' "$4" | tr '[:lower:]' '[:upper:]')"; LANE_NOTE="${5:-}"

# Single-flight per exact task+commit. Reclaims/retries must not start overlapping
# Fusion/deterministic runs for the same immutable artifact.
LOCK_DIR="$HOME/.hermes/state/review-locks"
mkdir -p "$LOCK_DIR"
LOCK_KEY="$(printf '%s__%s' "$TASK" "$COMMIT" | sha256sum | cut -c1-24)"
exec 9>"$LOCK_DIR/$LOCK_KEY.lock"
if ! flock -n 9; then
  die "review already running for task=$TASK commit=$COMMIT; queue/retry instead of duplicating" 4
fi

# Normalize packet dir: strip a leading host prefix like "robin:" if present.
PKT="${PKT_IN#*:}"; case "$PKT" in /*) : ;; *) PKT="/$PKT" ;; esac
[ -d "$PKT" ] || die "packet dir not found on Robin: $PKT (scp it to robin:/tmp/ first)"
[ -f "$PKT/diff.patch" ] || echo "send-review: warning — $PKT/diff.patch not found" >&2
case "$RISK" in LOW|MEDIUM|HIGH) : ;; *) die "risk must be LOW|MEDIUM|HIGH (got '$4')";; esac

# Author stamp is MANDATORY and is checked HERE, before any lane spends a token — the review
# paths fail closed on it anyway, but failing at the maker turns a confusing mid-run abort into
# an instant, actionable error. The author family decides which providers are BARRED from
# grading the work; guessing it wrong seats the author's own family (runs 2267, 2371).
python3 - "$PKT/review-request.md" <<'PYCHECK' || die "packet does not declare its author (see above)" 2
import sys
sys.path.insert(0, "/home/fluxlabs/.hermes/scripts")
from fusionkit import authorship
try:
    text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
except OSError as e:
    print(f"send-review: cannot read review-request.md: {e}", file=sys.stderr); sys.exit(1)
try:
    fam, src = authorship.parse_author(text)
except authorship.AuthorDeclarationError as e:
    print(f"send-review: {e}", file=sys.stderr); sys.exit(1)
print(f"send-review: author={fam} ({src})", file=sys.stderr)
PYCHECK

# Default lane instruction per policy (Gemini independent lanes; PRC retired).
# Lane selection per the Fleet Model & Review Toolchain Policy (review-model
# breakdown by review type/author). Claude-authored code is reviewed by
# INDEPENDENT lanes only -- Claude never reviews its own output. The review
# orchestrator reliably runs "primary + one second opinion", so use that shape.
if [ -z "$LANE_NOTE" ]; then
  # INDEPENDENT provider panel, UNANIMOUS AMONG AVAILABLE (runs 2367-2368). The fleet
  # coder is the Hermes builder on GPT (OpenAI); every panelist is a DIFFERENT, non-author
  # provider so none can self-review: Gemini (Google) + Opus (Anthropic) + Grok (xAI). Each
  # has a SAME-PROVIDER fallback chain; a provider whose whole chain is down is SKIPPED, not
  # blocked. Ships only if every AVAILABLE provider PASSes. Handled by the panel path below.
  LANE_NOTE="INDEPENDENT provider review panel (UNANIMOUS among available; per-provider same-provider fallbacks; a fully-unavailable provider is skipped, NOT blocked) -- Gemini 3.1-pro (Google) + Opus (Anthropic) + Grok (xAI). All independent of the GPT author; no self-review. PRC lanes retired."
fi

# Work takes the DETERMINISTIC review path: the LLM review driver is unreliable
# (ignores lane guidance, sometimes does not call the review tool at all). So we
# run the mandated INDEPENDENT panel directly -- gemini-3.1-pro + claude-opus-4-8
# + grok-4.6, UNANIMOUS (all must PASS) -- and record the same signed verdict.
# No driver, no lane discretion. PRC review models (GLM/Kimi/MiniMax) RETIRED.
# Confidential/student-data carve-out ABORTS (no cleared lane) inside the script.
# ADD-ON C v2: L2 Robin Fusion is the DEFAULT review path for Claude-authored work
# (operator 2026-06-18). review-fusion.py falls back to the deterministic single lane when
# Fusion cannot render a verdict for INFRA reasons (below-quorum/transient, oversize >120k,
# judge unavailable); it queues on POLICY rejects (degraded-high, independence) so safety holds.
# SC-gate (WIREFRAME-SC / PRD-SC) reviews ALWAYS take the deterministic SC contract
# path (review-deterministic.py review-type mode) -- never Fusion/freeform. Dispatch
# half of the run-557 fix: the verdict carries review_type + a rubric scorecard and is
# enforced fail-closed at record time.
if grep -qiE "^review-type:[[:space:]]*(wireframe|prd|general|code)\b" "$PKT/review-request.md" 2>/dev/null; then
  echo "send-review: explicit review-type -> deterministic structured contract path" >&2
  exec python3 "$HOME/.hermes/scripts/review-deterministic.py" "$TASK" "$COMMIT" "$PKT" "$RISK"
fi
# Opt OUT with `review-mode: deterministic` (forces the single deterministic lane).
if grep -qiE "^review-mode:[[:space:]]*deterministic" "$PKT/review-request.md" 2>/dev/null; then
  echo "send-review: review-mode=deterministic -> single deterministic independent lane (opt-out)" >&2
  exec python3 "$HOME/.hermes/scripts/review-deterministic.py" "$TASK" "$COMMIT" "$PKT" "$RISK"
fi
# L2 Robin Fusion is the DEFAULT path for ALL code reviews (run 2371). Fusion derives the author
# family per packet (a declared "author-provider:" / "claude-code-authored:" stamp wins; otherwise
# the configured default in ~/.hermes/review-author-default) and DROPS every seat whose family
# matches the author -- so independence is enforced for Claude-authored AND GPT-authored work.
#
# It used to fire ONLY on "claude-code-authored: yes". Everything else -- notably the kanban GPT
# builder, which stamps no author at all -- fell through to the webui LLM path below, which
# reviews with Robin host-default cx/gpt-5.5-review (GPT) and runs NO independence check. With the
# builder pool now on GPT that was a live GPT-reviews-GPT hole, armed for the next unstamped
# packet. Routing every code review through Fusion closes it structurally rather than by habit.
echo "send-review: -> L2 Robin Fusion (author-aware independent panel; deterministic fallback on infra-unavailable)" >&2
exec python3 "$HOME/.hermes/scripts/review-fusion.py" "$TASK" "$COMMIT" "$PKT" "$RISK"
# The webui LLM path below is now UNREACHABLE for code review (the exec above always fires). It is
# left in place only as a break-glass manual lane. It has NO independence guard -- do not restore
# it as a default without one.

# Snapshot existing verdicts so we can detect THIS run's new file.
BEFORE="$(ls -t "$VERDICT_DIR/${TASK}__"* 2>/dev/null | head -1 || true)"

# 1) Open a review session.
SID="$(curl -fsS "$ROBIN_URL/api/session/new" -H 'Content-Type: application/json' \
        -d '{"workspace":"/home/fluxlabs/workspace"}' \
      | python3 -c 'import sys,json;print(json.load(sys.stdin)["session"]["session_id"])')" \
  || die "could not open a session on Robin's webui ($ROBIN_URL)"

# 2) Pin the focused review toolset (safe default).
curl -fsS "$ROBIN_URL/api/session/toolsets" -H 'Content-Type: application/json' \
     -d "{\"session_id\":\"$SID\",\"toolsets\":[\"review\"]}" >/dev/null || true

# 3) Build the request body with python (no diff content; zero shell escaping) and send.
REQ="$(mktemp)"; trap 'rm -f "$REQ"' EXIT
python3 - "$SID" "$TASK" "$COMMIT" "$PKT" "$RISK" "$LANE_NOTE" > "$REQ" <<'PY'
import sys, json
sid, task, commit, pkt, risk, lane = sys.argv[1:7]
msg = (f"Initiate review for task {task}. Commit {commit}. "
       f"Artifact packet at robin:{pkt}. {risk} risk — {lane}")
print(json.dumps({"session_id": sid, "message": msg, "workspace": "/home/fluxlabs/workspace"}))
PY
curl -fsS "$ROBIN_URL/api/chat/start" -H 'Content-Type: application/json' --data @"$REQ" >/dev/null \
  || die "failed to send review request"
echo "send-review: requested $TASK (session $SID); waiting for Robin's signed verdict…" >&2

# 4) Wait for a NEW signed verdict file for this task.
NEW=""
for _ in $(seq 1 "$POLL_MAX"); do
  CUR="$(ls -t "$VERDICT_DIR/${TASK}__"* 2>/dev/null | head -1 || true)"
  if [ -n "$CUR" ] && [ "$CUR" != "$BEFORE" ]; then NEW="$CUR"; break; fi
  sleep "$POLL_SECS"
done
[ -n "$NEW" ] || die "no verdict after ~$((POLL_SECS*POLL_MAX))s — inspect Robin's webui session $SID" 4

# 5) Verify the signature (key is local on Robin) and print the verdict to stdout.
python3 - "$NEW" "$KEY_FILE" <<'PY'
import sys, json, hmac, hashlib
f, keyf = sys.argv[1], sys.argv[2]
d = json.load(open(f)); p = d["payload"]
try:
    key = bytes.fromhex(open(keyf).read().strip())
except Exception:
    key = open(keyf).read().strip().encode()
canon = json.dumps(p, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
ok = hmac.new(key, canon.encode(), hashlib.sha256).hexdigest() == d["signature"]
print("==== ROBIN VERDICT ====")
print(f"task:       {p['task_id']}")
print(f"verdict:    {p['verdict']}")
print(f"lane:       {p['lane']}   second_opinion: {p.get('second_opinion_lane')}   lanes_run: {p.get('lanes_run')}")
print(f"commit:     {p.get('commit')}")
print(f"signature:  {'VALID' if ok else 'INVALID'}  (key_id {d.get('key_id')})")
print(f"fetched_via:{p.get('fetched_via')}")
print(f"artifact:   {f}")
print()
print(p.get("summary", ""))
if p.get("risks"): print("\nRisks: " + p["risks"])
if p.get("what_would_change_my_mind"): print("\nWhat would change my mind: " + p["what_would_change_my_mind"])
sys.exit(0 if ok else 3)
PY
