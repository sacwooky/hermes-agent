---
name: kanban-worker
description: Pitfalls, examples, and edge cases for Hermes Kanban workers. The lifecycle itself is auto-injected into every worker's system prompt as KANBAN_GUIDANCE (from agent/prompt_builder.py); this skill is what you load when you want deeper detail on specific scenarios.
version: 2.1.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [kanban, multi-agent, collaboration, workflow, pitfalls]
    related_skills: [kanban-orchestrator]
---

# Kanban Worker — Pitfalls and Examples

> You're seeing this skill because the Hermes Kanban dispatcher spawned you as a worker with `--skills kanban-worker` — it's loaded automatically for every dispatched worker. The **lifecycle** (6 steps: orient → work → heartbeat → block/complete) also lives in the `KANBAN_GUIDANCE` block that's auto-injected into your system prompt. This skill is the deeper detail: good handoff shapes, retry diagnostics, edge cases.

## Board naming convention

When Keith says "the board" to an agent/worker, default to that agent or workstream's **Hermes Kanban board**. Do not jump to Linear, Trello, GitHub Projects, or another external tracker unless Keith explicitly names that system. Linear is a source/import target only when the request says Linear.

## Phase 12/13: Know the orchestrator's decomposition gate

The orchestrator skill enforces two rules that workers should understand:

### Phase 12 — decomposition requires packet
Before an orchestrator fans out child cards, an approval packet (scope, AC, risk, operator approval) must exist on-file. This is a hard gate — the orchestrator blocks itself with `packet-required` if no packet is found.

### Phase 13 — unannounced work is coordination defect
Cards representing new work without an approval packet or prior operator acknowledgment are coordination defects. Repair/maintenance cards are exempt.

**What this means for workers:**
- If you're working on a card that was part of a properly-decomposed fan-out, the orchestrator already verified the packet. You can proceed normally.
- If you encounter a `packet-required` block on a sibling card, don't try to unblock it by bypassing the gate — the missing packet is a real coordination defect that needs operator input.
- If you're working as an orchestrator yourself (rare — usually a separate profile), you MUST enforce Phase 12/13 before creating child cards. See the kanban-orchestrator skill for the full enforcement procedure.

**Root cause:** Slice 4/5 incident (runs/2026-06-05-003) — work dispatched without operator visibility or signed scope.

## Worker mutation scope

A Kanban worker may be scoped to its own task and unable to mutate sibling/child cards, even when it can read and comment on them. If your reconciliation result says child cards are stale duplicates but `kanban_complete`, `kanban_archive`, or sibling mutation is refused, do **not** complete the parent if that would promote stale children. Instead:

1. Comment the exact disposition for each affected child.
2. Block the parent with a board-admin action request naming the child IDs to archive/close.
3. State whether completing the parent would incorrectly promote stale work.

Flow Manager / Jake can then perform the board-admin cleanup.

## Builder lane selection and fallback

For repo-backed code/review/QA cards, do **not** do substantial implementation directly as a generic Hermes worker. Use a supervised coding-agent lane, then you verify and report through Kanban.

### Available builder lanes

| Lane | Tool | Primary use | Fallback when |
|---|---|---|---|
| `claude` | Claude Code (`claude -p`) | Default for all code-heavy work | Primary lane |
| `codex` | Codex CLI (`codex exec`) | OpenAI-based builder; good for GPT-native code | Claude Code rate-limited or unavailable |
| `kimi` | Kimi CLI (`kimi --prompt`) | Moonshot-based builder | Both Claude and Codex unavailable |
| `opencode` | OpenCode (`opencode run`) | Free models (MiniMax-M3-free, DeepSeek-flash-free); zero-cost fallback | All paid lanes unavailable |

### Lane selection rules

1. **Default**: always try `claude` lane first.
2. **Card-specified**: if the card body or PRD explicitly names a lane (`use codex lane`, `kimi preferred for this task`), use that lane.
3. **Rate-limit fallback**: if the primary lane returns `unavailable` / `rate_limit` / `auth failed`, try the next lane in order: `claude` → `codex` → `kimi` → `opencode`.
4. **Budget-aware**: if a card has a tight budget note, prefer `opencode` (free) or `codex`/`kimi` over `claude` for large tasks.
5. **Kimi auth**: Kimi CLI uses OAuth (`kimi login`). If it fails with 401, the OAuth session expired. Run `kimi login` interactively on that host. Do NOT debug API keys — `KIMI_API_KEY` in `.env` is for Hermes provider fallback, not the CLI lane.

### Z.AI and MiniMax as Hermes conversational fallbacks (NOT CLI lanes)

Z.AI (GLM) and MiniMax are configured as **Hermes model provider fallbacks** for conversational/chat assistance only. They are NOT standalone CLI builder lanes and should NOT be used for repo-backed code implementation.

If all CLI lanes fail and the task is conversational (research, planning, documentation), the Hermes agent can fall back to these providers. This is automatic via the profile's `fallback_providers` config.

Conversational fallback chain (Hermes agent level):
- Jake local / fluxlabs-cloud / Loki: GPT-5.5 → GLM-5-turbo → MiniMax-M3
- Morgan: GLM-5-turbo

**Kimi is NOT in the conversational fallback chain.** Kimi is reserved for the CLI builder lane only.

### Claude Code lane (primary)

Invoke when all are true:
- `$HERMES_KANBAN_WORKSPACE` is a real repo/worktree/project directory
- the card involves code, tests, build, refactor, code review, or QA of repo behavior
- the card is not blocked on a human gate such as credentials, Supabase, production deploy, billing/spend, secrets, or owner approval

Required command:
```bash
~/.hermes/scripts/kanban-claude-code-lane.sh "Card $HERMES_KANBAN_TASK: <specific task + acceptance criteria>. Work only in $HERMES_KANBAN_WORKSPACE. Do not push, deploy, read secrets, change credentials, delete destructively, or spend money. Run relevant tests/builds and report changed files, commands run, verification, and risks."
```

If Claude Code is unavailable/auth-broken, try the next lane instead of blocking:
```bash
~/.hermes/scripts/kanban-codex-lane.sh "<same prompt>"
```

### Codex lane (fallback #1)

```bash
~/.hermes/scripts/kanban-codex-lane.sh "Card $HERMES_KANBAN_TASK: <specific task + acceptance criteria>. Work only in $HERMES_KANBAN_WORKSPACE. Do not push, deploy, read secrets, change credentials, delete destructively, or spend money. Run relevant tests/builds and report changed files, commands run, verification, and risks."
```

If Codex is unavailable:
```bash
~/.hermes/scripts/kanban-kimi-lane.sh "<same prompt>"
```

### Kimi lane (fallback #2)

```bash
~/.hermes/scripts/kanban-kimi-lane.sh "Card $HERMES_KANBAN_TASK: <specific task + acceptance criteria>. Work only in $HERMES_KANBAN_WORKSPACE. Do not push, deploy, read secrets, change credentials, delete destructively, or spend money. Run relevant tests/builds and report changed files, commands run, verification, and risks."
```

**Kimi CLI auth: OAuth only.**
- Kimi CLI uses OAuth (`kimi login`), not API keys.
- The `KIMI_API_KEY` in `.env` is for Hermes `kimi-coding` provider fallback only — it does NOT authenticate the Kimi CLI.
- If Kimi CLI fails with 401, the OAuth session has expired or the host is not trusted. Run `kimi login` interactively on that host to re-authenticate.
- Do NOT set `KIMI_API_KEY` for the Kimi CLI lane — it is ignored.

If Kimi is unavailable:
```bash
~/.hermes/scripts/kanban-opencode-lane.sh "<same prompt>"
```

### OpenCode lane (fallback #3)

```bash
~/.hermes/scripts/kanban-opencode-lane.sh "Card $HERMES_KANBAN_TASK: <specific task + acceptance criteria>. Work only in $HERMES_KANBAN_WORKSPACE. Do not push, deploy, read secrets, change credentials, delete destructively, or spend money. Run relevant tests/builds and report changed files, commands run, verification, and risks."
```

OpenCode supports free models out of the box (no API key required). See `references/opencode-free-models-setup.md` for installation and configuration.

**User preference**: Keith prefers direct provider integration (Z.AI, MiniMax via Hermes config) over wrapper tools when paid models are available. Use OpenCode only as a zero-cost fallback when Claude/Codex/Kimi are unavailable.

Available free models:
- `opencode/minimax-m3-free` — default for build
- `opencode/deepseek-v4-flash-free` — default for plan
- `opencode/mimo-v2.5-free`
- `opencode/nemotron-3-ultra-free`

### All lanes exhausted

If no lane is available, block mechanically with:
```
builder-lanes-unavailable: claude=<error>, codex=<error>, kimi=<error>, opencode=<error>; operator repair needed, not product input.
```

### Your job as lifecycle owner

Your job remains the lifecycle owner regardless of which lane writes the code:
1. Inspect the card first
2. Call the appropriate lane for the repo work
3. Inspect the lane output
4. Verify `git diff` and tests yourself where needed
5. Then `kanban_complete` or `kanban_block`

The coding agent is not the release manager and must not mark the card done by prose alone.

Do not use any builder lane for pure research, vault-only cleanup, or true human approval/credential gates.

## Claude Code supervised repo lane (legacy — see Builder lane selection above)

## Workspace handling

Your workspace kind determines how you should behave inside `$HERMES_KANBAN_WORKSPACE`:

| Kind | What it is | How to work |
|---|---|---|
| `scratch` | Fresh tmp dir, yours alone | Read/write freely; it gets GC'd when the task is archived. |
| `dir:<path>` | Shared persistent directory | Other runs will read what you write. Treat it like long-lived state. Path is guaranteed absolute (the kernel rejects relative paths). |
| `worktree` | Git worktree at the resolved path | If `.git` doesn't exist, run `git worktree add <path> ${HERMES_KANBAN_BRANCH:-wt/$HERMES_KANBAN_TASK}` from the main repo first, then cd and work normally. Commit work here. |

## Tenant isolation

If `$HERMES_TENANT` is set, the task belongs to a tenant namespace. When reading or writing persistent memory, prefix memory entries with the tenant so context doesn't leak across tenants:

- Good: `business-a: Acme is our biggest customer`
- Bad (leaks): `Acme is our biggest customer`

## Good summary + metadata shapes

The `kanban_complete(summary=..., metadata=...)` handoff is how downstream workers read what you did. For code/repo tasks, always include `commit_hash` (from `git rev-parse --short HEAD`) alongside changed files and diff. Patterns that work:

**Coding task:**
```python
kanban_complete(
    summary="shipped rate limiter — token bucket, keys on user_id with IP fallback, 14 tests pass",
    metadata={
        "changed_files": ["rate_limiter.py", "tests/test_rate_limiter.py"],
        "tests_run": 14,
        "tests_passed": 14,
        "commit_hash": "abc1234",
        "decisions": ["user_id primary, IP fallback for unauthenticated requests"],
    },
)
```

**Coding task that needs human review (review-required):**

For most code-changing tasks, the work isn't truly *done* until a human reviewer has eyes on it. Block instead of complete, with `reason` prefixed `review-required: ` so the dashboard surfaces the row as needing review. Drop the structured metadata (changed files, commit hash, test counts, diff/PR url) into a comment first, since `kanban_block` only carries the human-readable reason — comments are the durable annotation channel. Reviewer either approves and runs `hermes kanban unblock <id>` (which re-spawns you with the comment thread for any follow-ups) or asks for changes via another comment.

```python
import json

kanban_comment(
    body="review-required handoff:\n" + json.dumps({
        "changed_files": ["rate_limiter.py", "tests/test_rate_limiter.py"],
        "tests_run": 14,
        "tests_passed": 14,
        "commit_hash": "abc1234",
        "diff_path": "/path/to/worktree",  # or PR url if pushed
        "decisions": ["user_id primary, IP fallback for unauthenticated requests"],
    }, indent=2),
)
kanban_block(
    reason="review-required: rate limiter shipped, 14/14 tests pass — needs eyes on the user_id/IP fallback choice before merging",
)
```

Use `kanban_complete` only when the task is genuinely terminal — e.g. a one-line typo fix, a docs change with no functional consequences, or a research task where the artifact IS the writeup itself.

**Research task:**
```python
kanban_complete(
    summary="3 competing libraries reviewed; vLLM wins on throughput, SGLang on latency, Tensorrt-LLM on memory efficiency",
    metadata={
        "sources_read": 12,
        "recommendation": "vLLM",
        "benchmarks": {"vllm": 1.0, "sglang": 0.87, "trtllm": 0.72},
    },
)
```

**Review task:**
```python
kanban_complete(
    summary="reviewed PR #123; 2 blocking issues found (SQL injection in /search, missing CSRF on /settings)",
    metadata={
        "pr_number": 123,
        "findings": [
            {"severity": "critical", "file": "api/search.py", "line": 42, "issue": "raw SQL concat"},
            {"severity": "high", "file": "api/settings.py", "issue": "missing CSRF middleware"},
        ],
        "approved": False,
    },
)
```

Shape `metadata` so downstream parsers (reviewers, aggregators, schedulers) can use it without re-reading your prose.

## Claiming cards you actually created

If your run produced new kanban tasks (via `kanban_create`), pass the ids in `created_cards` on `kanban_complete`. The kernel verifies each id exists and was created by your profile; any phantom id blocks the completion with an error listing what went wrong, and the rejected attempt is permanently recorded on the task's event log. **Only list ids you captured from a successful `kanban_create` return value — never invent ids from prose, never paste ids from earlier runs, never claim cards another worker created.**

```python
# GOOD — capture return values, then claim them.
c1 = kanban_create(title="remediate SQL injection", assignee="security-worker")
c2 = kanban_create(title="fix CSRF middleware", assignee="web-worker")

kanban_complete(
    summary="Review done; spawned remediations for both findings.",
    metadata={"pr_number": 123, "approved": False},
    created_cards=[c1["task_id"], c2["task_id"]],
)
```

```python
# BAD — claiming ids you don't have captured return values for.
kanban_complete(
    summary="Created remediation cards t_a1b2c3d4, t_deadbeef",  # hallucinated
    created_cards=["t_a1b2c3d4", "t_deadbeef"],                   # → gate rejects
)
```

If a `kanban_create` call fails (exception, tool_error), the card was NOT created — do not include a phantom id for it. Retry the create, or omit the id and mention the failure in your summary. The prose-scan pass also catches `t_<hex>` references in your free-form summary that don't resolve; these don't block the completion but show up as advisory warnings on the task in the dashboard.

## Block reasons that get answered fast

Bad: `"stuck"` — the human has no context.

Good: one sentence naming the specific decision you need. Leave longer context as a comment instead.

```python
kanban_comment(
    task_id=os.environ["HERMES_KANBAN_TASK"],
    body="Full context: I have user IPs from Cloudflare headers but some users are behind NATs with thousands of peers. Keying on IP alone causes false positives.",
)
kanban_block(reason="Rate limit key choice: IP (simple, NAT-unsafe) or user_id (requires auth, skips anonymous endpoints)?")
```

The block message is what appears in the dashboard / gateway notifier. The comment is the deeper context a human reads when they open the task.

## Fixing reviewer-blocked cards (operator pattern)

When a reviewer blocks a card with specific findings, the operator/Jake fix cycle is:

1. **Read the reviewer's comment** — `hermes kanban show <id>` and read every finding. Do not guess what was wrong; the comment is the authoritative spec.
2. **Fix each finding at the file level** — Create/patch/refresh the exact artifacts the reviewer named. If the reviewer cited line numbers in a specific file, update that file. If the reviewer said "doc X is missing," create doc X.
3. **Run verification** — At minimum: secret scan (grep for API key / token / PEM patterns over changed files) and any project-specific lint. Record zero findings as evidence.
4. **Comment before unblocking** — Write a structured comment on the card listing: (a) what was fixed, (b) which files changed, (c) verification results. This becomes the audit trail for the re-review.
5. **Unblock** — `hermes kanban unblock <id>`. The reviewer re-runs with the comment thread for context.

**Do not** unblock without commenting the fix evidence. The reviewer spawned a fresh context with no memory of your fix — the comment is their only signal that the findings were addressed.

### Source-controlled evidence files (pitfall)

Kanban card comments are ephemeral context. When a maintainer/ops card produces host inventories, install evidence, or verification results, **write a source-controlled file** in the vault/project docs — not just a Kanban comment. If the only record of "CLI parity completed on Loki" is a Kanban comment, the gap-matrix and evidence-file index become stale on the next review cycle.

Pattern: write evidence files under the project's ops/docs path (e.g. `ops/fleet-mcp-gateway/hosts/<host>-cli-parity-e2.md`), then reference them from index docs (`hosts/README.md`, `gap-matrix.md`). Kanban comments supplement; source files are the audit trail.

## Heartbeats worth sending

Good heartbeats name progress: `"epoch 12/50, loss 0.31"`, `"scanned 1.2M/2.4M rows"`, `"uploaded 47/120 videos"`.

Bad heartbeats: `"still working"`, empty notes, sub-second intervals. Every few minutes max; skip entirely for tasks under ~2 minutes.

## Retry scenarios

If you open the task and `kanban_show` returns `runs: [...]` with one or more closed runs, you're a retry. The prior runs' `outcome` / `summary` / `error` tell you what didn't work. Don't repeat that path. Typical retry diagnostics:

- `outcome: "timed_out"` — the previous attempt hit `max_runtime_seconds`. You may need to chunk the work or shorten it.
- `outcome: "crashed"` — OOM or segfault. Reduce memory footprint.
- `outcome: "spawn_failed"` + `error: "..."` — usually a profile config issue (missing credential, bad PATH). Ask the human via `kanban_block` instead of retrying blindly.
- `outcome: "reclaimed"` + `summary: "task archived..."` — operator archived the task out from under the previous run; you probably shouldn't be running at all, check status carefully.
- `outcome: "blocked"` — a previous attempt blocked; the unblock comment should be in the thread by now.

## Notification routing

You can configure the gateway to receive cross-profile Kanban task notifications by adding `notification_sources` to `~/.hermes/config.yaml`.
- `notification_sources: ['*']` accepts subscriptions from all profiles.
- `notification_sources: ['default', 'zilor-ppt']` or `"default,zilor-ppt"` restricts subscriptions to specified profiles.
- Omitting the key keeps the default behavior (profile isolation).

## Do NOT

- Call `delegate_task` as a substitute for `kanban_create`. `delegate_task` is for short reasoning subtasks inside YOUR run; `kanban_create` is for cross-agent handoffs that outlive one API loop.
- Call `clarify` to ask the human a question. You are running headless — there is no live user to answer. The call will time out (default ~120s) and the task will sit silently in `running` with no signal that it needs input. Use `kanban_comment` (context) + `kanban_block(reason=...)` (decision needed) instead — the task surfaces on the board as blocked, the operator sees it, unblocks with their answer in a comment, and you respawn with the thread.
- Modify files outside `$HERMES_KANBAN_WORKSPACE` unless the task body says to.
- Create follow-up tasks assigned to yourself — assign to the right specialist.
- Complete a task you didn't actually finish. Block it instead.

## Pitfalls

**Bash heredoc and quote escaping loops.** When writing multi-line scripts for remote execution, avoid bash heredocs (`cat <<'EOF'`). They break on nested quotes, variable expansion, and SSH command wrapping. Use `write_file` locally + `scp` + `ssh host 'bash script'` instead. See `references/bash-heredoc-quoting-pitfalls.md`.

**Task state can change between dispatch and your startup.** Between when the dispatcher claimed and when your process actually booted, the task may have been blocked, reassigned, or archived. Always `kanban_show` first. If it reports `blocked` or `archived`, stop — you shouldn't be running.

**Workspace may have stale artifacts.** Especially `dir:` and `worktree` workspaces can have files from previous runs. Read the comment thread — it usually explains why you're running again and what state the workspace is in.

**Don't rely on the CLI when the guidance is available.** The `kanban_*` tools work across all terminal backends (Docker, Modal, SSH). `hermes kanban <verb>` from your terminal tool will fail in containerized backends because the CLI isn't installed there. When in doubt, use the tool.

## CLI fallback (for scripting)

Every tool has a CLI equivalent for human operators and scripts:
- `kanban_show` ↔ `hermes kanban show <id> --json`
- `kanban_complete` ↔ `hermes kanban complete <id> --summary "..." --metadata '{...}'`
- `kanban_block` ↔ `hermes kanban block <id> "reason"`
- `kanban_create` ↔ `hermes kanban create "title" --assignee <profile> [--parent <id>]`
- etc.

Use the tools from inside an agent; the CLI exists for the human at the terminal.
