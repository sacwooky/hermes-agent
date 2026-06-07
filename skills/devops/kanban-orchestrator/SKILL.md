---
name: kanban-orchestrator
description: Decomposition playbook + anti-temptation rules for an orchestrator profile routing work through Kanban. The "don't do the work yourself" rule and the basic lifecycle are auto-injected into every kanban worker's system prompt; this skill is the deeper playbook when you're specifically playing the orchestrator role.
version: 3.1.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [kanban, multi-agent, orchestration, routing]
    references:
      - references/process-lanes-and-blockers.md
      - references/phase-12-13-intake.md
    related_skills: [kanban-worker]
---

# Kanban Orchestrator — Decomposition Playbook

## Recently learned references

- `references/reviewer-mustfix-repair-synthesis.md` — when reviewer resolver cards block with concrete must-fix language, synthesize one idempotent builder repair card instead of resolver-of-resolver chains while preserving human gates.
- `references/fleet-change-board-record-discipline.md` — when applying a fix across Jake/Morgan/Loki or other fleet boards, create/update visible story records per affected board, not only a central implementation card; if a direct patch came first, backfill records explicitly.

> The **core worker lifecycle** (including the `kanban_create` fan-out pattern and the "decompose, don't execute" rule) is auto-injected into every kanban process via the `KANBAN_GUIDANCE` system-prompt block. This skill is the deeper playbook when you're an orchestrator profile whose whole job is routing.

## Profiles are user-configured — not a fixed roster

Hermes setups vary widely. Some users run a single profile that does everything; some run a small fleet (`docker-worker`, `cron-worker`); some run a curated specialist team they've named themselves. There is **no default specialist roster** — the orchestrator skill does not know what profiles exist on this machine.

Before fanning out, you must ground the decomposition in the profiles that actually exist. The dispatcher silently fails to spawn unknown assignee names — it doesn't autocorrect, doesn't suggest, doesn't fall back. So a card assigned to `researcher` on a setup that only has `docker-worker` just sits in `ready` forever.

**Step 0: discover available profiles before planning.**

Use one of these:

- `hermes profile list` — prints the table of profiles configured on this machine. Run it through your terminal tool if you have one; otherwise ask the user.
- `kanban_list(assignee="<some-name>")` — sanity-check a single name. Returns an empty list (rather than an error) for an unknown assignee, so this only confirms a name you're already considering.
- **Just ask the user.** "What profiles do you have set up?" is a fine first turn when the goal needs more than one specialist.

Cache the result in your working memory for the rest of the conversation. Re-asking every turn wastes a tool call.

## When to use the board (vs. just doing the work)

Create Kanban tasks when any of these are true:

1. **Multiple specialists are needed.** Research + analysis + writing is three profiles.
2. **The work should survive a crash or restart.** Long-running, recurring, or important.
3. **The user might want to interject.** Human-in-the-loop at any step.
4. **Multiple subtasks can run in parallel.** Fan-out for speed.
5. **Review / iteration is expected.** A reviewer profile loops on drafter output.
6. **The audit trail matters.** Board rows persist in SQLite forever.

If *none* of those apply — it's a small one-shot reasoning task — use `delegate_task` instead or answer the user directly.

## UX/UI cards require wireframes + Tailscale review link

UX/UI changes are a special case. Before a UX/UI feature card leaves `triage` /
`todo` and gets dispatched to a builder, the card body MUST contain:

1. A `wireframe_link:` line pointing to an **active Tailscale-accessible
   review URL** (typically a `*.ts.net` MagicDNS host).
2. A saved artifact path under the project's `wireframes/` or `mockups/`
   folder, dated and versioned (e.g. `wireframes/2026-06-05-settings-dark-mode-v1.html`).

Why both: a wireframe saved only on disk gets stale fast and is hard for
Keith to review from a phone; a link with no saved artifact gets lost when
the link changes. The pair is the durable record.

How to enforce it:

- The card body for every UX/UI feature should be seeded with the standard
  stub from `wireframe_guard.card_body_template()` (see
  `scripts/wireframe_guard.py` in this skill). Edit the stub to fill in
  real values; do not delete the `wireframe_link:` line even if you are
  pasting the value elsewhere.
- Before promoting a UX/UI card from `triage` / `todo` to `ready`, run
  `python3 scripts/wireframe_guard.py --title <t> --body <b>`. The script
  returns a verdict JSON: `{"pass": true|false, "reason": ..., ...}`. A
  verdict with `pass: false` means the orchestrator should `kanban_block`
  the card with the reason and ask the user for the wireframe / link.
- The `verify_tailscale_link` helper inside the script validates the URL
  shape (scheme, host format, IP-literal ban) without doing a network probe
  by default. Pass `--probe` to additionally open a TCP connection. Do not
  pass `--probe` in CI / nightly loops; it would touch the live tailnet on
  every card.
- Detection is keyword-based and conservative. Cards whose body contains
  "no UX change", "backend only", "uxui: false", etc. are exempt; see
  `_OPT_OUT_PHRASES` in the script. If a card clearly IS UX/UI but the
  detector misses it (e.g. obscure product name), tag the card with
  `uxui: true` in the body and the guard will treat it as UX/UI.

The guard is a tool, not a gate in the engine. The orchestrator must call
it explicitly. Build it into your default workstream: when fanning out
features, run the guard over every new card's title+body and block any
UX/UI card that does not have a valid `wireframe_link` line. This is the
"right" answer to "where do wireframes live in the board?" — the
canonical record is the `wireframe_link:` line in the card body, and the
saved file under `wireframes/` / `mockups/` is the durable artifact.

Full card body template (use this verbatim, then edit):

```
## UX/UI gate (required)
wireframe_link: <tailscale-url>     # e.g. http://fluxlabs.tail6d84e.ts.net:3000/wireframes/2026-06-05-page-v1.html
Saved artifact: wireframes/<date>-<scope>-v<n>.html   # or mockups/<date>-<scope>-v<n>.png

## Affected pages
- <list every page/screen this change touches>

## Acceptance criteria
- [ ] Wireframe/mockup committed under wireframes/ or mockups/
- [ ] Active Tailscale review link posted (fluxlabs.tail*.ts.net or similar)
- [ ] Reviewer signs off on the visual direction before build approval
```

## The anti-temptation rules

Your job description says "route, don't execute." The rules that enforce that:

- **Do not execute the work yourself.** Your restricted toolset usually doesn't even include terminal/file/code/web for implementation. If you find yourself "just fixing this quickly" — stop and create a task for the right specialist.
- **For any concrete task, create a Kanban task and assign it.** Every single time.
- **Split multi-lane requests before creating cards.** A user prompt can contain several independent workstreams. Extract those lanes first, then create one card per lane instead of bundling unrelated work into a single implementer card.
- **Run independent lanes in parallel.** If two cards do not need each other's output, leave them unlinked so the dispatcher can fan them out. Link only true data dependencies.
- **Never create dependent work as independent ready cards.** If a card must wait for another card, pass `parents=[...]` in the original `kanban_create` call. Do not create it first and link it later, and do not rely on prose like "wait for T1" inside the body.
- **If no specialist fits the available profiles, ask the user which profile to create or which existing profile to use.** Do not invent profile names; the dispatcher will silently drop unknown assignees.
- **Decompose, route, and summarize — that's the whole job.**

## Phase 12/13: Decomposition gate and coordination discipline

> See `phase-12-13-intake.md` for the full specification.

### Phase 12 — decomposition requires packet

Before decomposing a goal into child cards, the orchestrator MUST verify an approval packet exists for the scope being decomposed. An approval packet is a written document covering: scope statement, acceptance criteria, risk assessment, and operator approval.

**What qualifies as a packet:**
- A markdown file in `docs/prds/`, `docs/plans/`, or a project's `docs/` directory
- For small scopes, the kanban card body itself IF it contains all four elements (scope, AC, risk, approval)

**Grandfathered packets** (pre-Phase 12, accepted without additional evidence):
- Any `docs/prds/APPROVAL-*.md` committed before 2026-06-06
- `conductor-vault-v0-rc1-release-note-2026-06-03.md`

**Enforcement — hard gate:**
1. Before calling `kanban_create` for a fan-out, check for an approval packet
2. If no packet is found, STOP — do not create children
3. Block the orchestrator card: `packet-required: decomposition of "<goal>" has no approval packet on-file`

### Phase 13 — unannounced work is coordination defect

Kanban cards that introduce new deliverables, features, or changes without a corresponding approval packet or prior operator acknowledgment constitute a coordination defect.

**Not affected:** repair cards for existing work, clarifying docs, maintenance (deps, lint, drift).

**Detection:** Before fan-out, check: (a) packet exists? (b) work discussed in prior session? (c) existing epic/story covers it? If all negative, the work is unannounced.

**Response:** Do NOT create children. Comment on the card explaining the gap. Block with clear reason naming what's missing. Notify operator via block notification.

**Root cause:** Slice 4/5 incident (runs/2026-06-05-003) — work dispatched without operator visibility or signed scope.

## Decomposition playbook

### Step 1 — Understand the goal

Ask clarifying questions if the goal is ambiguous. Cheap to ask; expensive to spawn the wrong fleet.

### Step 1.5 — Verify approval packet (Phase 12/13 gate)

Before proceeding to decomposition (Step 2), verify an approval packet exists for the goal:

1. Check the card body for packet reference (`docs/prds/APPROVAL-*`, `docs/plans/*`, or inline scope/AC/risk/approval)
2. Search the repo/vault for matching packet files
3. Check session history for prior operator approval of this scope
4. If no packet found → **STOP** and block the card with `packet-required` reason

This is a hard gate. Do not skip it even for "obvious" work. The Slice 4/5 incident showed that skipping this gate produces coordination defects.

### Step 2 — Sketch the task graph

Before creating anything, draft the graph out loud (in your response to the user). Treat every concrete workstream as a candidate card:

1. Extract the lanes from the request.
2. Map each lane to one of the profiles you discovered in Step 0. If a lane doesn't fit any existing profile, ask the user which to use or create.
3. Decide whether each lane is independent or gated by another lane.
4. Create independent lanes as parallel cards with no parent links.
5. Create synthesis/review/integration cards with parent links to the lanes they depend on. A child created with unfinished parents starts in `todo`; the dispatcher promotes it to `ready` only after every parent is done.

Examples of prompts that should fan out (using placeholder profile names — substitute whatever exists on the user's setup):

- "Build an app" → one card to a design-oriented profile for product/UI direction, one or two cards to engineering profiles for implementation, plus a later integration/review card if the user has a reviewer profile.
- "Fix blockers and check model variants" → one implementation card for the blocker fixes plus one discovery/research card for config/source verification. A final reviewer card can depend on both.
- "Research docs and implement" → a docs-research card can run in parallel with a codebase-discovery card; implementation waits only if it truly needs those findings.
- "Analyze this screenshot and find the related code" → one card to a vision-capable profile for the visual analysis while another searches the codebase.

Words like "also," "finally," or "and" do not automatically imply a dependency. They often mean "make sure this is covered before reporting back." Only link tasks when one card cannot start until another card's output exists.

Show the graph to the user before creating cards. Let them correct it — including which actual profile name should own each lane.

### Step 3 — Create tasks and link

Use the profile names from Step 0. The example below uses placeholders `<profile-A>`, `<profile-B>`, `<profile-C>` — replace them with what the user actually has.

```python
t1 = kanban_create(
    title="research: Postgres cost vs current",
    assignee="<profile-A>",  # whichever profile handles research on this setup
    body="Compare estimated infrastructure costs, migration costs, and ongoing ops costs over a 3-year window. Sources: AWS/GCP pricing, team time estimates, current Postgres bills from peers.",
    tenant=os.environ.get("HERMES_TENANT"),
)["task_id"]

t2 = kanban_create(
    title="research: Postgres performance vs current",
    assignee="<profile-A>",  # same profile, run in parallel
    body="Compare query latency, throughput, and scaling characteristics at our expected data volume (~500GB, 10k QPS peak). Sources: benchmark papers, public case studies, pgbench results if easy.",
)["task_id"]

t3 = kanban_create(
    title="synthesize migration recommendation",
    assignee="<profile-B>",  # whichever profile does synthesis/analysis
    body="Read the findings from T1 (cost) and T2 (performance). Produce a 1-page recommendation with explicit trade-offs and a go/no-go call.",
    parents=[t1, t2],
)["task_id"]

t4 = kanban_create(
    title="draft decision memo",
    assignee="<profile-C>",  # whichever profile drafts user-facing prose
    body="Turn the analyst's recommendation into a 2-page memo for the CTO. Match the tone of previous decision memos in the team's knowledge base.",
    parents=[t3],
)["task_id"]
```

`parents=[...]` gates promotion — children stay in `todo` until every parent reaches `done`, then auto-promote to `ready`. No manual coordination needed; the dispatcher and dependency engine handle it.

If the task graph has dependencies, create the parent cards first, capture their returned ids, and include those ids in the child card's `parents` list during the child `kanban_create` call. Avoid creating all cards in parallel and linking them afterward; that creates a window where the dispatcher can claim a child before its inputs exist.

#### UX/UI feature card body

UX/UI feature cards are the one case where the orchestrator must populate
specific fields in the card body itself (not just the title) so the
`wireframe_guard` can validate it later. Seed the body with the standard
stub (see "UX/UI cards require wireframes" above) and run
`wireframe_guard.check_card(...)` over the resulting `(title, body)` pair
before promoting. Example:

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "skills" / "devops" / "kanban-orchestrator" / "scripts"))
from wireframe_guard import check_card, card_body_template  # type: ignore

body = card_body_template() + "\n## Scope\nRedesign settings page so dark mode is the first toggle.\n"
verdict = check_card({"title": "Add dark mode toggle to settings", "body": body})
assert verdict["pass"], verdict["reason"]  # wireframe_link is a placeholder; replace first

t = kanban_create(
    title="Add dark mode toggle to settings",
    assignee="<builder-profile>",
    body=body,
)
```

In practice you replace the `<tailscale-url>` placeholder with the real
Tailscale link and re-run the guard before promoting the card to `ready`.

### Step 4 — Complete your own task

If you were spawned as a task yourself (e.g. a planner profile was assigned `T0: "investigate Postgres migration"`), mark it done with a summary of what you created:

```python
kanban_complete(
    summary="decomposed into T1-T4: 2 research lanes in parallel, 1 synthesis on their outputs, 1 prose draft on the recommendation",
    metadata={
        "task_graph": {
            "T1": {"assignee": "<profile-A>", "parents": []},
            "T2": {"assignee": "<profile-A>", "parents": []},
            "T3": {"assignee": "<profile-B>", "parents": ["T1", "T2"]},
            "T4": {"assignee": "<profile-C>", "parents": ["T3"]},
        },
    },
)
```

### Step 5 — Report back to the user

Tell them what you created in plain prose, naming the actual profiles you used:

> I've queued 4 tasks:
> - **T1** (`<profile-A>`): cost comparison
> - **T2** (`<profile-A>`): performance comparison, in parallel with T1
> - **T3** (`<profile-B>`): synthesizes T1 + T2 into a recommendation
> - **T4** (`<profile-C>`): turns T3 into a CTO memo
>
> The dispatcher will pick up T1 and T2 now. T3 starts when both finish. You'll get a gateway ping when T4 completes. Use the dashboard or `hermes kanban tail <id>` to follow along.

## Common patterns

**Durable conductor script:** When the user wants the swarm to "follow the process all the way through" instead of manually adding work to each profile, create a repo-local script that queues an idempotent gated graph (preflight → builder → QA/reviewer → acceptance → knowledge writeback). Keep dry-run as the default, require `--execute` to create cards, and use `--run-loop` only after explicit opt-in. See `references/durable-conductor-scripts.md` for the script shape, safety rules, and verification checklist.

**Cloud Kanban handoff:** When moving active boards from a local workstation to a cloud host, freeze/park the old runner, back up and copy `~/.hermes/kanban`, sync referenced repos/vault paths, respawn active workers on cloud, and verify the old host has no ready/running duplicates. See `references/cloud-kanban-handoff.md`.

**Durable roadmap conductor script:** When the user wants an entire roadmap submitted/prioritized and automated as a test, first create/verify the PM tracker project/epics/tickets, park later work in Backlog, then create a repo-local Kanban script that chains each ticket's preflight behind the previous ticket's KM writeback. This enforces `T2 -> T3 -> ...` with dependencies instead of prose. See `references/durable-roadmap-conductor-scripts.md`.

**Cloud next-release Kanban decomposition:** When a book/roadmap audit should become future release work on a cloud host, create a separate board, write direct cloud repo/vault paths into every card, dispatch research/preflight cards first, and gate build → QA → review → KM with parent dependencies. See `references/cloud-next-release-kanban-decomposition.md`.

**Server-local operator vault decomposition:** When recreating a Conductor/Jake-vault pattern for another operator on that operator's own server, create the board on the target host, use operator-local repo/vault paths in every card, dispatch research cards first, and gate build → QA → review → KM behind parent dependencies. See `references/server-local-operator-vault-kanban.md`.

**Blocked-work resolver watchdog:** When the operator says the orchestrator should always watch blocked work and actively resolve it, do not merely report blocked cards. Add a lightweight watchdog that scans all Kanban boards, classifies blocked cards, creates one idempotent resolver card per blocker, dispatches the board, and stays silent when no Keith input is needed. Important: `review-required` or “waiting for QA/reviewer/maintainer” is normal process, not a true blocker. Model it as a reviewer/QA/maintainer task. Generic approval/auth/access/process blockers should route to orchestrator first; alert Keith only for strategy/product-scope or add/remove/change-functionality questions. If a legacy worker put routine review in `blocked`, the watchdog may route a resolver and auto-complete the original only after PASS/APPROVED resolver evidence, then dispatch dependents. See `references/blocked-work-resolver-watchdog.md`.

**Multi-host watchdog/status rollout:** When the operator wants the cloud blocked-work resolver or Slack status behavior duplicated across local Jake, Morgan, Loki, or other Hermes hosts, replicate the script-only cron pattern per host, verify skills/profiles/gateway/Slack routing on each target, and adapt status scripts to host-local board names, vault paths, and service names. Do not print Slack tokens; verify by target listing and a live send. See `references/multi-host-kanban-watchdog-and-slack-status.md`.

**Fleet-wide blocked resolver rollout:** When the operator wants the already-working blocked resolver setup copied from one Hermes host to the rest of the fleet, treat it as an ops rollout: identify the canonical source script/cron, copy the resolver + notifier/test/design artifacts to each target, ensure `kanban-worker` and `kanban-orchestrator` are enabled for every profile, create the no-agent `every 2m` cron job, and verify py_compile/manual run/cron `Last run ok` per host. See `references/fleet-blocked-resolver-rollout.md`.

**Cloud-primary Kanban migration:** When moving remaining stories from a local Jake/workstation to a cloud Hermes host, treat the Kanban DB/logs as durable state and the gateway as a writer. Stop cloud gateway before rsyncing, back up cloud `~/.hermes/kanban`, copy board DB/logs plus supporting workspaces, explicitly reclaim/respawn any running card on cloud, and park local duplicates so the local gateway cannot claim them. See `references/cloud-primary-kanban-migration.md`.

**Hermes-wide orchestrator update:** When the operator asks whether a domain-specific bridge notification covers only one workstream, answer plainly and add a separate Hermes-wide status notifier if useful. It should report services, profile state, Kanban blocked/running/resolver cards, and relevant cron jobs on state change only. See `references/hermes-orchestrator-status-updates.md`.

**Specialist model assignment:** When the operator asks whether different swarm roles should use different models, do not reshuffle by vibe, but also do not stop at explanation if they gave permission to evaluate and assign. Research model strengths, verify the exact provider/model works in Hermes, then make the smallest low-risk profile assignment first (usually `maintainer`, `researcher`, or `ops-watch`; avoid mid-flight builder/reviewer/QA churn). See `references/swarm-model-assignment.md`.

**Supervised external builder lane (Claude Code/Codex/OpenCode):** If the user wants a specific coding agent CLI to serve as the builder, do not answer “can’t.” Create a dedicated builder lane/wrapper that reads the Kanban card context, runs the external CLI in the correct repo/worktree with explicit guardrails, blocks push/deploy/secrets/destructive commands, runs required verification, then posts a Kanban result/handoff. Keep Jake/Hermes as PM/router/reviewer and treat the external CLI as the mechanic, not the release manager.

**Fan-out + fan-in (research → synthesize):** N research-style cards with no parents, one synthesis card with all of them as parents.

**Parallel implementation + validation:** one implementer card makes the change while one explorer/researcher card verifies config, docs, or source mapping. A reviewer card can depend on both. Do not make the implementer own unrelated verification just because the user mentioned both in one sentence.

**Pipeline with gates:** Default project delivery pipeline is `planner/preflight → implementer/builder → QA/UAT + reviewer → acceptance → KM/writeback`. QA/UAT is a normal testing lane assigned to `qa`, not a blocked state. Acceptance depends on both QA/UAT and reviewer. Reviewer/QA only use blocked for a real defect, missing access, failed verification needing a fix, or explicit operator input; otherwise they complete with PASS/APPROVED evidence.

**Dashboard/process-lane honesty:** Do not tell the operator that role/process lanes are visible unless the Kanban UI actually renders them. Hermes engine statuses (`triage`, `todo`, `ready`, `running`, `blocked`, `done`, and any native `review` support) are not the same thing as delivery stages such as Build, Review, QA/UAT, Acceptance, or KM/Docs. Until the dashboard has a real process-lane/grouping view, model stages with explicit cards, assignees, titles, `current_step_key`/metadata where available, and dependency links; report plainly that the process exists in the graph but may still appear under Todo/Ready/Running in the UI.

**Process lanes vs dashboard columns:** Hermes' Kanban engine statuses and the visible dashboard columns are not the same thing. Do not tell the operator that lanes like `Review`, `QA/UAT`, `Acceptance`, or `KM/Docs` are visible unless you have verified the dashboard actually renders them. If the dashboard only shows engine states (`Triage`, `Todo`, `Ready`, `In Progress`, `Blocked`, `Done`), model process stages as explicit gated cards/metadata/assignees, and call out that a true Process Lane view still needs a UI patch.

**Auto-resolver guardrail:** A resolver PASS/APPROVED can auto-complete the original blocked card only for routine/process blockers such as `review-required` or iteration-budget cleanup. For real QA/UAT defects, broken links, failed tests, stale writeback, missing access, or product-policy questions, route the smallest repair, then unblock/re-run the original QA/reviewer gate. Repair evidence is not acceptance evidence.

**Process lanes vs visible dashboard columns:** Do not tell the operator that Review, QA/UAT, Acceptance, or KM/Docs “lanes” are visible unless you have verified the dashboard actually renders those columns. Hermes Kanban engine statuses may be only `triage/todo/ready/running/blocked/done` in the UI while the process is represented by task titles, assignees, dependencies, `current_step_key`, or metadata. When the operator asks for role lanes, distinguish clearly between (a) the durable process graph/cards and (b) actual board UI columns. If visibility matters, propose or queue a dashboard/process-view change such as `View by: Status | Process Lane | Assignee` rather than pretending process cards equal visual lanes.

**Same-profile queue:** N tasks, all assigned to the same profile, no dependencies between them. Dispatcher serializes — that profile processes them in priority order, accumulating experience in its own memory.

**Human-in-the-loop:** Any task can `kanban_block()` to wait for input. Dispatcher respawns after `/unblock`. The comment thread carries the full context.

## Pitfalls

**Inventing profile names that don't exist.** The dispatcher silently fails to spawn unknown assignees — the card just sits in `ready` forever. Always assign to a profile from your Step 0 discovery; ask the user if you're unsure.

**Bundling independent lanes into one card.** If the user asks for two independent outcomes, create two cards. Example: "fix blockers and check model variants" is not one fixer task; create a fixer/engineer card for the fixes and an explorer/researcher card for the variant check, then optionally gate review on both.

**Over-linking because of wording.** "Finally check X" may still be parallel with implementation if X is static config, docs, or source discovery. Link it after implementation only when the check depends on the implementation result.

**Forgetting dependency links.** If the task graph says `research -> implement -> review`, do not create all tasks as independent ready cards. Use parent links so implement/review cannot run before their inputs exist.

**Reassignment vs. new task.** If a reviewer blocks with "needs changes," create a NEW task linked from the reviewer's task — don't re-run the same task with a stern look. The new task is assigned to the original implementer profile.

**Argument order for links.** `kanban_link(parent_id=..., child_id=...)` — parent first. Mixing them up demotes the wrong task to `todo`.

**Don't pre-create the whole graph if the shape depends on intermediate findings.** If T3's structure depends on what T1 and T2 find, let T3 exist as a "synthesize findings" task whose own first step is to read parent handoffs and plan the rest. Orchestrators can spawn orchestrators.

**Tenant inheritance.** If `HERMES_TENANT` is set in your env, pass `tenant=os.environ.get("HERMES_TENANT")` on every `kanban_create` call so child tasks stay in the same namespace.

## Recovering stuck workers

When a worker profile keeps crashing, hallucinating, or getting blocked by its own mistakes (usually: wrong model, missing skill, broken credential), the kanban dashboard flags the task with a ⚠ badge and opens a **Recovery** section in the drawer. Three primary actions:

1. **Reclaim** (or `hermes kanban reclaim <task_id>`) — abort the running worker immediately and reset the task to `ready`. The existing claim TTL is ~15 min; this is the fast path out.
2. **Reassign** (or `hermes kanban reassign <task_id> <new-profile> --reclaim`) — switch the task to a different profile (one that exists on this setup) and let the dispatcher pick it up with a fresh worker.
3. **Change profile model** — the dashboard prints a copy-paste hint for `hermes -p <profile> model` since profile config lives on disk; edit it in a terminal, then Reclaim to retry with the new model.

**Bundled skill recovery:** If multiple Kanban workers immediately crash with `Error: Unknown skill(s): kanban-worker`, treat it as a profile skill-registry/setup issue, not a task-content issue. Restore the bundled Kanban skills per worker profile with `hermes -p <profile> skills reset kanban-worker --restore` and `hermes -p <profile> skills reset kanban-orchestrator --restore`, restart the gateway/dispatcher, then unblock and retry one card before unblocking all siblings. See `references/kanban-worker-skill-recovery.md`.

**Iteration-budget recovery:** A task blocked with “iteration budget exhausted” is not automatically a failed implementation. Inspect `hermes kanban --board <board> show <task>`, `runs`, `log`, and the repo diff. If the worker already made a coherent diff and ran enough verification, independently rerun the key checks, then manually `complete` the card with a transparent operator handoff/metadata and dispatch the dependent QA/reviewer cards. If evidence is weak or tests fail, do not complete it; create a new fix/retry card or reclaim/reassign.

Hallucination warnings appear on tasks where a worker's `kanban_complete(created_cards=[...])` claim included card ids that don't exist or weren't created by the worker's profile (the gate blocks the completion), or where the free-form summary references `t_<hex>` ids that don't resolve (advisory prose scan, non-blocking). Both produce audit events that persist even after recovery actions — the trail stays for debugging.
