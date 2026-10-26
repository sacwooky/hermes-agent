# Agent acceptance for the G3 epic gate

When an epic finishes, the autonomy sweep creates a G3 acceptance gate: a blocked task plus a pending
`epic_acceptance` approval. By default only a person can decide it (`hermes kanban accept <gate>` or the WebUI
approval). With agent acceptance on, an independent agent decides it instead, and the decision goes into the
owner's daily digest.

## Configuration

```yaml
kanban:
  autonomy:
    epic_acceptance:
      enabled: true
      approver: agent            # keith (default) | agent
      reviewer_assignee: product # the profile that reviews; default product
      fix_assignee: builder      # who fixes a rejected epic
```

Per-board overrides under `kanban.boards.<slug>.autonomy` work as for the other autonomy settings. An unknown
`approver` value logs a warning and falls back to `keith`.

## How it works

1. For each pending approval the sweep opens one **acceptance review** task (`work_item_type`
   `acceptance_review`) assigned to `reviewer_assignee`, carrying the gate's evidence. The gate stays blocked.
2. The reviewer checks the demo against the epic's acceptance criteria and finishes with `kanban_complete`,
   passing the top-level arguments `acceptance_decision` (`"accept"` or `"reject"`) and `acceptance_reason`
   (one line); the tool stores them in the run's metadata. The same keys inside `metadata`
   (`acceptance_decision`, `reason`) still work. `kanban_complete` refuses to finish a review without a valid
   decision, so the reviewer retries instead of leaving the gate for Keith. On a conductor board the review runs as
   its reviewer's own worker, never inside the conductor session.
3. On the next tick the approval is decided as `agent:<profile>` with the reason as its comment. Accept releases
   the gate; reject spawns one fix story, and when it lands the gate re-arms and a fresh review opens.

## Safety rules

- **Fail closed.** A review that finishes without `accept`/`reject` and a non-empty reason decides nothing: the
  gate gets an `acceptance_review_invalid` event and stays for the owner.
- **Independence.** If `reviewer_assignee` is the `fix_assignee` or built any story of the epic, no review opens:
  the gate gets an `acceptance_review_refused` event and stays for the owner.
- **One review per approval.** The review task carries an idempotency key, so a retry never opens a second one.
- **The owner still wins.** `hermes kanban accept` and the WebUI approval work as before and take effect if used
  first.
- **Digest.** Each agent decision is recorded with `~/.hermes/scripts/fleet-decided.py record --kind choice`. This
  is best effort: the outcome is written to the gate as an `acceptance_decision_recorded` event (`recorded`,
  `missing`, `rc=<n>` or `error: …`), and a failed record never blocks or undoes the decision.
