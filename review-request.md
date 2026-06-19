# Review request — advisory design_quality axis (experience-first builds Stage 3)

claude-code-authored: yes
branch: feat/design-quality-axis
base: realfork/main (current; rebased to 6b07bb307 — prior stale-base diff artifact removed)
commit: 3549aca7d
risk: medium (touches review-core files; ADVISORY/non-blocking; wired into the L1 review pass)

## Scope / intent
Stage 3 of experience-first builds (decision-experience-first-builds-v1). Adds an
ADVISORY, non-blocking design-quality review capability WITHOUT altering the binding
fusion verdict path. Wired into the L1 review pass (artifact/direction-scoped, fail-open).

## Files
- `hermes_cli/review_loop/design_quality.py` (NEW, 185 LOC): evidence schema
  (DesignEvidence), pure text evidence builder, 9-criterion rubric, injection-safe
  text-only `review_design_quality()` (fails safe to "insufficient_evidence").
- `hermes_cli/kanban_autonomy.py` (+8/-3): registers `design_quality` as a
  conformance axis (harvest tuple + kind_map). ADVISORY only.
- `hermes_cli/review_loop/fusion.py` (+12/-3): unconditional injection-safety note
  added to juror + judge system prompts. Verdict schema unchanged.
- `tests/hermes_cli/test_design_quality.py` (NEW): 8 tests.

## What to scrutinise (reviewer focus)
1. **No blocking-path contamination:** confirm `design_quality` is never consulted on
   any integration-gate / crosscheck / blocking path — only harvested into the G3
   acceptance packet. (See kanban_autonomy `_harvest_conformance_verdicts` +
   `record_conformance_verdict`; blocking logic around the `security` crosscheck must
   be unaffected.)
2. **Injection safety:** jurors/judges now treat all artifact content (code, HTML,
   wireframes, rationale) as untrusted data; embedded directives ignored. Confirm the
   framing is correct and does not weaken the existing review.
3. **Text-only honesty:** jurors are no-vision; the design reviewer must score
   "insufficient_evidence" for pixel-dependent criteria rather than hallucinate.
4. **Fail-safe:** failed/unparseable model output returns "insufficient_evidence",
   never a spurious pass.

## Tests
- `test_design_quality.py` + `test_fusion.py`: 22 passed.
- `test_kanban_autonomy.py`: 17 passed, 1 PRE-EXISTING failure
  (`test_unintegrated_sweep_creates_integrate_task`) that fails identically on the
  unmodified trunk — NOT caused by this change.

Test method (running install is `pip install -e .` off main tree):
    cd ~/.hermes/hermes-agent.worktrees/wt-stage3-design-quality
    PYTHONPATH="$PWD" ~/.hermes/hermes-agent/venv/bin/python -m pytest \
      tests/hermes_cli/test_design_quality.py tests/hermes_cli/test_fusion.py -q

## Live wiring (NOW INCLUDED — addresses prior BLOCK fusion-8862ae56)
The advisory review runs automatically: `run_l1_screen_for_review_task` (the review-phase,
non-blocking, fail-open, once-per-artifact L1 pass) now invokes
`run_advisory_design_review(conn, task_id, epic_id, ...)` after the L1 screen.
`_epic_for_story()` resolves the epic (epic root is the story's child in `task_links`);
the review lane reuses the L1 endpoint (ninerouter defaults). Non-UI tasks self-skip
(no G2 selected direction). The verdict records on the epic and `_harvest_conformance_verdicts`
surfaces it in the G3 packet. The advisory pass is fully isolated: a crash never breaks the
L1 review (test: `test_l1_hook_advisory_failure_is_isolated`). Stays advisory (never gates)
until calibrated on >=3 real UI/website boards.

## Prior BLOCK findings — resolution
- v1 finding (enforcement): RESOLVED in v2 — `_ADVISORY_CONFORMANCE_AXES`, crosscheck=True
  raises, B4/B5 independence gate bypassed for advisory axes (tests added).
- v2 finding (integration not wired): RESOLVED here — wired into the L1 review pass (above).

## v4: dedup hardening
The advisory pass now has an EXPLICIT once-per-artifact guard (its own `design_review_advisory` marker), idempotent even if the l1_screen emit fails — cannot record a duplicate verdict (test: `test_l1_hook_advisory_is_deduped`). Branch rebased onto current trunk so the diff is ONLY the 5 Stage-3 files.

## v5: artifact(direction)-scoped semantics
- Dedup is now DIRECTION-scoped, not task-scoped: the `design_review_advisory` marker carries the reviewed `selected_direction_id`. A Stage-2 re-pick (new direction) gets a fresh review; the same direction never re-runs. The advisory pass runs independent of the L1-screen dedup so a revised direction is not permanently suppressed (test: `test_l1_hook_advisory_reruns_on_new_direction`).
- Selection is deterministic: `_latest_selected_direction` takes the LATEST wireframe approval (list_task_approvals is newest-first) and the artifact actually carrying `selected_direction_id` (not blindly `[0]`).
