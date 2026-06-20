"""Hermes v18 orchestration / skill / doc validators (step S6).

Standalone, low-risk, additive. Every public function is a *pure* function over
plain Python dicts/strings: no side effects, no I/O, no DB access, no imports of
dispatcher/kanban core. Each validator returns a ``list[str]`` of human-readable
violation messages; an empty list means the input passed.

The four validators:

* :func:`validate_orchestration` — structural checks over an epic/feature/story
  task graph (labelling, parent chains, dependency ordering, traceability,
  acceptance criteria, blocked dispatch, runaway parallelism).
* :func:`validate_ui_story_qa` — a UI story may only be ``done`` once it carries
  browser evidence and SC-QA approval.
* :func:`lint_no_bare_gate_labels` — doc lint flagging bare ``G1``/``G2``/``G3``
  used as primary language rather than via a semantic gate name / legacy alias.
* :func:`check_skills_exist` — required skills must be installed, mapped, or
  explicitly conditional / create-on-first-need (no hollow skills).
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "STORY_PREFIX_RE",
    "validate_orchestration",
    "validate_ui_story_qa",
    "lint_no_bare_gate_labels",
    "check_skills_exist",
]

# A story title must begin with ``S<si>-F<fi>-E<ei> — `` (em-dash + space).
# Example: "S3-F2-E1 — Render the cart summary".
STORY_PREFIX_RE = re.compile(r"^S(?P<si>\d+)-F(?P<fi>\d+)-E(?P<ei>\d+)\s+—\s+")

# The bare label portion only, e.g. "S3-F2-E1", used for duplicate detection.
_STORY_LABEL_RE = re.compile(r"^(S\d+-F\d+-E\d+)\b")


def _meta(task: dict, key: str, default: Any = None) -> Any:
    """Read ``key`` from a task's ``metadata`` dict if present, else top level.

    Metadata-ish flags (``ui``, ``justified_out_of_order``, ``independent_proof``)
    may live either directly on the task dict or nested under a ``metadata`` key.
    Top-level wins; nested metadata is the fallback.
    """
    if key in task:
        return task[key]
    meta = task.get("metadata")
    if isinstance(meta, dict) and key in meta:
        return meta[key]
    return default


def validate_orchestration(
    tasks: list[dict], max_parallel: int = 4
) -> list[str]:
    """Validate an epic→feature→story orchestration graph.

    Parameters
    ----------
    tasks:
        List of task dicts. Recognised keys: ``id``, ``title``,
        ``work_item_type`` (``epic`` | ``feature`` | ``story``), ``parent_id``,
        ``depends_on`` (list of ids), ``acceptance_criteria`` (str),
        ``feature_goal`` (str), ``prd_req_ids`` (list), ``wireframe_ids``
        (list), ``status``, plus optional metadata flags ``ui``,
        ``justified_out_of_order``, ``independent_proof``.
    max_parallel:
        Maximum number of stories allowed in ``in_progress``/``dispatched`` at
        once before excessive-parallelism is flagged (unless each carries an
        ``independent_proof``).

    Returns a list of violation strings; empty means the graph is valid.
    """
    violations: list[str] = []

    by_id: dict[Any, dict] = {}
    for task in tasks:
        tid = task.get("id")
        by_id[tid] = task

    def label_of(task: dict) -> str:
        return str(task.get("id", task.get("title", "<unknown>")))

    # ---- duplicate Story labels (same S#-F#-E# prefix used twice) ----------
    seen_labels: dict[str, Any] = {}
    for task in tasks:
        if task.get("work_item_type") != "story":
            continue
        title = task.get("title") or ""
        m = _STORY_LABEL_RE.match(title)
        if not m:
            continue
        lbl = m.group(1)
        if lbl in seen_labels:
            violations.append(
                f"Task {label_of(task)}: duplicate Story label '{lbl}' "
                f"(already used by task {seen_labels[lbl]})"
            )
        else:
            seen_labels[lbl] = task.get("id")

    # ---- parent-type and prefix-vs-metadata checks ------------------------
    for task in tasks:
        wtype = task.get("work_item_type")
        if wtype == "story":
            parent = by_id.get(task.get("parent_id"))
            # story parent must be a feature
            if parent is None:
                violations.append(
                    f"Task {label_of(task)}: story has no resolvable parent "
                    f"(parent_id={task.get('parent_id')!r})"
                )
            elif parent.get("work_item_type") != "feature":
                violations.append(
                    f"Task {label_of(task)}: story parent must be a feature, "
                    f"but parent {label_of(parent)} is "
                    f"'{parent.get('work_item_type')}'"
                )

            # title-prefix vs metadata/parent-chain consistency
            title = task.get("title") or ""
            pm = STORY_PREFIX_RE.match(title)
            if not pm:
                violations.append(
                    f"Task {label_of(task)}: story title does not start with a "
                    f"valid 'S<si>-F<fi>-E<ei> — ' prefix (title={title!r})"
                )
            elif parent is not None and parent.get("work_item_type") == "feature":
                fi = pm.group("fi")
                ei = pm.group("ei")
                # feature ordinal/epic from feature title prefix "F<fi>-E<ei>"
                feat_title = parent.get("title") or ""
                fm = re.match(r"^F(?P<fi>\d+)-E(?P<ei>\d+)\b", feat_title)
                if fm is None:
                    violations.append(
                        f"Task {label_of(parent)}: feature title does not start "
                        f"with a valid 'F<fi>-E<ei>' prefix "
                        f"(title={feat_title!r})"
                    )
                else:
                    if fm.group("fi") != fi or fm.group("ei") != ei:
                        violations.append(
                            f"Task {label_of(task)}: story prefix F{fi}-E{ei} "
                            f"does not match parent feature "
                            f"F{fm.group('fi')}-E{fm.group('ei')}"
                        )
                    # epic from feature's parent
                    epic = by_id.get(parent.get("parent_id"))
                    if epic is not None and epic.get("work_item_type") == "epic":
                        em = re.match(
                            r"^E(?P<ei>\d+)\b", epic.get("title") or ""
                        )
                        if em is not None and em.group("ei") != ei:
                            violations.append(
                                f"Task {label_of(task)}: story epic ordinal "
                                f"E{ei} does not match epic "
                                f"E{em.group('ei')}"
                            )
        elif wtype == "feature":
            parent = by_id.get(task.get("parent_id"))
            # feature parent must be an epic
            if parent is None:
                violations.append(
                    f"Task {label_of(task)}: feature has no resolvable parent "
                    f"(parent_id={task.get('parent_id')!r})"
                )
            elif parent.get("work_item_type") != "epic":
                violations.append(
                    f"Task {label_of(task)}: feature parent must be an epic, "
                    f"but parent {label_of(parent)} is "
                    f"'{parent.get('work_item_type')}'"
                )

    # ---- story acceptance_criteria / feature_goal -------------------------
    for task in tasks:
        if task.get("work_item_type") != "story":
            continue
        if not (task.get("acceptance_criteria") or "").strip():
            violations.append(
                f"Task {label_of(task)}: story missing acceptance_criteria"
            )
        if not (task.get("feature_goal") or "").strip():
            violations.append(
                f"Task {label_of(task)}: story missing feature_goal"
            )

    # ---- helpers for ordinal extraction (story / feature) -----------------
    def story_ordinal(task: dict) -> tuple[int, int, int] | None:
        """Return (ei, fi, si) ordering tuple for a story, or None."""
        m = STORY_PREFIX_RE.match(task.get("title") or "")
        if not m:
            return None
        return (int(m.group("ei")), int(m.group("fi")), int(m.group("si")))

    # ---- dependency-vs-order ---------------------------------------------
    for task in tasks:
        if task.get("work_item_type") != "story":
            continue
        a_ord = story_ordinal(task)
        if a_ord is None:
            continue
        for dep_id in task.get("depends_on") or []:
            dep = by_id.get(dep_id)
            if dep is None or dep.get("work_item_type") != "story":
                continue
            b_ord = story_ordinal(dep)
            if b_ord is None:
                continue
            # A depends_on B but A is ordered strictly earlier than B
            if a_ord < b_ord:
                if not _meta(task, "justified_out_of_order", False):
                    violations.append(
                        f"Task {label_of(task)}: depends on {label_of(dep)} "
                        f"which is ordered later "
                        f"(E{a_ord[0]}-F{a_ord[1]}-S{a_ord[2]} depends on "
                        f"E{b_ord[0]}-F{b_ord[1]}-S{b_ord[2]}); set "
                        f"justified_out_of_order to allow"
                    )

    # ---- feature traceability (PRD + conditional wireframe) ---------------
    # Build parent->children for descendant lookups.
    children: dict[Any, list[dict]] = {}
    for task in tasks:
        children.setdefault(task.get("parent_id"), []).append(task)

    def feature_has_ui_descendant(feature: dict) -> bool:
        fid = feature.get("id")
        for child in children.get(fid, []):
            if child.get("work_item_type") == "story" and _meta(
                child, "ui", False
            ):
                return True
        return False

    for task in tasks:
        if task.get("work_item_type") != "feature":
            continue
        if not (task.get("prd_req_ids") or []):
            violations.append(
                f"Task {label_of(task)}: feature missing PRD traceability "
                f"(empty prd_req_ids)"
            )
        if feature_has_ui_descendant(task) and not (
            task.get("wireframe_ids") or []
        ):
            violations.append(
                f"Task {label_of(task)}: feature missing wireframe traceability "
                f"(empty wireframe_ids) but has a UI story descendant"
            )

    # ---- blocked story dispatched ----------------------------------------
    _active = ("in_progress", "dispatched", "ready")
    for task in tasks:
        if task.get("work_item_type") != "story":
            continue
        if task.get("status") not in _active:
            continue
        for dep_id in task.get("depends_on") or []:
            dep = by_id.get(dep_id)
            if dep is None:
                continue
            if dep.get("status") != "done":
                violations.append(
                    f"Task {label_of(task)}: blocked story dispatched "
                    f"(status='{task.get('status')}') while dependency "
                    f"{label_of(dep)} status='{dep.get('status')}' (not done)"
                )

    # ---- excessive parallelism -------------------------------------------
    running = [
        task
        for task in tasks
        if task.get("work_item_type") == "story"
        and task.get("status") in ("in_progress", "dispatched")
    ]
    if len(running) > max_parallel:
        if not all(_meta(t, "independent_proof", None) for t in running):
            ids = ", ".join(label_of(t) for t in running)
            violations.append(
                f"Excessive parallelism: {len(running)} stories "
                f"in_progress/dispatched exceeds max_parallel={max_parallel} "
                f"without independent_proof on each ({ids})"
            )

    return violations


def validate_ui_story_qa(story: dict) -> list[str]:
    """Validate that a UI story is not marked ``done`` without QA evidence.

    For a story whose metadata ``ui`` is True, being ``done`` requires:

    * a non-empty ``browser_evidence`` list, and
    * ``sc_qa_approved`` is boolean ``True``.

    Non-UI stories and not-yet-done stories produce no violations.
    """
    violations: list[str] = []
    if not _meta(story, "ui", False):
        return violations
    if story.get("status") != "done":
        return violations

    label = str(story.get("id", story.get("title", "<unknown>")))

    evidence = _meta(story, "browser_evidence", None)
    if not (isinstance(evidence, list) and len(evidence) > 0):
        violations.append(
            f"Task {label}: UI story marked 'done' without non-empty "
            f"browser_evidence"
        )

    if _meta(story, "sc_qa_approved", None) is not True:
        violations.append(
            f"Task {label}: UI story marked 'done' without sc_qa_approved=True"
        )

    return violations


# Words that, when near a bare gate label, mean it is a legacy alias or part of
# a semantic gate name rather than primary language.
_GATE_ALLOW_WORDS = ("legacy", "alias", "old")
_BARE_GATE_RE = re.compile(r"\bG([123])\b")
# A semantic gate name like INTAKE-GATE, WIREFRAME-SC-GATE, EPIC-ACCEPTANCE-GATE.
_SEMANTIC_GATE_RE = re.compile(r"\b[A-Z][A-Z-]*-GATE\b")
_ALLOW_RADIUS = 40


def lint_no_bare_gate_labels(text: str) -> list[str]:
    """Flag bare ``G1``/``G2``/``G3`` used as primary gate language.

    A bare gate token is allowed (not flagged) when, within ~40 characters of
    it, there appears one of the words 'legacy', 'alias', 'old', or a semantic
    gate name such as ``INTAKE-GATE`` / ``WIREFRAME-SC-GATE``. Returns
    line-numbered violation strings.
    """
    violations: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for m in _BARE_GATE_RE.finditer(line):
            start, end = m.start(), m.end()
            lo = max(0, start - _ALLOW_RADIUS)
            hi = min(len(line), end + _ALLOW_RADIUS)
            window = line[lo:hi]
            window_lower = window.lower()
            allowed = any(w in window_lower for w in _GATE_ALLOW_WORDS)
            if not allowed and _SEMANTIC_GATE_RE.search(window):
                allowed = True
            if not allowed:
                violations.append(
                    f"Line {lineno}: bare gate label '{m.group(0)}' used as "
                    f"primary language; use a semantic gate name (e.g. "
                    f"INTAKE-GATE) or mark it a legacy alias"
                )
    return violations


def check_skills_exist(
    required: list[str],
    installed: list[str],
    skill_map: dict[str, str],
) -> list[str]:
    """Validate that every required skill is satisfiable.

    A required skill passes if any of:

    * it is in ``installed`` directly; or
    * ``skill_map[skill]`` resolves to an installed skill; or
    * ``skill_map[skill]`` is conditional (value starts with ``'conditional:'``)
      or a create-on-first-need pointer (value starts with
      ``'create-on-first-need'``).

    ``create-on-first-need`` is explicitly allowed — it is a PI-Story pointer,
    not a hollow/fake skill. Returns one violation per genuinely missing skill.
    """
    installed_set = set(installed)
    violations: list[str] = []
    for skill in required:
        if skill in installed_set:
            continue
        mapped = skill_map.get(skill)
        if mapped is not None:
            if mapped.startswith("conditional:") or mapped.startswith(
                "create-on-first-need"
            ):
                continue
            if mapped in installed_set:
                continue
            violations.append(
                f"Skill '{skill}': mapped to '{mapped}' which is not installed"
            )
        else:
            violations.append(
                f"Skill '{skill}': required but not installed, mapped, or "
                f"marked conditional/create-on-first-need"
            )
    return violations
