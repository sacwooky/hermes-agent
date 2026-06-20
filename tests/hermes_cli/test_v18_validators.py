"""Tests for hermes_cli.v18_validators (step S6).

Covers a valid Epic→Feature→Story graph, every orchestration violation type,
UI-story QA gating, the bare-gate doc lint, and the skill-existence check.
"""

from __future__ import annotations

import copy

import pytest

from hermes_cli.v18_validators import (
    STORY_PREFIX_RE,
    check_skills_exist,
    lint_no_bare_gate_labels,
    validate_orchestration,
    validate_ui_story_qa,
)


# --------------------------------------------------------------------------- #
# Fixtures / graph builders
# --------------------------------------------------------------------------- #
def _valid_graph() -> list[dict]:
    """A minimal valid Epic→Feature→Story graph that passes all checks."""
    epic = {
        "id": "E1",
        "title": "E1 — Checkout epic",
        "work_item_type": "epic",
        "parent_id": None,
        "status": "ready",
    }
    feature = {
        "id": "F1",
        "title": "F1-E1 — Cart feature",
        "work_item_type": "feature",
        "parent_id": "E1",
        "status": "ready",
        "prd_req_ids": ["PRD-1"],
        "wireframe_ids": [],
    }
    story1 = {
        "id": "S1",
        "title": "S1-F1-E1 — Add item to cart",
        "work_item_type": "story",
        "parent_id": "F1",
        "status": "done",
        "depends_on": [],
        "acceptance_criteria": "Item appears in cart.",
        "feature_goal": "Customers can build a cart.",
    }
    story2 = {
        "id": "S2",
        "title": "S2-F1-E1 — Remove item from cart",
        "work_item_type": "story",
        "parent_id": "F1",
        "status": "ready",
        "depends_on": ["S1"],
        "acceptance_criteria": "Item disappears from cart.",
        "feature_goal": "Customers can build a cart.",
    }
    return [epic, feature, story1, story2]


# --------------------------------------------------------------------------- #
# Regex sanity
# --------------------------------------------------------------------------- #
def test_story_prefix_regex_matches_and_groups():
    m = STORY_PREFIX_RE.match("S3-F2-E1 — Render cart summary")
    assert m is not None
    assert (m.group("si"), m.group("fi"), m.group("ei")) == ("3", "2", "1")


def test_story_prefix_regex_requires_em_dash():
    # Plain hyphen separator must not match.
    assert STORY_PREFIX_RE.match("S1-F1-E1 - hyphen not em-dash") is None


# --------------------------------------------------------------------------- #
# validate_orchestration — happy path
# --------------------------------------------------------------------------- #
def test_valid_graph_passes():
    assert validate_orchestration(_valid_graph()) == []


# --------------------------------------------------------------------------- #
# Individual violation types
# --------------------------------------------------------------------------- #
def test_duplicate_story_label():
    graph = _valid_graph()
    # Make story2 reuse story1's label prefix.
    for t in graph:
        if t["id"] == "S2":
            t["title"] = "S1-F1-E1 — Duplicate label"
    out = validate_orchestration(graph)
    assert any("duplicate Story label" in v for v in out)


def test_prefix_metadata_mismatch_feature():
    graph = _valid_graph()
    # Story claims F9 but its parent feature is F1.
    for t in graph:
        if t["id"] == "S2":
            t["title"] = "S2-F9-E1 — Mismatched feature ordinal"
            t["depends_on"] = []
    out = validate_orchestration(graph)
    assert any("does not match parent feature" in v for v in out)


def test_prefix_metadata_mismatch_epic():
    graph = _valid_graph()
    # Story claims E9 but the epic chain is E1; feature must also agree, so
    # bump the feature prefix to F1-E9 to isolate the epic mismatch.
    for t in graph:
        if t["id"] == "F1":
            t["title"] = "F1-E9 — Cart feature"
        if t["id"] == "S1":
            t["title"] = "S1-F1-E9 — Add item"
        if t["id"] == "S2":
            t["title"] = "S2-F1-E9 — Remove item"
            t["depends_on"] = []
    out = validate_orchestration(graph)
    assert any("epic ordinal" in v and "does not match epic" in v for v in out)


def test_invalid_story_prefix_format():
    graph = _valid_graph()
    for t in graph:
        if t["id"] == "S2":
            t["title"] = "no prefix here at all"
            t["depends_on"] = []
    out = validate_orchestration(graph)
    assert any("does not start with a valid" in v for v in out)


def test_story_parent_must_be_feature():
    graph = _valid_graph()
    # Re-parent story1 directly under the epic.
    for t in graph:
        if t["id"] == "S1":
            t["parent_id"] = "E1"
    out = validate_orchestration(graph)
    assert any("story parent must be a feature" in v for v in out)


def test_feature_parent_must_be_epic():
    graph = _valid_graph()
    # Re-parent the feature under a story.
    for t in graph:
        if t["id"] == "F1":
            t["parent_id"] = "S1"
    out = validate_orchestration(graph)
    assert any("feature parent must be an epic" in v for v in out)


def test_dependency_vs_order_flagged():
    graph = _valid_graph()
    # Make S1 depend on the later S2 (out of order).
    for t in graph:
        if t["id"] == "S1":
            t["depends_on"] = ["S2"]
            t["status"] = "ready"  # avoid unrelated blocked-dispatch noise
        if t["id"] == "S2":
            t["status"] = "done"
    out = validate_orchestration(graph)
    assert any("ordered later" in v for v in out)


def test_dependency_vs_order_justified_ok():
    graph = _valid_graph()
    for t in graph:
        if t["id"] == "S1":
            t["depends_on"] = ["S2"]
            t["status"] = "ready"
            t["metadata"] = {"justified_out_of_order": True}
        if t["id"] == "S2":
            t["status"] = "done"
    out = validate_orchestration(graph)
    assert not any("ordered later" in v for v in out)


def test_feature_missing_prd_traceability():
    graph = _valid_graph()
    for t in graph:
        if t["id"] == "F1":
            t["prd_req_ids"] = []
    out = validate_orchestration(graph)
    assert any("missing PRD traceability" in v for v in out)


def test_feature_missing_wireframe_when_ui_descendant():
    graph = _valid_graph()
    # Mark a story as UI; feature has no wireframe ids → flag.
    for t in graph:
        if t["id"] == "S1":
            t["metadata"] = {"ui": True}
        if t["id"] == "F1":
            t["wireframe_ids"] = []
    out = validate_orchestration(graph)
    assert any("missing wireframe traceability" in v for v in out)


def test_feature_wireframe_not_required_without_ui_descendant():
    graph = _valid_graph()
    # No UI descendant, empty wireframe_ids → must NOT flag wireframe.
    out = validate_orchestration(graph)
    assert not any("missing wireframe traceability" in v for v in out)


def test_feature_wireframe_present_with_ui_descendant_ok():
    graph = _valid_graph()
    for t in graph:
        if t["id"] == "S1":
            t["metadata"] = {"ui": True}
        if t["id"] == "F1":
            t["wireframe_ids"] = ["WF-1"]
    out = validate_orchestration(graph)
    assert not any("missing wireframe traceability" in v for v in out)


def test_story_missing_acceptance_criteria():
    graph = _valid_graph()
    for t in graph:
        if t["id"] == "S2":
            t["acceptance_criteria"] = ""
    out = validate_orchestration(graph)
    assert any("missing acceptance_criteria" in v for v in out)


def test_story_missing_feature_goal():
    graph = _valid_graph()
    for t in graph:
        if t["id"] == "S2":
            t["feature_goal"] = ""
    out = validate_orchestration(graph)
    assert any("missing feature_goal" in v for v in out)


def test_blocked_story_dispatched():
    graph = _valid_graph()
    # S2 depends on S1 which is not done, but S2 is in_progress.
    for t in graph:
        if t["id"] == "S1":
            t["status"] = "ready"
        if t["id"] == "S2":
            t["status"] = "in_progress"
    out = validate_orchestration(graph)
    assert any("blocked story dispatched" in v for v in out)


def test_blocked_dispatch_ok_when_dependency_done():
    graph = _valid_graph()
    for t in graph:
        if t["id"] == "S1":
            t["status"] = "done"
        if t["id"] == "S2":
            t["status"] = "in_progress"
    out = validate_orchestration(graph)
    assert not any("blocked story dispatched" in v for v in out)


def _parallel_graph(n: int, *, with_proof: bool) -> list[dict]:
    epic = {
        "id": "E1",
        "title": "E1 — Epic",
        "work_item_type": "epic",
        "parent_id": None,
        "status": "ready",
    }
    feature = {
        "id": "F1",
        "title": "F1-E1 — Feature",
        "work_item_type": "feature",
        "parent_id": "E1",
        "status": "ready",
        "prd_req_ids": ["PRD-1"],
        "wireframe_ids": [],
    }
    tasks = [epic, feature]
    for i in range(1, n + 1):
        story = {
            "id": f"S{i}",
            "title": f"S{i}-F1-E1 — Story {i}",
            "work_item_type": "story",
            "parent_id": "F1",
            "status": "in_progress",
            "depends_on": [],
            "acceptance_criteria": "ac",
            "feature_goal": "fg",
        }
        if with_proof:
            story["metadata"] = {"independent_proof": "proof-doc"}
        tasks.append(story)
    return tasks


def test_excessive_parallelism_flagged():
    out = validate_orchestration(_parallel_graph(5, with_proof=False))
    assert any("Excessive parallelism" in v for v in out)


def test_excessive_parallelism_ok_with_independent_proof():
    out = validate_orchestration(_parallel_graph(5, with_proof=True))
    assert not any("Excessive parallelism" in v for v in out)


def test_parallelism_within_limit_ok():
    out = validate_orchestration(_parallel_graph(4, with_proof=False))
    assert not any("Excessive parallelism" in v for v in out)


def test_custom_max_parallel_param():
    out = validate_orchestration(
        _parallel_graph(3, with_proof=False), max_parallel=2
    )
    assert any("Excessive parallelism" in v for v in out)


def test_valid_graph_is_not_mutated():
    graph = _valid_graph()
    snapshot = copy.deepcopy(graph)
    validate_orchestration(graph)
    assert graph == snapshot  # pure function, no side effects


# --------------------------------------------------------------------------- #
# validate_ui_story_qa
# --------------------------------------------------------------------------- #
def test_ui_story_done_without_evidence_fails():
    story = {
        "id": "S1",
        "status": "done",
        "metadata": {"ui": True},
    }
    out = validate_ui_story_qa(story)
    assert any("browser_evidence" in v for v in out)
    assert any("sc_qa_approved" in v for v in out)


def test_ui_story_done_with_evidence_and_approval_passes():
    story = {
        "id": "S1",
        "status": "done",
        "metadata": {
            "ui": True,
            "browser_evidence": ["screenshot-1.png"],
            "sc_qa_approved": True,
        },
    }
    assert validate_ui_story_qa(story) == []


def test_ui_story_evidence_present_but_not_approved_fails():
    story = {
        "id": "S1",
        "status": "done",
        "metadata": {
            "ui": True,
            "browser_evidence": ["screenshot-1.png"],
            "sc_qa_approved": False,
        },
    }
    out = validate_ui_story_qa(story)
    assert any("sc_qa_approved" in v for v in out)
    assert not any("browser_evidence" in v for v in out)


def test_non_ui_story_done_passes():
    story = {"id": "S1", "status": "done", "metadata": {"ui": False}}
    assert validate_ui_story_qa(story) == []


def test_ui_story_not_done_passes():
    story = {"id": "S1", "status": "in_progress", "metadata": {"ui": True}}
    assert validate_ui_story_qa(story) == []


def test_ui_story_flag_top_level():
    # ui / browser_evidence / sc_qa_approved at top level (not nested) also work.
    story = {
        "id": "S1",
        "status": "done",
        "ui": True,
        "browser_evidence": [],
        "sc_qa_approved": True,
    }
    out = validate_ui_story_qa(story)
    assert any("browser_evidence" in v for v in out)


# --------------------------------------------------------------------------- #
# lint_no_bare_gate_labels
# --------------------------------------------------------------------------- #
def test_bare_gate_flagged():
    out = lint_no_bare_gate_labels("The work must clear G1 before merge.")
    assert len(out) == 1
    assert "G1" in out[0]
    assert out[0].startswith("Line 1:")


def test_bare_gate_legacy_alias_allowed():
    out = lint_no_bare_gate_labels("INTAKE-GATE (formerly G1, legacy alias)")
    assert out == []


def test_bare_gate_near_semantic_name_allowed():
    out = lint_no_bare_gate_labels("WIREFRAME-SC-GATE replaces the old G2 step")
    assert out == []


def test_bare_gate_alias_word_allowed():
    out = lint_no_bare_gate_labels("G3 is an alias of EPIC-ACCEPTANCE-GATE")
    assert out == []


def test_bare_gate_line_numbers():
    text = "first line clean\nmust pass G2 here\nanother clean line\nand G3 too"
    out = lint_no_bare_gate_labels(text)
    assert len(out) == 2
    assert out[0].startswith("Line 2:")
    assert out[1].startswith("Line 4:")


def test_g4_not_flagged():
    # Only G1/G2/G3 are gate labels.
    assert lint_no_bare_gate_labels("G4 and G0 are not gates") == []


# --------------------------------------------------------------------------- #
# check_skills_exist
# --------------------------------------------------------------------------- #
def test_skill_installed_directly_passes():
    assert check_skills_exist(["build"], ["build", "test"], {}) == []


def test_skill_mapped_to_installed_passes():
    out = check_skills_exist(["compile"], ["build"], {"compile": "build"})
    assert out == []


def test_skill_conditional_passes():
    out = check_skills_exist(
        ["fancy-thing"], [], {"fancy-thing": "conditional:needs-gpu"}
    )
    assert out == []


def test_skill_create_on_first_need_passes():
    out = check_skills_exist(
        ["future-skill"],
        [],
        {"future-skill": "create-on-first-need:PI-Story-42"},
    )
    assert out == []


def test_skill_genuinely_missing_fails():
    out = check_skills_exist(["ghost"], ["build"], {})
    assert len(out) == 1
    assert "ghost" in out[0]


def test_skill_mapped_to_missing_target_fails():
    out = check_skills_exist(["compile"], ["test"], {"compile": "build"})
    assert len(out) == 1
    assert "not installed" in out[0]


def test_mixed_required_skills():
    out = check_skills_exist(
        ["build", "compile", "fancy", "ghost"],
        ["build", "linker"],
        {
            "compile": "linker",
            "fancy": "create-on-first-need",
        },
    )
    # only 'ghost' should fail
    assert len(out) == 1
    assert "ghost" in out[0]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
