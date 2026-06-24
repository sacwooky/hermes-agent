"""Tests for the deterministic SC-gate review dispatch builder (run-557 fix).

Validates that the builder inlines artifacts as text, marks URLs/paths
unreadable, prompts for the full v18 SC contract, and produces payloads that the
REAL ``enforce_supreme_court_contract`` accepts when conforming and rejects when
hollow — i.e. the leak (a review that clears the gate without a real scorecard)
cannot recur through this path.
"""

from __future__ import annotations

import json

import pytest

from hermes_cli.kanban_autonomy import enforce_supreme_court_contract
from hermes_cli.review_loop import sc_review


class _ChatStub:
    """ninerouter.chat-compatible stub: returns a fixed reply object."""

    def __init__(self, content, ok=True, error=None):
        self._content = content
        self._ok = ok
        self._error = error

    def __call__(self, model, messages, *, max_tokens=None, json_mode=None):
        self.last_messages = messages
        return type("R", (), {"content": self._content, "ok": self._ok, "error": self._error})()


def _conforming_reply(verdict="approved"):
    return {
        "verdict": verdict,
        "confidence": 0.9,
        "scorecard": {"user_request_alignment": 9, "user_flow": 8},
        "blocking_issues": [],
        "advisory_issues": [],
        "missing_skill_findings": [],
        "required_repair_actions": [],
        "evidence_reviewed": ["inlined wireframe html"],
        "calibration_substrate_flags": [],
    }


# ---- evidence inlining -------------------------------------------------------

def test_render_wireframe_inlines_text_and_marks_urls_unreadable():
    text = sc_review.render_wireframe_evidence(
        {
            "named_experience_direction": "Field-notes editorial",
            "html_excerpt": "<main>hello</main>",
            "options": [{"id": "A", "rationale": "sidebar-led"}],
            "tailscale_url": "https://x.ts.net/wf.html",
            "png_paths": ["/srv/.../a.png"],
        }
    )
    assert "Field-notes editorial" in text
    assert "<main>hello</main>" in text
    assert "sidebar-led" in text
    # URLs/paths recorded but explicitly unreadable
    assert "CANNOT open these" in text
    assert "x.ts.net/wf.html" in text
    assert "/srv/.../a.png" in text


def test_render_prd_inlines_body_and_marks_doc_link_unreadable():
    text = sc_review.render_prd_evidence(
        {
            "raw_user_intent": "let trips auto-file receipts",
            "prd_body": "## Scope\nIn: parsing. Out: refunds.",
            "credential_matrix": "supabase: provisioned_fluxcreds_handle",
            "google_doc_url": "https://docs.google.com/d/abc",
        }
    )
    assert "auto-file receipts" in text
    assert "In: parsing" in text
    assert "CANNOT open these" in text
    assert "docs.google.com/d/abc" in text


def test_render_returns_none_when_empty():
    assert sc_review.render_wireframe_evidence(None) is None
    assert sc_review.render_wireframe_evidence({}) is None
    assert sc_review.render_prd_evidence(None) is None


def test_excerpt_caps_long_text():
    big = {"prd_body": "x" * 9000}
    text = sc_review.render_prd_evidence(big)
    assert "[truncated]" in text
    assert len(text) < 9000


# ---- prompt shape ------------------------------------------------------------

def test_build_sc_messages_demands_every_contract_field():
    msgs = sc_review.build_sc_messages("wireframe", "evidence")
    system = msgs[0]["content"]
    for field in sc_review.SC_REQUIRED_FIELDS:
        assert field in system, f"prompt omits required field {field}"
    # v18 verdict vocabulary + the right rubric
    for v in sc_review.SC_PASS_VERDICTS + sc_review.SC_BLOCK_VERDICTS:
        assert v in system
    assert "wireframe-rubric.md" in system
    assert msgs[1]["content"] == "evidence"


# ---- parsing (fail-closed) ---------------------------------------------------

def test_parse_sc_reply_variants():
    assert sc_review.parse_sc_reply({"verdict": "approved"}) == {"verdict": "approved"}
    assert sc_review.parse_sc_reply('prefix {"verdict": "approved"} suffix') == {"verdict": "approved"}
    assert sc_review.parse_sc_reply("not json") is None
    assert sc_review.parse_sc_reply("") is None
    assert sc_review.parse_sc_reply("[1,2,3]") is None  # array, not object


# ---- payload assembly does NOT fabricate -------------------------------------

def test_build_payload_stamps_type_without_fabricating_fields():
    payload = sc_review.build_sc_review_payload("t_1", "PRD", {"verdict": "approved"})
    assert payload["task_id"] == "t_1"
    assert payload["review_type"] == "prd"
    # it must NOT invent a scorecard or the list fields (hollow stays hollow)
    assert "scorecard" not in payload
    assert "blocking_issues" not in payload


# ---- round-trip through the REAL contract enforcer ---------------------------

def test_conforming_review_passes_real_contract():
    chat = _ChatStub(json.dumps(_conforming_reply()))
    payload = sc_review.review_sc(
        "evidence", review_type="wireframe", task_id="t_1", chat=chat, model="m"
    )
    assert payload["review_type"] == "wireframe"
    # the REAL enforcer accepts it ⇒ a genuine scorecard clears the gate
    assert enforce_supreme_court_contract(payload, "wireframe") is None


def test_hollow_pass_is_rejected_by_real_contract():
    # the exact 557 failure: a bare "PASS" with no scorecard
    chat = _ChatStub(json.dumps({"verdict": "approved"}))
    payload = sc_review.review_sc(
        "evidence", review_type="wireframe", task_id="t_1", chat=chat, model="m"
    )
    assert payload["review_type"] == "wireframe"  # stamped...
    violation = enforce_supreme_court_contract(payload, "wireframe")
    assert violation  # ...so the hollow verdict is caught, not cleared
    assert "hollow" in violation.lower()


def test_failed_call_is_fail_closed():
    chat = _ChatStub(None, ok=False, error="429")
    payload = sc_review.review_sc(
        "evidence", review_type="prd", task_id="t_1", chat=chat, model="m"
    )
    assert payload["review_type"] == "prd"
    assert payload["verdict"] in sc_review.SC_BLOCK_VERDICTS
    # a fail-closed payload still has no scorecard ⇒ contract would reject it too
    assert enforce_supreme_court_contract(payload, "prd")


def test_unparseable_reply_is_fail_closed():
    chat = _ChatStub("the design looks great, PASS")
    payload = sc_review.review_sc(
        "evidence", review_type="wireframe", task_id="t_1", chat=chat, model="m"
    )
    assert payload["verdict"] in sc_review.SC_BLOCK_VERDICTS
    assert enforce_supreme_court_contract(payload, "wireframe")
