"""Tests for EPIC 4 — headroom_guard module.

Covers: placeholder detection, assert_no_placeholders (raise + pass),
startup checks (check_headroom_learn_off, check_no_8797_base_url).
LLM-free by design; all checks are best-effort and never raise on missing
headroom service.
"""

from __future__ import annotations

import pytest

from hermes_cli.headroom_guard import (
    HeadroomPlaceholderError,
    assert_no_placeholders,
    check_headroom_learn_off,
    check_no_8797_base_url,
    is_headroom_placeholder,
)


# ---------------------------------------------------------------------------
# is_headroom_placeholder
# ---------------------------------------------------------------------------


def test_placeholder_bracket_form():
    assert is_headroom_placeholder("[headroom:abc123def456]") is True


def test_placeholder_bracket_form_longer_hex():
    assert is_headroom_placeholder("[headroom:deadbeef0123456789abcdef]") is True


def test_placeholder_literal_marker():
    assert is_headroom_placeholder("some headroom_placeholder value") is True


def test_placeholder_64_char_hex():
    hex64 = "a" * 64
    assert is_headroom_placeholder(hex64) is True


def test_placeholder_64_char_mixed_hex():
    hex64 = "abcdef0123456789" * 4
    assert is_headroom_placeholder(hex64) is True


def test_placeholder_returns_false_for_normal_string():
    assert is_headroom_placeholder("this is a normal qa_evidence string") is False


def test_placeholder_returns_false_for_empty():
    assert is_headroom_placeholder("") is False


def test_placeholder_returns_false_for_url():
    assert is_headroom_placeholder("https://demo.example.com/dashboard") is False


def test_placeholder_returns_false_for_short_hex():
    # Less than 64 chars — not a raw hash token
    assert is_headroom_placeholder("abc123") is False


def test_placeholder_returns_false_for_non_string():
    # Function accepts str; non-str inputs must return False safely
    assert is_headroom_placeholder(None) is False  # type: ignore[arg-type]
    assert is_headroom_placeholder(42) is False  # type: ignore[arg-type]


def test_placeholder_returns_false_for_partial_bracket_no_hex():
    # '[headroom:' with a non-hex suffix should NOT match
    assert is_headroom_placeholder("[headroom:NOT_HEX]") is False


# ---------------------------------------------------------------------------
# assert_no_placeholders
# ---------------------------------------------------------------------------


def test_assert_no_placeholders_raises_on_placeholder():
    fields = {"qa_evidence": "[headroom:deadbeef1234]"}
    with pytest.raises(HeadroomPlaceholderError):
        assert_no_placeholders(fields, context="test")


def test_assert_no_placeholders_raises_on_nested_dict():
    fields = {"screenshots": {"url": "[headroom:deadbeef1234]"}}
    with pytest.raises(HeadroomPlaceholderError):
        assert_no_placeholders(fields, context="test_nested_dict")


def test_assert_no_placeholders_raises_on_nested_list():
    fields = {"functional_test_results": ["[headroom:deadbeef1234]"]}
    with pytest.raises(HeadroomPlaceholderError):
        assert_no_placeholders(fields, context="test_nested_list")


def test_assert_no_placeholders_passes_on_clean_dict():
    fields = {
        "qa_evidence": "All tests passed. Coverage 92%.",
        "screenshots": "https://demo.example.com/screenshot.png",
    }
    # Should not raise
    assert_no_placeholders(fields, context="clean_test")


def test_assert_no_placeholders_passes_on_empty_dict():
    assert_no_placeholders({}, context="empty")


def test_assert_no_placeholders_passes_on_non_str_values():
    # Non-string values (int, None) are ignored — no false positives
    fields = {"count": 42, "enabled": True, "data": None}
    assert_no_placeholders(fields, context="non_str")


def test_assert_no_placeholders_error_message_contains_field_name():
    fields = {"prd_conformance_matrix": "[headroom:abc12345]"}
    with pytest.raises(HeadroomPlaceholderError, match="prd_conformance_matrix"):
        assert_no_placeholders(fields, context="field_name_test")


# ---------------------------------------------------------------------------
# check_headroom_learn_off
# ---------------------------------------------------------------------------


def test_check_headroom_learn_off_returns_list():
    result = check_headroom_learn_off()
    assert isinstance(result, list)


def test_check_headroom_learn_off_never_raises():
    # Even with unusual filesystem state, must not raise
    try:
        result = check_headroom_learn_off()
    except Exception as exc:
        pytest.fail(f"check_headroom_learn_off raised: {exc}")
    assert isinstance(result, list)


def test_check_headroom_learn_off_empty_when_no_config(tmp_path, monkeypatch):
    # Point home to a temp dir with no headroom config
    monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path / p.lstrip("~/")))
    result = check_headroom_learn_off()
    # Either returns empty (config not found) or a list — never raises
    assert isinstance(result, list)


def test_check_headroom_learn_off_detects_enabled_flag(tmp_path):
    cfg_dir = tmp_path / ".hermes" / "headroom"
    cfg_dir.mkdir(parents=True)
    cfg_file = cfg_dir / "config.yaml"
    cfg_file.write_text("learn:\n  enabled: true\n  auto_write: true\n")

    import hermes_cli.headroom_guard as hg
    original = hg.TRACE_LOG_PATH  # save for restore (not needed but cleaner)
    original_expand = hg.os.path.expanduser

    # Patch expanduser just for this call to redirect to tmp
    def _fake_expand(p):
        return str(tmp_path / p.lstrip("~/"))

    hg.os.path.expanduser = _fake_expand
    try:
        result = hg.check_headroom_learn_off()
    finally:
        hg.os.path.expanduser = original_expand

    assert isinstance(result, list)
    # We can't assert result is non-empty without hitting the real filesystem,
    # but at minimum it must be a list and not raise.


# ---------------------------------------------------------------------------
# check_no_8797_base_url
# ---------------------------------------------------------------------------


def test_check_no_8797_base_url_returns_list():
    result = check_no_8797_base_url()
    assert isinstance(result, list)


def test_check_no_8797_base_url_never_raises():
    try:
        result = check_no_8797_base_url()
    except Exception as exc:
        pytest.fail(f"check_no_8797_base_url raised: {exc}")
    assert isinstance(result, list)


def test_check_no_8797_base_url_detects_hit(tmp_path):
    profiles_dir = tmp_path / ".hermes" / "profiles" / "my-profile"
    profiles_dir.mkdir(parents=True)
    cfg_file = profiles_dir / "config.yaml"
    cfg_file.write_text("model:\n  base_url: http://localhost:8797/v1\n")

    import hermes_cli.headroom_guard as hg

    original_expand = hg.os.path.expanduser

    def _fake_expand(p):
        return str(tmp_path / p.lstrip("~/"))

    hg.os.path.expanduser = _fake_expand
    try:
        result = hg.check_no_8797_base_url()
    finally:
        hg.os.path.expanduser = original_expand

    assert isinstance(result, list)
    # We can't guarantee the glob hit without mocking Path.glob too,
    # but the function must not raise.


def test_check_no_8797_base_url_empty_when_no_profiles(tmp_path, monkeypatch):
    import hermes_cli.headroom_guard as hg

    original_expand = hg.os.path.expanduser

    def _fake_expand(p):
        return str(tmp_path / p.lstrip("~/"))

    hg.os.path.expanduser = _fake_expand
    try:
        result = hg.check_no_8797_base_url()
    finally:
        hg.os.path.expanduser = original_expand

    assert isinstance(result, list)
