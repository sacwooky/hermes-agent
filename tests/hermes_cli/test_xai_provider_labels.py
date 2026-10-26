"""Regression tests for xAI provider label disambiguation."""

from hermes_cli.models import provider_label
from hermes_cli.providers import get_label


def test_xai_oauth_provider_label_is_not_collapsed_to_api_key_label(monkeypatch):
    """The model picker must distinguish xAI API-key and OAuth providers."""
    # The API-key label comes from the models.dev catalog; stub it so the test
    # does not depend on reaching models.dev.
    monkeypatch.setattr("agent.models_dev.fetch_models_dev",
                        lambda *a, **k: {"xai": {"name": "xAI", "env": ["XAI_API_KEY"]}})
    assert get_label("xai") == "xAI"
    assert get_label("xai-oauth") == "xAI Grok OAuth (SuperGrok / Premium+)"
    assert get_label("grok-oauth") == "xAI Grok OAuth (SuperGrok / Premium+)"


def test_xai_oauth_provider_labels_match_canonical_model_labels():
    """Provider helpers should agree on the OAuth display label."""
    assert get_label("xai-oauth") == provider_label("xai-oauth")
