import pytest


def test_factory_creates_cheaperinference_provider():
    from src.core.llm.factory import create_llm_provider
    from src.core.llm.providers.cheaperinference import CheaperInferenceProvider

    provider = create_llm_provider(
        "cheaperinference",
        model="gpt-5.4-mini",
        api_key="ci-test-key",
    )

    assert isinstance(provider, CheaperInferenceProvider)
    assert provider.model == "gpt-5.4-mini"
    assert provider.api_endpoint == "https://api.cheaperinference.com/v1/chat/completions"
    assert provider.api_key == "ci-test-key"


def test_factory_requires_cheaperinference_api_key(monkeypatch):
    import src.core.llm.factory as factory_module
    from src.core.llm.factory import create_llm_provider

    monkeypatch.delenv("CHEAPERINFERENCE_API_KEY", raising=False)
    monkeypatch.setattr(factory_module, "CHEAPERINFERENCE_API_KEY", "")

    with pytest.raises(ValueError, match="CHEAPERINFERENCE_API_KEY"):
        create_llm_provider("cheaperinference", model="gpt-5.4-mini")


def test_cheaperinference_provider_fallback_models_without_key():
    from src.core.llm.providers.cheaperinference import CheaperInferenceProvider

    provider = CheaperInferenceProvider(api_key="", model="gpt-5.4-mini")
    models = provider._get_fallback_models()

    model_ids = {model["id"] for model in models}
    assert "gpt-5.4-mini" in model_ids
    assert "claude-sonnet-5" in model_ids
