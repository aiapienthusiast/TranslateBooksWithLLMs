"""
Cheaper Inference LLM provider.

Cheaper Inference exposes an OpenAI-compatible LLM API that serves text
models from several labs behind one endpoint.
"""

from typing import List, Optional, Union

from .openai import OpenAICompatibleProvider


class CheaperInferenceProvider(OpenAICompatibleProvider):
    """OpenAI-compatible provider for Cheaper Inference."""

    API_URL = "https://api.cheaperinference.com/v1/chat/completions"
    MODELS_URL = "https://api.cheaperinference.com/v1/models"

    FALLBACK_MODELS = [
        "gpt-5.4-mini",
        "gpt-5.4",
        "claude-sonnet-5",
        "gemini-3.1-pro",
        "deepseek-v4-flash",
    ]

    def __init__(
        self,
        api_key: Union[str, List[str]],
        model: str = "gpt-5.4-mini",
        api_endpoint: Optional[str] = None,
    ):
        super().__init__(
            api_endpoint=api_endpoint or self.API_URL,
            model=model,
            api_key=api_key,
            context_window=128000,
            provider_name="cheaperinference",
        )

    async def get_available_models(self) -> list:
        """Fetch available Cheaper Inference text models from the OpenAI-compatible catalog."""
        if not self.api_key:
            return self._get_fallback_models()

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            }
            client = await self._get_client()
            response = await client.get(self.MODELS_URL, headers=headers, timeout=15)
            response.raise_for_status()

            models = []
            for model in response.json().get("data", []):
                model_id = model.get("id", "")
                if not model_id:
                    continue
                # The catalog also lists image and video models.
                if model.get("type", "text") != "text":
                    continue
                models.append({
                    "id": model_id,
                    "name": model_id,
                    "owned_by": model.get("owned_by", "cheaperinference"),
                    "context_length": model.get("context_length"),
                })

            models.sort(key=lambda item: item["name"].lower())
            return models or self._get_fallback_models()

        except Exception as exc:
            print(f"Warning: Failed to fetch Cheaper Inference models: {exc}")
            return self._get_fallback_models()

    def _get_fallback_models(self) -> list:
        return [
            {"id": model, "name": model, "owned_by": "cheaperinference", "context_length": 128000}
            for model in self.FALLBACK_MODELS
        ]
