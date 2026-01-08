from __future__ import annotations
import os
from typing import Dict

try:
    from .ollama import OllamaProvider
    from .gemini import GeminiProvider
    from .openai import OpenAIProvider
except ImportError:
    from providers.ollama import OllamaProvider
    from providers.gemini import GeminiProvider
    from providers.openai import OpenAIProvider

class ProviderRegistry:
    def __init__(self) -> None:
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self._ollama = OllamaProvider(self.ollama_host)
        self._gemini = GeminiProvider(self.google_api_key)
        self._openai = OpenAIProvider(self.openai_api_key)

    @property
    def gemini_enabled(self) -> bool:
        return self._gemini.enabled

    def get(self, provider: str):
        if provider == "ollama":
            return self._ollama
        if provider == "gemini":
            return self._gemini
        if provider == "openai":
            return self._openai
        raise KeyError(f"Unknown provider: {provider}")
