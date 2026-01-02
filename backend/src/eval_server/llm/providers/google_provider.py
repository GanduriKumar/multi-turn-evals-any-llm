from __future__ import annotations

import os
from typing import Any, Dict, List

from ..provider_registry import LLMProvider, register_provider


class GoogleProvider(LLMProvider):
    """Scaffold adapter for Google Gemini (dry-run)."""

    def __init__(self) -> None:
        self._name = "google"
        self._version = "v1"
        self._api_key = None
        self._model = "gemini-1.5-pro"
        self._prefix = "DRYRUN:"

    def initialize(self, **kwargs: Any) -> None:
        self._api_key = kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY")
        self._model = kwargs.get("model", self._model)
        self._prefix = str(kwargs.get("prefix", self._prefix))

    def generate(self, prompt: str, context: List[str] | None = None) -> str:
        ctx_part = " | ".join(context or [])
        if ctx_part:
            return f"{self._prefix} google[{self._model}] [{ctx_part}] -> {prompt}"
        return f"{self._prefix} google[{self._model}] {prompt}"

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self._name,
            "version": self._version,
            "model": self._model,
            "dry_run": True,
        }


register_provider("google", GoogleProvider)
