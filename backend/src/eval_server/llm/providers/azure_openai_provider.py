from __future__ import annotations

import os
from typing import Any, Dict, List

from ..provider_registry import LLMProvider, register_provider


class AzureOpenAIProvider(LLMProvider):
    """Scaffold adapter for Azure OpenAI (dry-run)."""

    def __init__(self) -> None:
        self._name = "azure-openai"
        self._version = "v1"
        self._api_key = None
        self._endpoint = None
        self._deployment = "gpt-4o-mini"
        self._prefix = "DRYRUN:"

    def initialize(self, **kwargs: Any) -> None:
        self._api_key = kwargs.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY")
        self._endpoint = kwargs.get("endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT")
        self._deployment = kwargs.get("deployment", self._deployment)
        self._prefix = str(kwargs.get("prefix", self._prefix))

    def generate(self, prompt: str, context: List[str] | None = None) -> str:
        ctx_part = " | ".join(context or [])
        if ctx_part:
            return f"{self._prefix} azure-openai[{self._deployment}] [{ctx_part}] -> {prompt}"
        return f"{self._prefix} azure-openai[{self._deployment}] {prompt}"

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self._name,
            "version": self._version,
            "deployment": self._deployment,
            "dry_run": True,
        }


register_provider("azure", AzureOpenAIProvider)
