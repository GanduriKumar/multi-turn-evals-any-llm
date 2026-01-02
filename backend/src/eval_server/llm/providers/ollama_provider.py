from __future__ import annotations

import os
from typing import Any, Dict, List

from ..provider_registry import LLMProvider, register_provider


class OllamaProvider(LLMProvider):
    """Scaffold adapter for local Ollama (dry-run)."""

    def __init__(self) -> None:
        self._name = "ollama"
        self._version = "v1"
        self._model = "llama3"
        self._host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._prefix = "DRYRUN:"

    def initialize(self, **kwargs: Any) -> None:
        self._model = kwargs.get("model", self._model)
        self._host = kwargs.get("host", self._host)
        self._prefix = str(kwargs.get("prefix", self._prefix))

    def generate(self, prompt: str, context: List[str] | None = None) -> str:
        ctx_part = " | ".join(context or [])
        if ctx_part:
            return f"{self._prefix} ollama[{self._model}@{self._host}] [{ctx_part}] -> {prompt}"
        return f"{self._prefix} ollama[{self._model}@{self._host}] {prompt}"

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self._name,
            "version": self._version,
            "model": self._model,
            "host": self._host,
            "dry_run": True,
        }


register_provider("ollama", OllamaProvider)
