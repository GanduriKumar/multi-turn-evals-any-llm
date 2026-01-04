from __future__ import annotations

import os
from typing import Any, Dict, List

from ..provider_registry import LLMProvider, register_provider


class OpenAIProvider(LLMProvider):
    """Scaffold adapter for OpenAI models (dry-run by default)."""

    def __init__(self) -> None:
        self._name = "openai"
        self._version = "v1"
        self._api_key = None
        self._model_id = "gpt-4o-mini"
        self._base_url = None
        self._prefix = "DRYRUN:"

    def initialize(self, **kwargs: Any) -> None:
        self._api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        self._model_id = kwargs.get("model_id", self._model_id)
        self._base_url = kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL")
        self._prefix = str(kwargs.get("prefix", self._prefix))

    def generate(self, prompt: str, context: List[str] | None = None) -> str:
        if not self._api_key:
            return self._dry_run_generate(prompt, context)

        try:
            import openai
        except ImportError:
            return self._dry_run_generate(prompt, context)

        client = openai.OpenAI(api_key=self._api_key, base_url=self._base_url)
        messages = []
        
        if context:
            for line in context:
                role = "system"
                content = line
                if line.upper().startswith("USER:"):
                    role = "user"
                    content = line[5:].strip()
                elif line.upper().startswith("ASSISTANT:"):
                    role = "assistant"
                    content = line[10:].strip()
                elif line.upper().startswith("SYSTEM:"):
                    role = "system"
                    content = line[7:].strip()
                
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=self._model_id,
                messages=messages,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            # Fallback or re-raise? Re-raising is better for "Real" provider.
            raise RuntimeError(f"OpenAI API error: {e}")

    def _dry_run_generate(self, prompt: str, context: List[str] | None = None) -> str:
        # Dry-run deterministic response to avoid external calls in tests
        ctx_part = " | ".join(context or [])
        if ctx_part:
            return f"{self._prefix} openai[{self._model_id}] [{ctx_part}] -> {prompt}"
        return f"{self._prefix} openai[{self._model_id}] {prompt}"

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self._name,
            "version": self._version,
            "model_id": self._model_id,
            "dry_run": True,
        }


register_provider("openai", OpenAIProvider)
