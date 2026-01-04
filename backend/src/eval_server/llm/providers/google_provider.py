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
        if not self._api_key:
            return self._dry_run_generate(prompt, context)

        try:
            import google.generativeai as genai
        except ImportError:
            return self._dry_run_generate(prompt, context)

        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(self._model)

        history = []
        if context:
            for line in context:
                role = "user"
                content = line
                if line.upper().startswith("USER:"):
                    role = "user"
                    content = line[5:].strip()
                elif line.upper().startswith("ASSISTANT:"):
                    role = "model"
                    content = line[10:].strip()
                elif line.upper().startswith("SYSTEM:"):
                    role = "user"
                    content = f"System instruction: {line[7:].strip()}"
                
                history.append({"role": role, "parts": [content]})

        try:
            chat = model.start_chat(history=history)
            response = chat.send_message(prompt)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Google API error: {e}")

    def _dry_run_generate(self, prompt: str, context: List[str] | None = None) -> str:
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
