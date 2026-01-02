from __future__ import annotations

import os
from typing import Any, Dict, List

from ..provider_registry import LLMProvider, register_provider


class AWSBedrockProvider(LLMProvider):
    """Scaffold adapter for AWS Bedrock (dry-run)."""

    def __init__(self) -> None:
        self._name = "aws-bedrock"
        self._version = "v1"
        self._region = os.getenv("AWS_REGION", "us-east-1")
        self._model_id = "anthropic.claude-3-haiku"
        self._prefix = "DRYRUN:"

    def initialize(self, **kwargs: Any) -> None:
        self._region = kwargs.get("region", self._region)
        self._model_id = kwargs.get("model_id", self._model_id)
        self._prefix = str(kwargs.get("prefix", self._prefix))

    def generate(self, prompt: str, context: List[str] | None = None) -> str:
        ctx_part = " | ".join(context or [])
        if ctx_part:
            return f"{self._prefix} bedrock[{self._model_id}@{self._region}] [{ctx_part}] -> {prompt}"
        return f"{self._prefix} bedrock[{self._model_id}@{self._region}] {prompt}"

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self._name,
            "version": self._version,
            "model_id": self._model_id,
            "region": self._region,
            "dry_run": True,
        }


register_provider("aws", AWSBedrockProvider)
