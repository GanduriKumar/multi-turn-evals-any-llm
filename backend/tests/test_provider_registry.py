from __future__ import annotations

from typing import Any, List

import pytest

from eval_server.llm.provider_registry import (
    LLMProvider,
    create_provider,
    get_provider_class,
    list_providers,
    register_provider,
)


def test_dummy_provider_registration_and_generation():
    # Import ensures auto-registration of DummyProvider under name 'dummy'
    from eval_server.llm.providers import dummy_provider  # noqa: F401

    # Ensure the provider class is retrievable
    cls = get_provider_class("dummy")
    provider = create_provider("dummy", prefix="TEST:")

    assert provider.metadata()["name"] == "dummy"
    # Deterministic output without context
    out1 = provider.generate("Hello")
    assert out1 == "TEST: Hello"
    # Deterministic output with context
    out2 = provider.generate("Hello", context=["A", "B"]) 
    assert out2 == "TEST: [A | B] -> Hello"


def test_duplicate_registration_raises_error():
    class TempProvider(LLMProvider):
        def initialize(self, **kwargs: Any) -> None:
            pass

        def generate(self, prompt: str, context: List[str] | None = None) -> str:
            return prompt

        def metadata(self):
            return {"name": "temp"}

    # First registration should succeed
    register_provider("temp", TempProvider)

    # Second registration with same name should fail
    with pytest.raises(ValueError):
        register_provider("temp", TempProvider)
