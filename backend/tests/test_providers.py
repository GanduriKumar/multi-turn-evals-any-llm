import asyncio
import os
import types

import pytest

from providers.registry import ProviderRegistry
from providers.types import ProviderRequest

@pytest.mark.asyncio
async def test_registry_init():
    r = ProviderRegistry()
    assert r.get("ollama") is not None
    # gemini may be disabled when key absent
    _ = r.get("gemini")

@pytest.mark.asyncio
async def test_ollama_mock(monkeypatch):
    r = ProviderRegistry()
    ollama = r.get("ollama")

    async def fake_chat(self, req):
        return types.SimpleNamespace(ok=True, content="hello", latency_ms=1, provider_meta={})

    monkeypatch.setattr(type(ollama), "chat", fake_chat, raising=True)

    resp = await ollama.chat(ProviderRequest(model="llama3.2:2b", messages=[{"role": "user", "content": "hi"}], metadata={}))
    assert resp.ok
    assert resp.content == "hello"

@pytest.mark.asyncio
async def test_gemini_disabled_without_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    r = ProviderRegistry()
    gemini = r.get("gemini")
    resp = await gemini.chat(ProviderRequest(model="gemini-2.5", messages=[{"role": "user", "content": "hi"}], metadata={}))
    assert not resp.ok
    assert "disabled" in (resp.error or "").lower()
