from __future__ import annotations
import time
from typing import Dict, Any, List
import httpx

try:
    from .types import ProviderRequest, ProviderResponse
except ImportError:
    from providers.types import ProviderRequest, ProviderResponse


class OpenAIProvider:
    def __init__(self, api_key: str | None) -> None:
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def chat(self, req: ProviderRequest) -> ProviderResponse:
        if not self.enabled:
            return ProviderResponse(False, "", 0, {}, error="OpenAI disabled: missing OPENAI_API_KEY")
        t0 = time.perf_counter()
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": req.model,
            "messages": req.messages,
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                r = await client.post(url, json=payload, headers=headers)
                latency_ms = int((time.perf_counter() - t0) * 1000)
                if r.status_code != 200:
                    return ProviderResponse(False, "", latency_ms, {"status": r.status_code}, error=r.text)
                data = r.json()
                content = (
                    (data.get("choices", [{}])[0] or {})
                    .get("message", {})
                    .get("content", "")
                )
                meta = {
                    "model": data.get("model"),
                    "usage": data.get("usage"),
                }
                return ProviderResponse(True, content, latency_ms, meta)
            except Exception as e:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                return ProviderResponse(False, "", latency_ms, {}, error=str(e))
