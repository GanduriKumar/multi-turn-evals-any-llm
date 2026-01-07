from __future__ import annotations
import os
import time
from typing import Dict, Any, List
import httpx

from providers.types import ProviderRequest, ProviderResponse

GEMINI_API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

class GeminiProvider:
    def __init__(self, api_key: str | None) -> None:
        self.api_key = api_key

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    async def chat(self, req: ProviderRequest) -> ProviderResponse:
        if not self.enabled:
            return ProviderResponse(False, "", 0, {}, error="Gemini disabled: missing GOOGLE_API_KEY")
        t0 = time.perf_counter()
        url = GEMINI_API.format(model=req.model, key=self.api_key)
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "\n".join([m.get("content", "") for m in req.messages])}]
                }
            ]
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                r = await client.post(url, json=payload)
                latency_ms = int((time.perf_counter() - t0) * 1000)
                if r.status_code != 200:
                    return ProviderResponse(False, "", latency_ms, {"status": r.status_code}, error=r.text)
                data = r.json()
                text = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                )
                return ProviderResponse(True, text, latency_ms, {"candidates": len(data.get("candidates", []))})
            except Exception as e:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                return ProviderResponse(False, "", latency_ms, {}, error=str(e))
