from __future__ import annotations
import time
from typing import Dict, Any, List
import httpx

from providers.types import ProviderRequest, ProviderResponse

class OllamaProvider:
    def __init__(self, host: str = "http://localhost:11434") -> None:
        self.base_url = host.rstrip("/")

    async def chat(self, req: ProviderRequest) -> ProviderResponse:
        t0 = time.perf_counter()
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": req.model,
            "messages": req.messages,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                r = await client.post(url, json=payload)
                latency_ms = int((time.perf_counter() - t0) * 1000)
                if r.status_code != 200:
                    return ProviderResponse(False, "", latency_ms, {"status": r.status_code}, error=r.text)
                data = r.json()
                content = data.get("message", {}).get("content", "")
                meta = {k: data.get(k) for k in ("total_duration", "load_duration", "prompt_eval_count", "eval_count")}
                return ProviderResponse(True, content, latency_ms, meta)
            except Exception as e:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                return ProviderResponse(False, "", latency_ms, {}, error=str(e))
