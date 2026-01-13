from __future__ import annotations
import time
from typing import List
import httpx
import os

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

class OllamaEmbeddings:
    def __init__(self, host: str | None = None) -> None:
        self.base_url = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Call Ollama's embeddings API. Ollama expects a single 'prompt' string per request.
        We support batch by issuing one request per text.
        """
        url = f"{self.base_url}/api/embeddings"
        out: List[List[float]] = []
        # minimal retry per item
        import asyncio
        for t in texts:
            last_err: Exception | None = None
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        r = await client.post(url, json={"model": EMBED_MODEL, "prompt": t})
                        r.raise_for_status()
                        data = r.json()
                        if isinstance(data, dict) and "embedding" in data and isinstance(data["embedding"], list):
                            out.append(data["embedding"]) 
                            break
                        # Some servers might return {'embeddings': [[...]]}
                        if isinstance(data, dict) and "embeddings" in data and isinstance(data["embeddings"], list):
                            emb = data["embeddings"][0] if data["embeddings"] else []
                            out.append(emb)
                            break
                        raise RuntimeError("unexpected embeddings response")
                except Exception as e:
                    last_err = e
                    await asyncio.sleep(0.2 * (attempt + 1))
            else:
                raise RuntimeError(f"embedding failed after retries: {last_err}")
        return out

    @staticmethod
    def cosine(a: List[float], b: List[float]) -> float:
        import math
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(y*y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
