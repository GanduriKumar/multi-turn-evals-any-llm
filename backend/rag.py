from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _snip(t: str, n: int = 300) -> str:
    t = (t or "").replace("\n", " ").strip()
    return t if len(t) <= n else (t[: n - 1] + "…")


@dataclass
class RAGEntry:
    text: str
    meta: Dict[str, Any]


class SimpleRAGIndex:
    """
    Lightweight in-memory RAG index with optional embedding cache.
    - entries: list of RAGEntry
    - vectors: list of embedding vectors aligning with entries
    """

    def __init__(self, entries: List[RAGEntry]) -> None:
        self.entries = entries
        self.vectors: Optional[List[List[float]]] = None

    async def ensure_embeddings(self, embedder) -> None:
        if self.vectors is not None:
            return
        texts = [e.text for e in self.entries]
        if not texts:
            self.vectors = []
            return
        try:
            self.vectors = await embedder.embed(texts)
        except Exception as e:
            # propagate to caller; they may fallback to keyword
            raise e

    async def search(self, query: str, embedder, top_k: int = 6) -> List[Tuple[RAGEntry, float]]:
        if not query or not self.entries:
            return []
        await self.ensure_embeddings(embedder)
        if self.vectors is None or len(self.vectors) != len(self.entries):
            return []
        # Embed query
        qv = (await embedder.embed([query]))[0]
        # cosine similarity
        from .embeddings.ollama_embed import OllamaEmbeddings  # local import to avoid cycles
        sims: List[Tuple[int, float]] = []
        for i, v in enumerate(self.vectors):
            try:
                s = OllamaEmbeddings.cosine(qv, v)
            except Exception:
                s = 0.0
            sims.append((i, s))
        sims.sort(key=lambda x: x[1], reverse=True)
        out: List[Tuple[RAGEntry, float]] = []
        for i, s in sims[: max(1, top_k)]:
            out.append((self.entries[i], s))
        return out


def build_dataset_index(dataset: Dict[str, Any], restrict_conversation_id: Optional[str] = None, max_turns_per_conv: int = 12) -> SimpleRAGIndex:
    entries: List[RAGEntry] = []
    convs = (dataset or {}).get("conversations") or []
    for c in convs:
        cid = str(c.get("conversation_id"))
        if restrict_conversation_id and cid != str(restrict_conversation_id):
            continue
        title = c.get("conversation_title") or c.get("conversation_slug") or cid
        dom = (c.get("metadata") or {}).get("domain") or c.get("domain")
        beh = (c.get("metadata") or {}).get("behavior") or c.get("behavior")
        header = " | ".join([s for s in [f"Title: {title}", f"Domain: {dom}", f"Behavior: {beh}"] if s and not s.endswith(": None")])
        if header:
            entries.append(RAGEntry(text=header, meta={"section": "header", "conversation_id": cid}))
        turns = (c.get("turns") or [])[: max_turns_per_conv]
        for t in turns:
            idx = int(t.get("turn_index", 0))
            u = t.get("text") or t.get("user") or t.get("user_text") or t.get("user_prompt_snippet") or ""
            a = t.get("assistant") or t.get("assistant_text") or t.get("assistant_output_snippet") or ""
            if u:
                entries.append(RAGEntry(text=f"[Conv {cid} · {idx+1}] User: {_snip(u)}", meta={"section": "user", "conversation_id": cid, "turn_index": idx}))
            if a:
                entries.append(RAGEntry(text=f"[Conv {cid} · {idx+1}] Assistant: {_snip(a)}", meta={"section": "assistant", "conversation_id": cid, "turn_index": idx}))
    return SimpleRAGIndex(entries)


def build_report_index(results: Dict[str, Any], max_turns_per_conv: int = 12) -> SimpleRAGIndex:
    entries: List[RAGEntry] = []
    convs = (results or {}).get("conversations") or []
    for c in convs:
        cid = str(c.get("conversation_id"))
        title = c.get("conversation_title") or c.get("conversation_slug") or cid
        summ = c.get("summary") or {}
        header = f"Report: {title} — pass={bool(summ.get('conversation_pass'))} failed_turns={summ.get('failed_turns_count')}"
        if summ.get("failed_metrics"):
            header += f"; failed_metrics={', '.join((summ.get('failed_metrics') or [])[:8])}"
        entries.append(RAGEntry(text=header, meta={"section": "summary", "conversation_id": cid}))
        turns = (c.get("turns") or [])[: max_turns_per_conv]
        for t in turns:
            idx = int(t.get("turn_index", 0))
            u = t.get("user_prompt_snippet") or t.get("user") or t.get("text") or ""
            a = t.get("assistant_output_snippet") or t.get("assistant") or ""
            m = t.get("metrics") or {}
            changed = ",".join(sorted([k for k, v in m.items() if isinstance(v, dict) and v.get("changed")]))
            meta_bits = [f"turn_pass={t.get('turn_pass')}"]
            if changed:
                meta_bits.append(f"metrics_changed={changed}")
            desc = f"[Conv {cid} · {idx+1}] {'; '.join(meta_bits)}"
            if u:
                desc += f"\nU: {_snip(u, 220)}"
            if a:
                desc += f"\nA: {_snip(a, 220)}"
            entries.append(RAGEntry(text=desc, meta={"section": "turn", "conversation_id": cid, "turn_index": idx}))
    return SimpleRAGIndex(entries)
