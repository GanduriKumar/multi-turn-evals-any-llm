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
        """Robust hybrid retrieval: embeddings + keyword scoring with field-aware boosts.
        Designed to avoid misses for a wide range of report/dataset questions.
        """
        if not query or not self.entries:
            return []
        await self.ensure_embeddings(embedder)
        if self.vectors is None or len(self.vectors) != len(self.entries):
            return []
        # Embed query
        qv = (await embedder.embed([query]))[0]
        from .embeddings.ollama_embed import OllamaEmbeddings  # local import to avoid cycles
        sims: List[Tuple[int, float]] = []
        for i, v in enumerate(self.vectors):
            try:
                s = OllamaEmbeddings.cosine(qv, v)
            except Exception:
                s = 0.0
            sims.append((i, s))
        sims.sort(key=lambda x: x[1], reverse=True)

        # Keyword scoring
        import re
        def tokenize(text: str) -> list[str]:
            toks = re.findall(r"[a-zA-Z0-9_]+", text.lower())
            stop = {"the","a","an","and","or","is","to","of","in","on","for","with","by","at","as","it","this","that"}
            return [t for t in toks if t not in stop]

        ql = query.lower()
        q_tokens = tokenize(query)
        metric_terms = {"exact","semantic","consistency","adherence","hallucination","bleu","rouge","f1","precision","recall"}
        wants_metric = ("metric" in ql) or any(m in ql for m in metric_terms)
        wants_domain = ("domain" in ql)
        wants_behavior = ("behavior" in ql)
        status_query = any(w in ql for w in ("pass","passed","conversation_pass","fail","failed","status"))

        # Precompute document frequencies for IDF weighting
        docs_tokens: list[list[str]] = []
        for e in self.entries:
            docs_tokens.append(tokenize(e.text))
        from collections import Counter
        N_docs = max(1, len(docs_tokens))
        df_counter: Counter[str] = Counter()
        for toks in docs_tokens:
            df_counter.update(set(toks))
        def idf(term: str) -> float:
            # smooth IDF
            return max(0.0, __import__("math").log(1.0 + N_docs / (1.0 + df_counter.get(term, 0))))

        kw_scores: list[float] = []
        qset = set(q_tokens)
        q_idf_sum = sum(idf(t) for t in qset) or 1.0
        for idx, e in enumerate(self.entries):
            t = e.text
            etoks = docs_tokens[idx]
            eset = set(etoks)
            # IDF-weighted overlap
            inter = qset & eset
            overlap = sum(idf(tok) for tok in inter) / q_idf_sum
            phrase_boost = 0.2 if t.lower().find(ql.strip()) >= 0 and len(ql.strip()) >= 4 else 0.0
            field_boost = 0.0
            sec = (e.meta or {}).get("section")
            tl = t.lower()
            if status_query and sec == "summary":
                field_boost += 0.1
            if wants_metric and ("failed_metrics=" in tl or "metrics_changed=" in tl or sec == "summary"):
                field_boost += 0.1
            if wants_domain and "domain:" in tl:
                field_boost += 0.1
            if wants_behavior and "behavior:" in tl:
                field_boost += 0.1
            kw_scores.append(min(1.0, overlap + phrase_boost + field_boost))

        # Combine scores
        # Normalize cosine from [-1,1] to [0,1]
        sims01 = [(i, max(0.0, min(1.0, (s + 1.0) / 2.0))) for i, s in sims]
        # Dynamic weighting
        alpha, beta = 0.7, 0.3
        if len(q_tokens) <= 2:
            alpha, beta = 0.5, 0.5
        if sims01 and sims01[0][1] < 0.1:
            alpha, beta = 0.4, 0.6

        scored: list[Tuple[int, float]] = []
        for i, s in sims01:
            kw = kw_scores[i] if i < len(kw_scores) else 0.0
            scored.append((i, alpha * s + beta * kw))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Ensure critical keyword hits are present for status queries
        keyword_indices: List[int] = []
        if status_query:
            wants_pass = ("passed" in ql) or ("pass" in ql and "fail" not in ql) or ("conversation_pass=true" in ql)
            wants_fail = ("fail" in ql or "failed" in ql or "conversation_pass=false" in ql) and not wants_pass
            for i, e in enumerate(self.entries):
                tl = e.text.lower()
                if wants_pass and ("status: passed" in tl or "conversation_pass=true" in tl or "passed=true" in tl or "failed turns: 0" in tl):
                    keyword_indices.append(i)
                elif wants_fail and ("status: failed" in tl or "conversation_pass=false" in tl or "failed=true" in tl):
                    keyword_indices.append(i)

        picked: list[int] = []
        for i, _s in scored:
            if i not in picked:
                picked.append(i)
            if len(picked) >= max(1, top_k):
                break
        for i in keyword_indices:
            if i not in picked:
                picked.append(i)
            if len(picked) >= max(1, top_k):
                break

        out: List[Tuple[RAGEntry, float]] = []
        score_map = {i: s for i, s in scored}
        for i in picked[: max(1, top_k)]:
            out.append((self.entries[i], score_map.get(i, 0.5)))
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
        slug = c.get("conversation_slug")
        dom = (c.get("metadata") or {}).get("domain") or c.get("domain")
        beh = (c.get("metadata") or {}).get("behavior") or c.get("behavior")
        summ = c.get("summary") or {}
        passed = bool(summ.get("conversation_pass"))
        status = "passed" if passed else "failed"
        failed_turns = summ.get("failed_turns_count")
        # Enrich header with explicit, searchable synonyms
        header_bits = [
            f"Report: {title}",
            f"ConversationID: {cid}",
            f"Slug: {slug}" if slug else None,
            f"Domain: {dom}" if dom else None,
            f"Behavior: {beh}" if beh else None,
            f"pass={passed}",
            f"failed_turns={failed_turns}",
            f"Status: {status}",
            f"conversation_pass={'true' if passed else 'false'}",
            f"passed={'true' if passed else 'false'}",
            f"failed={'false' if passed else 'true'}",
        ]
        header = "; ".join([h for h in header_bits if h])
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
