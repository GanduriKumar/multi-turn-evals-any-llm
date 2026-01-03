from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import re

from ..scoring.normalizer import canonicalize_text


@dataclass(frozen=True)
class HallucinationConfig:
    """Configuration for hallucination risk scoring.

    min_jaccard: Minimum token coverage for a sentence to be considered grounded.
    min_tokens: Minimum number of content tokens in an output sentence to evaluate it.
    stopwords: Stopword list used to compute content tokens. If None, a default small set is used.
    """

    min_jaccard: float = 0.6
    min_tokens: int = 2
    stopwords: Optional[Sequence[str]] = None


_DEFAULT_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "of",
    "and",
    "or",
    "to",
    "in",
    "on",
    "for",
    "with",
    "by",
    "as",
    "at",
    "that",
    "this",
    "it",
    "be",
    "from",
    "has",
    "have",
}


_SENT_SPLIT = re.compile(r"[.!?;\n]+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _split_sentences(text: str) -> List[str]:
    t = canonicalize_text(text)
    parts = [p.strip() for p in _SENT_SPLIT.split(t) if p.strip()]
    return parts


def _content_tokens(text: str, stopwords: Optional[Iterable[str]] = None) -> List[str]:
    t = canonicalize_text(text)
    sw = set(stopwords or _DEFAULT_STOPWORDS)
    toks = [m.group(0) for m in _TOKEN_RE.finditer(t)]
    # Filter stopwords and very short tokens (length <= 2)
    return [tok for tok in toks if tok not in sw and len(tok) > 2]


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0


def _best_sentence_overlap(out_sent: str, ctx_sents: List[str], stopwords: Optional[Iterable[str]]) -> Tuple[float, str]:
    """Return best coverage score of out_sent by any context sentence and the matching sentence.

    Coverage is defined as |tokens(out) ∩ tokens(ctx)| / |tokens(out)|.
    """
    out_tok = _content_tokens(out_sent, stopwords)
    if not out_tok:
        return 0.0, ""
    best = 0.0
    best_ctx = ""
    out_set = set(out_tok)
    for s in ctx_sents:
        ctx_tokens = set(_content_tokens(s, stopwords))
        if not ctx_tokens:
            continue
        inter = len(out_set & ctx_tokens)
        cov = inter / len(out_set)
        if cov > best:
            best = cov
            best_ctx = s
    return best, best_ctx


def score_hallucination(
    output_text: str,
    context: Union[str, Sequence[str], Mapping[str, str], None],
    *,
    config: Optional[HallucinationConfig] = None,
) -> Tuple[float, Dict[str, object]]:
    """Compute hallucination risk for an output given context.

    - Risk 0.0 means fully grounded; 1.0 means fully ungrounded among considered sentences.
    - If no evaluable sentences (e.g., too short), returns 0.0 risk.
    - Context may be a string, a list of strings, or a mapping of id->string.
    """
    cfg = config or HallucinationConfig()
    stop = cfg.stopwords or _DEFAULT_STOPWORDS

    # Build context sentences
    ctx_texts: List[str] = []
    if isinstance(context, str):
        ctx_texts = _split_sentences(context)
    elif isinstance(context, Mapping):
        for _, v in context.items():
            ctx_texts.extend(_split_sentences(str(v)))
    elif isinstance(context, (list, tuple)):
        for v in context:
            ctx_texts.extend(_split_sentences(str(v)))

    out_sents = _split_sentences(output_text)

    considered = 0
    unsupported = 0
    per_sentence: List[Dict[str, object]] = []

    for s in out_sents:
        toks = _content_tokens(s, stop)
        if len(toks) < cfg.min_tokens:
            # too short to evaluate, ignore but track
            per_sentence.append({"text": s, "evaluated": False})
            continue

        considered += 1
        best, best_ctx = _best_sentence_overlap(s, ctx_texts, stop)
        grounded = best >= cfg.min_jaccard
        if not grounded:
            unsupported += 1
        per_sentence.append(
            {
                "text": s,
                "evaluated": True,
                "best_jaccard": best,
                "grounded": grounded,
                "matched_context": best_ctx,
            }
        )

    risk = (unsupported / considered) if considered > 0 else 0.0
    details: Dict[str, object] = {
        "considered": considered,
        "unsupported": unsupported,
        "sentences": per_sentence,
        "thresholds": {"min_jaccard": cfg.min_jaccard, "min_tokens": cfg.min_tokens},
    }
    return risk, details


__all__ = ["HallucinationConfig", "score_hallucination"]
