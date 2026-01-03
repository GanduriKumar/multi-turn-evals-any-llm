from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Callable

import math

from ..evaluation.structured import compare_structured
from ..scoring.normalizer import canonicalize_text


@dataclass(frozen=True)
class SemanticConfig:
    """Configuration for semantic similarity matching.

    Attributes:
        threshold: Minimum cosine similarity to count as a full match.
        partial_credit: If True, return similarity score directly; if False, return 1.0 if >= threshold else 0.0.
    """

    threshold: float = 0.8
    partial_credit: bool = True


def _cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 if a zero vector is encountered or lengths differ.
    """
    va = list(float(x) for x in a)
    vb = list(float(x) for x in b)
    if len(va) != len(vb) or len(va) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(y * y for y in vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def exact_match(actual_text: str, expected_variants: Iterable[str]) -> float:
    """Return 1.0 if the actual_text matches any expected variant (case-insensitive), else 0.0."""
    a = canonicalize_text(actual_text)
    for v in expected_variants or {}:
        if a == canonicalize_text(v):
            return 1.0
    return 0.0


def semantic_match(
    actual_text: str,
    expected_text: str,
    *,
    embedder: Optional[Callable[[str], Iterable[float]]] = None,
    config: Optional[SemanticConfig] = None,
) -> float:
    """Compute semantic similarity score between actual and expected using an embedder.

    - embedder: callable that maps a string -> Iterable[float] embedding vector.
    - If no embedder is provided, fall back to exact_match behaviour (binary score).
    - If partial_credit is True (default), return cosine similarity in [0,1].
      If False, return 1.0 if similarity >= threshold else 0.0.
    """
    cfg = config or SemanticConfig()
    if embedder is None:
        # Fallback to exact match when no embeddings available
        return exact_match(actual_text, [expected_text])

    a_vec = embedder(canonicalize_text(actual_text))
    e_vec = embedder(canonicalize_text(expected_text))
    sim = _cosine_similarity(a_vec, e_vec)
    if cfg.partial_credit:
        # Clamp to [0,1] to be safe
        return max(0.0, min(1.0, sim))
    return 1.0 if sim >= cfg.threshold else 0.0


def structured_match(actual_struct: Mapping[str, Any], expected_struct: Mapping[str, Any]) -> float:
    """Return a score in [0,1] comparing two structured objects.

    Full credit (1.0) for perfect match on all expected keys; otherwise partial credit is the
    fraction of expected keys that match (case-insensitive for strings).
    """
    if not expected_struct:
        return 0.0
    passed, details = compare_structured(actual_struct, expected_struct)
    if passed:
        return 1.0
    mismatches: Dict[str, Any] = details.get("mismatches", {})
    total = len(expected_struct)
    matched = total - len(mismatches)
    return matched / total if total > 0 else 0.0


def score_correctness(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    mode: str = "exact",
    embedder: Optional[Callable[[str], Iterable[float]]] = None,
    semantic: Optional[SemanticConfig] = None,
) -> float:
    """Score per-turn correctness across modes: exact | semantic | structured.

    Inputs:
      - actual: {"text": str, "structured": Mapping}
      - expected: {"text": str | None, "variants": list[str] | None, "structured": Mapping | None}
      - mode: selects matching algorithm
      - embedder: required for semantic mode (string -> vector)

    Returns a float score in [0,1].
    """
    m = (mode or "").strip().lower()

    if m == "structured":
        return structured_match(actual.get("structured", {}) or {}, expected.get("structured", {}) or {})

    # Default to text-based matching
    a_text = str(actual.get("text", ""))
    # Prefer explicit single text, else variants list
    variants = expected.get("variants") or []
    if m == "exact":
        # If variants provided, succeed if any matches. Otherwise use expected["text"].
        if variants:
            return exact_match(a_text, variants)
        return exact_match(a_text, [str(expected.get("text", ""))])

    if m == "semantic":
        # Use first variant if provided; else expected text
        exp_text = None
        if variants:
            exp_text = str(variants[0])
        else:
            exp_text = str(expected.get("text", ""))
        return semantic_match(a_text, exp_text, embedder=embedder, config=semantic)

    # Fallback: exact
    return exact_match(a_text, variants or [str(expected.get("text", ""))])


__all__ = [
    "SemanticConfig",
    "exact_match",
    "semantic_match",
    "structured_match",
    "score_correctness",
]
