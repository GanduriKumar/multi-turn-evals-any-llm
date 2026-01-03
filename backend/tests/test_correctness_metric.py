from __future__ import annotations

import math
from typing import List

from eval_server.metrics.correctness import (
    SemanticConfig,
    exact_match,
    semantic_match,
    structured_match,
    score_correctness,
)


def dummy_embedder_factory(dim: int = 3):
    """Return a toy embedder that maps text to a vector deterministically.

    Strategy: hash characters into buckets to create a simple numeric vector.
    Not a real embedding, but stable for testing cosine similarity behaviour.
    """
    def embed(text: str) -> List[float]:
        buckets = [0.0] * dim
        for i, ch in enumerate(text):
            buckets[i % dim] += (ord(ch) % 31) / 30.0
        return buckets

    return embed


def test_exact_match_variants():
    assert exact_match("Hello World", ["hello world"]) == 1.0
    assert exact_match("Hello", ["hi", "hey", "HELLO"]) == 1.0
    assert exact_match("Nope", ["Different"]) == 0.0


def test_structured_match_partial_and_full():
    expected = {"decision": "APPROVE", "reason_code": "A1"}
    # Full match
    assert structured_match({"decision": "approve", "reason_code": "A1", "extra": 1}, expected) == 1.0
    # Partial: only decision matches
    score = structured_match({"decision": "approve", "reason_code": "B2"}, expected)
    assert math.isclose(score, 0.5, rel_tol=1e-9)
    # None match
    score = structured_match({"decision": "deny", "reason_code": "B2"}, expected)
    assert score == 0.0


def test_semantic_match_with_embedder_partial_credit():
    embed = dummy_embedder_factory(dim=5)
    # Identical string -> similarity 1.0
    assert semantic_match("test", "test", embedder=embed) == 1.0
    # Different but similar length/characters -> similarity between 0 and 1
    sim = semantic_match("hello", "hEllo!", embedder=embed)
    assert 0.0 <= sim <= 1.0


def test_semantic_match_binary_threshold():
    embed = dummy_embedder_factory(dim=4)
    # Use a very high threshold so non-identical strings fail under the toy embedder
    cfg = SemanticConfig(threshold=0.999, partial_credit=False)
    # High threshold, identical text passes
    assert semantic_match("abc", "abc", embedder=embed, config=cfg) == 1.0
    # Different text likely below threshold -> 0.0
    assert semantic_match("abc", "xyz", embedder=embed, config=cfg) == 0.0


def test_score_correctness_modes():
    actual = {"text": "hello world", "structured": {"decision": "approve"}}
    expected = {"text": "Hello World", "variants": ["HELLO WORLD"], "structured": {"decision": "APPROVE"}}

    # exact via variants
    s = score_correctness(actual, expected, mode="exact")
    assert s == 1.0

    # semantic using embedder
    embed = dummy_embedder_factory()
    s = score_correctness(actual, expected, mode="semantic", embedder=embed)
    assert 0.0 <= s <= 1.0

    # structured mode
    s = score_correctness(actual, expected, mode="structured")
    assert s == 1.0
