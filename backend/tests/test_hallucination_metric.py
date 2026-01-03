from __future__ import annotations

from eval_server.metrics.hallucination import HallucinationConfig, score_hallucination


def test_hallucination_fully_grounded_paragraph():
    context = (
        "Acme Corp reported revenue of $10M in Q3. The growth was 15% year over year. "
        "The company announced a new product line in September."
    )
    output = "Acme Corp reported revenue of $10M in Q3. Growth was 15% year over year."
    risk, details = score_hallucination(output, context)
    assert risk == 0.0
    assert details["considered"] >= 1


def test_hallucination_unsupported_sentence_increases_risk():
    context = "The sky is blue and clear today over the city of Paris."
    output = "The sky is blue. Mars has an ocean."
    risk, details = score_hallucination(output, context, config=HallucinationConfig(min_tokens=2))
    # One sentence grounded, one unsupported -> risk ~ 0.5
    assert 0.45 <= risk <= 0.55
    assert details["unsupported"] == 1


def test_hallucination_with_list_context():
    context = ["Item A is available", "Item B launches in May"]
    output = "Item B launches in May. Item C is discontinued."
    risk, details = score_hallucination(output, context)
    assert 0.4 <= risk <= 0.6


def test_hallucination_ignores_short_sentences():
    context = "Product X ships soon."
    output = "OK. Yes. Great. Product X ships soon."
    # Only the longer sentence should be evaluated, which is grounded
    risk, details = score_hallucination(output, context)
    assert risk == 0.0
    assert details["considered"] == 1
