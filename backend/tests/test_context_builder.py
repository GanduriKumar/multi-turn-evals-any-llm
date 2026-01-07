from context_builder import build_context


def test_build_context_last4_and_budget():
    turns = [
        {"role": "user", "text": f"turn{i}"} for i in range(1, 8)
    ]
    state = {"order_id": "A1", "user_intent": "refund"}
    ctx = build_context("commerce", turns, state, max_tokens=64)
    assert ctx["audit"]["used_turn_count"] == 4
    assert ctx["messages"][0]["role"] == "system"
    # last 4 user turns included
    assert ctx["messages"][-1]["content"].endswith("turn7")
    assert ctx["audit"]["token_estimate"] <= 64
