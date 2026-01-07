from state_extractor import extract_state


def test_extract_state_commerce():
    turns = [
        {"role": "user", "text": "Where is my order #A123?"},
        {"role": "assistant", "text": "Please share the order ID."},
        {"role": "user", "text": "Order ID is A123"},
        {"role": "assistant", "text": "We can issue refund of $10 after it's shipped."},
    ]
    s = extract_state("commerce", turns)
    assert s["user_intent"] in ("order_status", "refund")
    assert s["order_id"] == "A123"
    assert s["refund_amount"] == 10.0
    assert "after_shipment" in s["policy_flags"]


def test_extract_state_banking():
    turns = [
        {"role": "user", "text": "Please transfer $100 from acct id 9XYZ to saving."},
        {"role": "assistant", "text": "Approved. Proceeding."},
    ]
    s = extract_state("banking", turns)
    assert s["user_intent"] == "transfer"
    assert s["amount"] == 100.0
    assert s["account_id"] == "9XYZ"
    assert s["decision"] == "ALLOW"
