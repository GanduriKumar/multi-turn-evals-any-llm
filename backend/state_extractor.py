from __future__ import annotations
import re
from typing import Dict, List, Optional, Any

COMMON_INTENTS = {
    "commerce": {
        "order_status": [r"where is my order", r"track(ing)? order", r"order status"],
        "refund": [r"refund", r"money back"],
        "return": [r"return(ing)?", r"return window"],
        "exchange": [r"exchange"],
        "shipping_delay": [r"delay(ed)? shipping", r"late delivery"],
        "coupon": [r"coupon", r"promo"],
        "replacement": [r"replacement"],
    },
    "banking": {
        "balance_inquiry": [r"balance", r"how much (money|funds)"],
        "transfer": [r"transfer", r"send money"],
        "charge_dispute": [r"dispute", r"chargeback", r"unauthorized charge"],
        "card_block": [r"block (my )?card", r"freeze card"],
        "loan_status": [r"loan status", r"application status"],
        "fee_waiver": [r"waive fee", r"fee refund"],
    },
}

DECISION_MAP = {
    r"\b(approve(d)?|allowed?|grant(ed)?)\b": "ALLOW",
    r"\b(deny|denied|cannot|can't|not able)\b": "DENY",
    r"\b(partial|partly)\b": "PARTIAL",
}

POLICY_FLAGS_PATTERNS = {
    "after_shipment": [r"after (it'?s )?shipped", r"shipped already"],
    "outside_return_window": [r"outside (the )?return window", r"past (the )?return window"],
    "no_receipt": [r"no receipt"],
    "max_refund": [r"max(imum)? refund", r"exceeds threshold", r"over (the )?limit"],
}

ACCOUNT_PAT = re.compile(r"\b(?:acct|account)\s*(?:id|number|#)?\s*[:#]?\s*([A-Z0-9-]{4,})\b", re.I)
ORDER_PAT = re.compile(r"\b(?:order(?:\s*(?:id|number))?)\s*(?:is|:)?\s*([A-Z0-9-]{4,})\b|#([A-Z0-9-]{4,})\b", re.I)
AMOUNT_REFUND_PAT = re.compile(r"(?:refund(?: of)?|reimburse(?:ment)?(?: of)?)\s*\$?\s*([0-9]+(?:\.[0-9]{1,2})?)", re.I)
AMOUNT_GENERAL_PAT = re.compile(r"\$\s*([0-9]+(?:\.[0-9]{1,2})?)\b")


def _detect_intent(domain: str, text: str) -> Optional[str]:
    text_l = text.lower()
    for intent, pats in COMMON_INTENTS.get(domain, {}).items():
        for p in pats:
            if re.search(p, text_l):
                return intent
    return None


def _collect_policy_flags(text: str) -> List[str]:
    out: List[str] = []
    tl = text.lower()
    for flag, pats in POLICY_FLAGS_PATTERNS.items():
        for p in pats:
            if re.search(p, tl):
                out.append(flag)
                break
    return out


def _detect_decision(text: str) -> Optional[str]:
    for pat, val in DECISION_MAP.items():
        if re.search(pat, text, re.I):
            return val
    return None


def extract_state(domain: str, turns: List[Dict[str, str]], prev_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Deterministic extractor over the full transcript (turns as list of {role,user|assistant; text}).
    Returns a compact state object aggregated from all turns. No LLM usage.
    """
    state: Dict[str, Any] = {
        "user_intent": None,
        "decision": None,
        "next_action": None,
        "policy_flags": [],
        "notes": None,
    }
    if domain == "commerce":
        state.update({
            "order_id": None,
            "items": None,
            "totals": None,
            "refund_amount": None,
        })
    elif domain == "banking":
        state.update({
            "account_id": None,
            "amount": None,
            "kyc_status": None,
            "limit_flags": [],
        })

    if prev_state:
        # start from previous known values
        for k, v in prev_state.items():
            state[k] = v

    # Process turns sequentially, latest info wins
    for t in turns:
        role = t.get("role", "").lower()
        text = t.get("text", "")
        # intent primarily from user turns
        if role == "user":
            intent = _detect_intent(domain, text)
            if intent:
                state["user_intent"] = intent
        # decisions/actions from assistant turns
        if role == "assistant":
            dec = _detect_decision(text)
            if dec:
                state["decision"] = dec
            if re.search(r"issue (a )?refund|process(ing)? refund", text, re.I):
                state["next_action"] = "issue_refund"
            elif re.search(r"confirm order|confirmed order", text, re.I):
                state["next_action"] = "confirm_order"
            elif re.search(r"escalat(e|ion)", text, re.I):
                state["next_action"] = "escalate"
            elif re.search(r"need (more )?info|provide details", text, re.I):
                state["next_action"] = "request_more_info"

        # Common flags and notes
        flags = _collect_policy_flags(text)
        if flags:
            for f in flags:
                if f not in state["policy_flags"]:
                    state["policy_flags"].append(f)

        # Domain-specific extraction
        if domain == "commerce":
            m = ORDER_PAT.search(text)
            if m:
                order_id = m.group(1) or m.group(2)
                state["order_id"] = order_id
            mr = AMOUNT_REFUND_PAT.search(text)
            if mr:
                state["refund_amount"] = float(mr.group(1))
            mt = re.search(r"total\s*\$?\s*([0-9]+(?:\.[0-9]{1,2})?)", text, re.I)
            if mt:
                state["totals"] = float(mt.group(1))
        elif domain == "banking":
            ma = ACCOUNT_PAT.search(text)
            if ma:
                state["account_id"] = ma.group(1)
            mamt = AMOUNT_GENERAL_PAT.search(text)
            if mamt:
                state["amount"] = float(mamt.group(1))
            if re.search(r"kyc (ok|passed)", text, re.I):
                state["kyc_status"] = "ok"
            elif re.search(r"kyc (fail|flag)", text, re.I):
                state["kyc_status"] = "flag"
            if re.search(r"limit exceeded|over limit|above limit", text, re.I):
                if "limit_exceeded" not in state.get("limit_flags", []):
                    state.setdefault("limit_flags", []).append("limit_exceeded")

    return state
