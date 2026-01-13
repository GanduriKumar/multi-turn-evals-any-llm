from __future__ import annotations
import hashlib
from typing import Any, Dict, Tuple

from .policy_facts import load_policy_and_facts
from .canonical_a2_lib import compose_canonical_a2
import re


def _hash_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def u1_text(domain: str, axes: Dict[str, str]) -> str:
    ps = axes.get("price_sensitivity")
    bb = axes.get("brand_bias")
    av = axes.get("availability")
    pb = axes.get("policy_boundary")
    return (
        f"I need help with my recent order. I'm {ps} on price, {bb} on brand, availability is {av}, and "
        f"this seems {pb}. What can you do?"
    )


def u2_text(domain: str, axes: Dict[str, str]) -> str:
    av = axes.get("availability")
    bb = axes.get("brand_bias")
    if av in ("sold_out", "out_of_stock"):
        return "Actually, the item is unavailable now. Can we substitute or refund?"
    if bb == "hard":
        return "I prefer the same brand if possible; otherwise, suggest close alternatives."
    return "To clarify, I can share the order ID and accept a similar item if needed."


def u3_text(domain: str, axes: Dict[str, str]) -> str:
    ps = axes.get("price_sensitivity")
    pb = axes.get("policy_boundary")
    if pb == "near_edge_allowed":
        return "I need this resolved today. If policy allows near the edge, can you expedite a resolution or partial credit?"
    if ps == "high":
        return "Cost matters a lot for me. Could we consider a partial refund or store credit if a full refund isn't possible?"
    return "Timing is tight on my side. What are my fastest options to resolve this now?"


def u4_text(domain: str, axes: Dict[str, str]) -> str:
    av = axes.get("availability")
    pb = axes.get("policy_boundary")
    if av in ("sold_out", "out_of_stock"):
        return "If it's unavailable, can you split the order and ship an alternative while issuing a partial refund for the rest?"
    if pb == "within_policy":
        return "Assuming we stay within policy, what exact steps will you take and what should I expect next?"
    return "If policy is strict here, what's the compliant workaround that still helps me today?"


def canonical_a2(behavior: str, policy_text: str, facts_text: str, axes: Dict[str, str]) -> str:
    return compose_canonical_a2(behavior, policy_text, facts_text, axes)


def build_records(
    *,
    domain: str,
    behavior: str,
    axes: Dict[str, str],
    version: str = "1.0.0",
    seed: int = 42,
    user_turns: int = 2,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    policy_text, facts_text = load_policy_and_facts(domain, axes, seed)
    def _slug(s: str) -> str:
        # lower, replace non-alphanum with '-', collapse repeats, trim dashes
        s2 = re.sub(r"[^A-Za-z0-9._-]+", "-", s.strip().lower())
        s2 = re.sub(r"-+", "-", s2).strip('-')
        return s2
    convo_id = f"{_slug(domain)}.{_slug(behavior)}." + ",".join(f"{k}={v}" for k, v in axes.items()) + f".{_hash_id(domain+behavior+str(axes))}"

    # dataset conversation: only user turns (assistant turns generated at run-time)
    dataset_conv = {
        "conversation_id": convo_id,
        "metadata": {
            "domain_label": domain,
            "behavior": behavior,
            "axes": axes,
            "policy_excerpt": policy_text[:280],
            "facts_bullets": facts_text,
            "short_description": f"{behavior} with axes {axes}",
        },
        "turns": [],
    }
    # Clamp supported range to 1..4 (templates provided for U1..U4)
    try:
        ut = int(user_turns)
    except Exception:
        ut = 2
    ut = 1 if ut < 1 else (4 if ut > 4 else ut)
    dataset_conv["turns"].append({"role": "user", "text": u1_text(domain, axes)})
    if ut >= 2:
        dataset_conv["turns"].append({"role": "user", "text": u2_text(domain, axes)})
    if ut >= 3:
        dataset_conv["turns"].append({"role": "user", "text": u3_text(domain, axes)})
    if ut >= 4:
        dataset_conv["turns"].append({"role": "user", "text": u4_text(domain, axes)})

    # golden: final outcome Allowed + canonical A2 for final turn
    # Golden expected index depends on number of user turns k: final assistant at index 2*k-1
    expected_turn_index = 2 * ut - 1
    golden_entry = {
        "conversation_id": convo_id,
        "turns": [
            {"turn_index": expected_turn_index, "expected": {"variants": [canonical_a2(behavior, policy_text, facts_text, axes)]}},
        ],
        "final_outcome": {"decision": "ALLOW"},
        "constraints": {"respect_policy": True},
    }

    # Ensure unique dataset_id per scenario to avoid overwrite collisions when saving multiple scenarios
    scenario_hash = _hash_id(domain + behavior + str(axes))
    dataset_doc = {
        "dataset_id": f"{_slug(domain)}-{_slug(behavior)}-v{version}-{scenario_hash}",
        "version": version,
        "metadata": {"domain": "commerce", "difficulty": "mixed", "tags": ["risk_weighted"]},
        "conversations": [dataset_conv],
    }

    golden_doc = {"dataset_id": dataset_doc["dataset_id"], "version": version, "entries": [golden_entry]}

    return dataset_doc, golden_doc
