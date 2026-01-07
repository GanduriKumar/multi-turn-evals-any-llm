from __future__ import annotations
from typing import Dict, List, Any, Tuple
import json

# very rough token estimator (~4 chars per token)
_DEF_TOKENS_PER_CHAR = 1 / 4.0


def approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) * _DEF_TOKENS_PER_CHAR))


def _render_state_summary(state: Dict[str, Any]) -> str:
    # compact JSON for determinism and readability
    return json.dumps({k: v for k, v in state.items() if v not in (None, [], {})}, separators=(",", ":"))


def _clip_text_to_tokens(text: str, max_tokens: int) -> Tuple[str, bool]:
    if approx_tokens(text) <= max_tokens:
        return text, False
    # keep tail since recent info is often at the end
    # convert token budget to char budget
    char_budget = max(1, int(max_tokens / _DEF_TOKENS_PER_CHAR))
    clipped = text[-char_budget:]
    # mark truncation
    if len(clipped) < len(text):
        clipped = "…" + clipped
    return clipped, True


def build_context(domain: str, turns: List[Dict[str, str]], state: Dict[str, Any], max_tokens: int = 2048) -> Dict[str, Any]:
    """
    Build provider-ready messages from state + last 4 turns with a simple token budget safeguard.
    Returns { messages: [...], audit: {...} }.
    """
    # last 4 raw turns
    recent_turns = turns[-4:] if len(turns) > 4 else list(turns)

    system_content = (
        f"You are an assistant for {domain}. "
        f"Use the following current state to answer succinctly and accurately.\n"
        f"STATE={_render_state_summary(state)}"
    )
    messages = [{"role": "system", "content": system_content}]

    for t in recent_turns:
        role = t.get("role", "user")
        content = t.get("text", "")
        messages.append({"role": role, "content": content})

    # apply a simple budget: clip per-message content (do not drop messages)
    truncated = False
    total = 0
    budget = max_tokens
    new_messages: List[Dict[str, str]] = []
    for m in messages:
        content = m["content"]
        # leave at least 1 token per remaining message
        remaining_msgs = len(messages) - len(new_messages)
        # naive fair-share budget
        fair = max(16, budget // max(1, remaining_msgs))
        clipped, did = _clip_text_to_tokens(content, fair)
        total += approx_tokens(clipped)
        budget = max(0, budget - approx_tokens(clipped))
        truncated = truncated or did
        new_messages.append({"role": m["role"], "content": clipped})

    audit = {
        "used_turn_count": len(recent_turns),
        "truncated": truncated,
        "token_estimate": total,
        "max_tokens": max_tokens,
    }

    return {"messages": new_messages, "audit": audit}
