from __future__ import annotations
from typing import Dict, List, Any, Tuple, Optional
import json

# Support both package and top-level imports in tests/CLI
try:
    from .system_prompt import build_system_prompt, DEFAULT_PARAMS  # type: ignore
except Exception:  # ImportError when run as top-level module
    from system_prompt import build_system_prompt, DEFAULT_PARAMS  # type: ignore

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


def build_context(domain: str, turns: List[Dict[str, str]], state: Dict[str, Any], max_tokens: int = 1800, conv_meta: Optional[Dict[str, Any]] = None, params_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build provider-ready messages from state + last 4 turns with a simple token budget safeguard.
    Returns { messages: [...], audit: {...} }.
    """
    # last 5 raw turns to preserve more context
    recent_turns = turns[-5:] if len(turns) > 5 else list(turns)

    # Try to pull policy+facts from conversation metadata if provided (new datasets)
    cm = conv_meta or {}
    policy = cm.get("policy_excerpt")
    facts = cm.get("facts_bullets")
    axes = cm.get("axes")
    behavior = cm.get("behavior") or ""
    sp = None
    try:
        if policy and facts and axes:
            sp = build_system_prompt(domain=domain, behavior=behavior, axes=axes, policy_text=policy, facts_text=facts)
    except Exception:
        sp = None
    if sp is not None:
        system_content = sp.content + f"\nSTATE={_render_state_summary(state)}"
        sys_params = dict(sp.params or {})
    else:
        # Fallback prompt when dataset lacks policy/facts metadata.
        # Still include explicit Output Requirements and FINAL_STATE instruction so
        # downstream scoring can reliably extract the final outcome.
        req = (
            "Output Requirements:\n"
            "- Be brief; ask clarifiers only if blocking.\n"
            "- Final answer must be policy-compliant and actionable.\n"
            "- Do not invent facts.\n"
            "- End with: FINAL_STATE: {\"decision\": \"ALLOW|DENY|PARTIAL\", \"next_action\": <string|null>, \"refund_amount\": <number|null>, \"policy_flags\": [<strings>] }\n"
        )
        system_content = (
            f"You are an assistant for {domain}. Follow company policy while being helpful and concise.\n\n"
            f"{req}\n"
            f"STATE={_render_state_summary(state)}"
        )
        sys_params = dict(DEFAULT_PARAMS)
    # Apply explicit overrides last
    if params_override:
        try:
            sys_params.update({k: v for k, v in params_override.items() if v is not None})
        except Exception:
            pass
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

    return {"messages": new_messages, "audit": audit, "params": sys_params}
