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
    Build provider-ready messages from state + fixed last 5 turns.
    Deterministic clipping: allocate a static token cap to the system message and
    an even cap to each of the included turns. No messages are dropped.
    Returns { messages: [...], audit: {...} }.
    """
    # last N raw turns to preserve more context (default 5); allow override via env
    try:
        win = int((params_override or {}).get("window_turns")) if params_override and isinstance(params_override.get("window_turns"), (int, float)) else None
    except Exception:
        win = None
    if win is None:
        try:
            win = int(os.getenv("EVAL_CONTEXT_WINDOW_TURNS", "5"))
        except Exception:
            win = 5
    win = 5 if win is None else max(1, int(win))
    recent_turns = turns[-win:] if len(turns) > win else list(turns)

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

    # Deterministic clipping: static caps (system gets a fixed share; turns share the rest)
    truncated = False
    total = 0
    new_messages: List[Dict[str, str]] = []
    msg_count = len(messages)
    if msg_count <= 0:
        return {"messages": [], "audit": {"used_turn_count": 0, "truncated": False, "token_estimate": 0, "max_tokens": max_tokens}, "params": sys_params}

    # Allocate ~35% to system, remainder evenly across turns; enforce a sane per-message floor
    system_cap = int(max_tokens * 0.35)
    if msg_count == 1:
        caps = [max_tokens]
    else:
        per_turn_cap = max(16, (max_tokens - system_cap) // (msg_count - 1))
        caps = [system_cap] + [per_turn_cap] * (msg_count - 1)

    for m, cap in zip(messages, caps):
        content = m["content"]
        clipped, did = _clip_text_to_tokens(content, cap)
        total += approx_tokens(clipped)
        truncated = truncated or did
        new_messages.append({"role": m["role"], "content": clipped})

    audit = {
        "used_turn_count": len(recent_turns),
        "truncated": truncated,
        "token_estimate": total,
        "max_tokens": max_tokens,
        "context_mode": "deterministic_fixed5",
    }

    return {"messages": new_messages, "audit": audit, "params": sys_params}
