from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict


DEFAULT_PARAMS = {
    "temperature": 0.2,
    "max_tokens": 400,
    "top_p": 1.0,
    "presence_penalty": 0,
    "frequency_penalty": 0,
}


@dataclass
class SystemPrompt:
    content: str
    params: Dict[str, Any]


def build_system_prompt(
    *,
    domain: str,
    behavior: str,
    axes: Dict[str, str],
    policy_text: str,
    facts_text: str,
    params_override: Dict[str, Any] | None = None,
    max_len: int = 3500,
) -> SystemPrompt:
    """Compose a single system message including policy and scenario facts.

    Sections: Role, Safety/Policy, Scenario Facts, Output Requirements.
    Truncates conservatively if length exceeds max_len.
    """
    role = "You are a commerce assistant. Follow policy strictly; be concise and actionable."
    policy = policy_text.strip()
    facts = facts_text.strip()

    req = (
        "Output Requirements:\n"
        "- Be brief; ask clarifiers only if blocking.\n"
        "- Final answer must be policy-compliant and actionable.\n"
        "- Do not invent facts; rely on Scenario Facts/Policy.\n"
        "- End with: FINAL_STATE: {\"decision\": \"ALLOW|DENY|PARTIAL\", \"next_action\": <string|null>, \"refund_amount\": <number|null>, \"policy_flags\": [<strings>] }\n"
    )

    axes_line = ", ".join(f"{k}={v}" for k, v in axes.items()) if axes else ""
    header = f"Domain: {domain} | Behavior: {behavior}" + (f" | Axes: {axes_line}" if axes_line else "")

    content = (
        f"{header}\n\n"
        f"Role:\n{role}\n\n"
        f"Safety/Policy:\n{policy}\n\n"
        f"Scenario Facts:\n{facts}\n\n"
        f"{req}"
    )

    if len(content) > max_len:
        content = content[: max_len - 1] + "…"

    params = dict(DEFAULT_PARAMS)
    if params_override:
        params.update(params_override)

    return SystemPrompt(content=content, params=params)
