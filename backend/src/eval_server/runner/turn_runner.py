from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Union, Callable

from ..llm import LLMProvider, create_provider  # ensures providers are imported and registered
from ..utils.truncation import make_truncator


@dataclass(frozen=True)
class TurnRunResult:
    turn_id: str
    prompt: str
    response: str
    context: List[str]


ToolOutputHook = Callable[[str, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], Any]


def _stringify_tool_call(tc: Mapping[str, Any]) -> str:
    name = str(tc.get("tool_name", "")).strip()
    args = tc.get("arguments")
    res = tc.get("result")
    return f"TOOL: {name} args={args} result={res}"


def _stringify_turn(
    turn: Mapping[str, Any],
    *,
    tool_output_hook: Optional[ToolOutputHook] = None,
    conversation: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    parts: List[str] = []
    role = str(turn.get("role", "")).strip().upper() or "UNKNOWN"
    content = str(turn.get("content", "") or "")
    if content:
        parts.append(f"{role}: {content}")
    # include any tool_calls on the turn
    tool_calls = turn.get("tool_calls") or []
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if isinstance(tc, Mapping):
                # If no result provided, attempt to simulate via hook
                if (tc.get("result") is None) and tool_output_hook is not None:
                    try:
                        computed = tool_output_hook(
                            str(tc.get("tool_name", "")),
                            tc.get("arguments") or {},
                            conversation or {},
                            turn,
                        )
                        # Create a shallow copy to avoid mutating input
                        tc = dict(tc)
                        tc["result"] = computed
                    except Exception as e:
                        # Record error as result for transparency
                        tc = dict(tc)
                        tc["result"] = f"ERROR:{e}"
                parts.append(_stringify_tool_call(tc))
    return parts


def _memory_to_lines(memory: Optional[Union[str, Mapping[str, Any], List[str]]]) -> List[str]:
    """Convert a memory object into context lines.

    - str => ["MEMORY: <str>"]
    - list[str] => ["MEMORY: <line>" for line in list]
    - mapping => ["MEMORY: <k>=<v>" for top-level items]
    - None => []
    """
    if memory is None:
        return []
    lines: List[str] = []
    if isinstance(memory, str):
        lines.append(f"MEMORY: {memory}")
    elif isinstance(memory, list):
        for item in memory:
            lines.append(f"MEMORY: {item}")
    elif isinstance(memory, Mapping):
        for k, v in memory.items():
            lines.append(f"MEMORY: {k}={v}")
    else:
        # Fallback stringification
        lines.append(f"MEMORY: {memory}")
    return lines


class TurnRunner:
    """Runs a conversation turn-by-turn, invoking an LLM for each user turn.

    Maintains a growing textual context of prior turns and tool calls.
    """

    def __init__(self, provider: LLMProvider | str, **provider_init: Any) -> None:
        if isinstance(provider, str):
            self._provider: LLMProvider = create_provider(provider, **provider_init)
        else:
            self._provider = provider
        self._tool_hook: Optional[ToolOutputHook] = None

    def run(
        self,
        conversation: Mapping[str, Any],
        *,
        memory: Optional[Union[str, Mapping[str, Any], List[str]]] = None,
        truncation_policy: Optional[str] = None,
        truncation_params: Optional[Dict[str, int]] = None,
        tool_output_hook: Optional[ToolOutputHook] = None,
    ) -> List[TurnRunResult]:
        results: List[TurnRunResult] = []
        context: List[str] = []
        turns = conversation.get("turns") or []

        for turn in turns:
            if not isinstance(turn, Mapping):
                continue
            hook = tool_output_hook or self._tool_hook
            turn_ctx_parts = _stringify_turn(turn, tool_output_hook=hook, conversation=conversation)
            role = str(turn.get("role", "")).strip().lower()

            if role == "user":
                # For a user turn, include prior context + current turn's tool_calls (if any)
                # Tool calls should appear before the prompt content in context.
                # Build current-context snapshot without the user's content (prompt)
                # Separate tool-call parts from content part
                tool_parts = [p for p in turn_ctx_parts if p.startswith("TOOL:")]
                content_part = [p for p in turn_ctx_parts if not p.startswith("TOOL:")]
                # The prompt is the raw content of the user turn
                prompt_text = str(turn.get("content", "") or "")
                # Merge memory lines with prior context, then tool calls for this turn
                mem_lines = _memory_to_lines(memory)
                # Context passed to the provider consists of prior context + memory + any tool call lines for this turn
                base_context = context + mem_lines + tool_parts
                # Apply truncation policy if provided
                if truncation_policy:
                    trunc = make_truncator(truncation_policy, **(truncation_params or {}))
                    used_context = trunc(base_context)
                else:
                    used_context = base_context
                response = self._provider.generate(prompt_text, context=used_context)

                # Record result
                results.append(TurnRunResult(
                    turn_id=str(turn.get("turn_id", "")),
                    prompt=prompt_text,
                    response=response,
                    context=list(used_context),
                ))

                # After generating, extend the global context with the user content and assistant response
                # Include the full user line (content_part) to context, then assistant response
                context.extend(content_part)
                context.append(f"ASSISTANT: {response}")
            else:
                # Non-user turns contribute their lines to the context
                context.extend(turn_ctx_parts)

        return results


__all__ = ["TurnRunner", "TurnRunResult"]
