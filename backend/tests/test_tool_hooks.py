from __future__ import annotations

from typing import Any, Mapping

from eval_server.runner.turn_runner import TurnRunner, ToolOutputHook


def test_tool_output_hook_injection():
    # Conversation where the user turn references a tool but no result provided
    conversation = {
        "version": "1.0.0",
        "conversation_id": "c-tools",
        "turns": [
            {"turn_id": "t1", "role": "system", "content": "Use tools when needed."},
            {"turn_id": "t2", "role": "user", "content": "Search docs", "tool_calls": [
                {"tool_name": "doc_search", "arguments": {"query": "policy"}}
            ]},
        ]
    }

    def hook(tool_name: str, args: Mapping[str, Any], conv: Mapping[str, Any], turn: Mapping[str, Any]) -> Any:
        assert tool_name == "doc_search"
        assert args.get("query") == "policy"
        return {"doc_id": 99, "snippet": "policy text"}

    runner = TurnRunner("dummy", prefix="H:")
    results = runner.run(conversation, tool_output_hook=hook)

    assert len(results) == 1
    # Ensure the context contains the resolved tool output
    ctx = results[0].context
    line = next((l for l in ctx if l.startswith("TOOL:") and "doc_search" in l), None)
    assert line is not None
    assert "doc_id" in line and "99" in line
    assert "snippet" in line
