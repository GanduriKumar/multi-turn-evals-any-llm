from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from eval_server.runner.turn_runner import TurnRunner
import eval_server.llm  # ensure providers register


def test_turn_runner_context_growth_with_dummy_provider():
    # Use the sample conversation; responses will be deterministic with dummy provider
    repo = Path(__file__).resolve().parents[2]
    conv_path = repo / "configs" / "datasets" / "examples" / "conversation_001.json"
    import json
    conversation = json.loads(conv_path.read_text(encoding="utf-8"))

    runner = TurnRunner("dummy", prefix="RUN:")
    results = runner.run(conversation)

    # Expect exactly one user turn in the sample (t2), so one result
    assert len(results) == 1

    r0 = results[0]
    # The context should include prior system message and any tool call lines for the user turn
    assert any(line.startswith("SYSTEM:") for line in r0.context)
    assert any(line.startswith("TOOL:") for line in r0.context)

    # The prompt is the user's content
    assert r0.prompt == "Summarize our leave policy"
    # Dummy provider response deterministic prefix and context usage
    assert r0.response.startswith("RUN:")

    # Ensure that after the run, assistant response was added to global context (implicit via runner)
    # We can't access internal context, but we can run again and ensure context starts with prior assistant reply
    results2 = runner.run(conversation)
    # On a new run, runner starts with empty context by design; so results remain consistent in size
    assert len(results2) == 1


def test_turn_runner_handles_tool_calls_in_context():
    # Build a small conversation with multiple user turns and tool calls
    conversation = {
        "version": "1.0.0",
        "conversation_id": "c-demo",
        "turns": [
            {"turn_id": "t1", "role": "system", "content": "Be concise."},
            {"turn_id": "t2", "role": "user", "content": "What is 2+2?", "tool_calls": [
                {"tool_name": "calculator", "arguments": {"expr": "2+2"}, "result": 4}
            ]},
            {"turn_id": "t3", "role": "assistant", "content": "It's 4."},
            {"turn_id": "t4", "role": "user", "content": "And 3*3?"}
        ]
    }

    runner = TurnRunner("dummy", prefix="D:")
    results = runner.run(conversation)

    # There are two user turns, hence two provider calls
    assert len(results) == 2

    # First call context should contain system line and tool call
    ctx0 = results[0].context
    assert any(line.startswith("SYSTEM:") for line in ctx0)
    assert any(line.startswith("TOOL:") for line in ctx0)

    # Second call context should include prior user+assistant exchange
    ctx1 = results[1].context
    assert any("ASSISTANT:" in line for line in ctx1)
    # No tool call on second user turn, so only prior context lines
    assert not any(line.startswith("TOOL:") and ("expr" in line) for line in ctx1)
