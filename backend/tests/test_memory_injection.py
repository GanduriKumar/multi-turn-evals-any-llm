from __future__ import annotations

from pathlib import Path
import json

from eval_server.runner.turn_runner import TurnRunner
import eval_server.llm  # ensure providers register


def test_memory_string_injected_into_context():
    repo = Path(__file__).resolve().parents[2]
    conv_path = repo / "configs" / "datasets" / "examples" / "conversation_001.json"
    conversation = json.loads(conv_path.read_text(encoding="utf-8"))

    runner = TurnRunner("dummy", prefix="M:")
    results = runner.run(conversation, memory="summary: user asked about leave policy")

    assert len(results) == 1
    ctx = results[0].context
    assert any(line.startswith("MEMORY:") for line in ctx)
    assert any("summary: user asked" in line for line in ctx)


def test_memory_mapping_and_list_injected():
    conversation = {
        "version": "1.0.0",
        "conversation_id": "c",
        "turns": [
            {"turn_id": "t1", "role": "system", "content": "hello"},
            {"turn_id": "t2", "role": "user", "content": "question"},
        ]
    }

    runner = TurnRunner("dummy", prefix="X:")
    # mapping memory
    res1 = runner.run(conversation, memory={"topic": "math", "lang": "en"})
    ctx1 = res1[0].context
    assert any(line.startswith("MEMORY: topic=") for line in ctx1)
    assert any(line.startswith("MEMORY: lang=") for line in ctx1)

    # list memory
    res2 = runner.run(conversation, memory=["previous: greeted", "state: warmup"])
    ctx2 = res2[0].context
    assert any(line.startswith("MEMORY: previous:") for line in ctx2)
    assert any(line.startswith("MEMORY: state:") for line in ctx2)
