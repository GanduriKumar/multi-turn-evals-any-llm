from __future__ import annotations

from pathlib import Path
import json

from eval_server.runner.turn_runner import TurnRunner


def _build_long_conversation(n: int = 12):
    turns = [{"turn_id": "t1", "role": "system", "content": "You are helpful."}]
    # Alternate user/assistant
    for i in range(2, n + 1):
        if i % 2 == 0:
            turns.append({"turn_id": f"t{i}", "role": "user", "content": f"msg {i}"})
        else:
            turns.append({"turn_id": f"t{i}", "role": "assistant", "content": f"resp {i}"})
    return {"version": "1.0.0", "conversation_id": "long", "turns": turns}


def test_truncation_full_history():
    conv = _build_long_conversation(8)
    runner = TurnRunner("dummy", prefix="T:")

    results = runner.run(conv, truncation_policy="full")
    # For each user turn, context length should grow monotonically
    ctx_lengths = [len(r.context) for r in results]
    assert all(b >= a for a, b in zip(ctx_lengths, ctx_lengths[1:]))


def test_truncation_windowed_history():
    conv = _build_long_conversation(14)
    runner = TurnRunner("dummy", prefix="T:")

    window = 3
    results = runner.run(conv, truncation_policy="windowed", truncation_params={"window_size": window})
    # Each context should not exceed the window size
    assert all(len(r.context) <= window for r in results)


def test_truncation_summarized_history():
    conv = _build_long_conversation(10)
    runner = TurnRunner("dummy", prefix="T:")

    results = runner.run(conv, truncation_policy="summarized", truncation_params={"summary_max_items": 4})
    # Context should be a single SUMMARY line for each call
    for r in results:
        assert len(r.context) == 1
        assert r.context[0].startswith("SUMMARY:")
