from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from eval_server.schemas.llm import LLMRequest, LLMResponse, ModelMetadata, TokenUsage


def test_llm_request_and_response_serialization():
    meta = ModelMetadata(provider="dummy", model_id="dummy-1.0")
    req = LLMRequest(
        request_id="req-123",
        Promptstring="Say hi",
        context=["Hello", " World ", ""],
        model=meta,
    )

    # Serialize to JSON and back
    req_json = req.model_dump_json()
    req2 = LLMRequest.model_validate_json(req_json)
    assert req2.model.provider == "dummy"
    assert req2.context == ["Hello", "World"]

    # Build response
    created = datetime.now(timezone.utc)
    completed = created + timedelta(milliseconds=25)

    usage = TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    resp = LLMResponse(
        request_id=req.request_id,
        model=meta,
        text="Hi",
        usage=usage,
        latency_ms=25.0,
        provider_metadata={"engine": "test"},
        created_at=created,
        completed_at=completed,
    )

    resp_json = resp.model_dump_json()
    resp2 = LLMResponse.model_validate_json(resp_json)
    assert resp2.text == "Hi"
    assert resp2.usage.total_tokens == 5
    assert resp2.completed_at >= resp2.created_at


def test_invalid_inputs_raise_validation_errors():
    # Missing required fields
    with pytest.raises(Exception):
        LLMRequest(Promptstring="Hi", context=[], model=ModelMetadata(provider="x", model_id="y"))

    # Empty prompt
    with pytest.raises(Exception):
        LLMRequest(request_id="id", Promptstring="  ", context=[], model=ModelMetadata(provider="x", model_id="y"))

    # Wrong context types
    with pytest.raises(Exception):
        LLMRequest(request_id="id", Promptstring="ok", context=[1, 2], model=ModelMetadata(provider="x", model_id="y"))

    # TokenUsage total mismatch
    with pytest.raises(Exception):
        TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=3)

    # Response time ordering invalid
    meta = ModelMetadata(provider="dummy", model_id="dummy-1.0")
    with pytest.raises(Exception):
        LLMResponse(
            request_id="id",
            model=meta,
            text="ok",
            usage=TokenUsage(prompt_tokens=0, completion_tokens=1, total_tokens=1),
            latency_ms=0.1,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            completed_at=datetime(2023, 1, 1, tzinfo=timezone.utc),
        )
