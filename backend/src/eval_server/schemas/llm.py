from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator, ConfigDict


class ModelMetadata(BaseModel):
    """Minimal model/provider identification metadata for an LLM call."""

    provider: str = Field(description="Provider registry name, e.g., 'openai', 'dummy'")
    model_id: str = Field(description="Model identifier or version for the provider")

    @field_validator("provider", "model_id")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must be a non-empty string")
        return v


class TokenUsage(BaseModel):
    """Token accounting for a single generation."""

    prompt_tokens: int = Field(ge=0, default=0)
    completion_tokens: int = Field(ge=0, default=0)
    total_tokens: int = Field(ge=0, default=0)

    @model_validator(mode="after")
    def _check_totals(self) -> "TokenUsage":
        expected = self.prompt_tokens + self.completion_tokens
        if self.total_tokens == 0:
            # Auto-compute if not set explicitly
            object.__setattr__(self, "total_tokens", expected)
        elif self.total_tokens != expected:
            raise ValueError("total_tokens must equal prompt_tokens + completion_tokens")
        return self


class LLMRequest(BaseModel):
    """Schema for an LLM generation request."""

    # Accept both alias (Promptstring) and field name (prompt) during parsing
    model_config = ConfigDict(populate_by_name=True)

    request_id: str = Field(description="Unique id for correlating request/response")
    prompt: str = Field(alias="Promptstring", description="The prompt to generate from")
    context: List[str] = Field(default_factory=list, description="Conversation context strings")
    model: ModelMetadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("request_id")
    @classmethod
    def _valid_req_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("request_id must be non-empty")
        return v

    @field_validator("prompt")
    @classmethod
    def _non_empty_prompt(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Promptstring must be a non-empty string")
        return v

    @field_validator("context")
    @classmethod
    def _normalize_context(cls, v: List[str]) -> List[str]:
        cleaned: List[str] = []
        for item in v:
            if not isinstance(item, str):
                raise TypeError("context entries must be strings")
            s = item.strip()
            if s:
                cleaned.append(s)
        return cleaned

    @field_validator("created_at")
    @classmethod
    def _ensure_timezone(cls, dt: datetime) -> datetime:
        # Ensure timezone-aware; default is UTC
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt


class LLMResponse(BaseModel):
    """Schema for an LLM generation response."""

    request_id: str = Field(description="Echo of request id for correlation")
    model: ModelMetadata
    text: str = Field(description="Generated completion text")
    usage: TokenUsage
    latency_ms: float = Field(ge=0.0, description="Latency for the call in milliseconds")
    provider_metadata: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific metadata")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("request_id")
    @classmethod
    def _non_empty_req_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("request_id must be non-empty")
        return v

    @field_validator("text")
    @classmethod
    def _non_empty_text(cls, v: str) -> str:
        if not isinstance(v, str) or v == "":
            raise ValueError("text must be a non-empty string (can be whitespace)")
        return v

    @field_validator("created_at", "completed_at")
    @classmethod
    def _ensure_tz(cls, dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    @model_validator(mode="after")
    def _time_ordering(self) -> "LLMResponse":
        if self.completed_at < self.created_at:
            raise ValueError("completed_at must be greater than or equal to created_at")
        return self


__all__ = [
    "ModelMetadata",
    "TokenUsage",
    "LLMRequest",
    "LLMResponse",
]
