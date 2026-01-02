from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Type


class LLMProvider(ABC):
    """Abstract base class for LLM provider adapters."""

    @abstractmethod
    def initialize(self, **kwargs: Any) -> None:
        """Initialize the provider with configuration/secrets."""

    @abstractmethod
    def generate(self, prompt: str, context: List[str] | None = None) -> str:
        """Generate a completion deterministically for testing where possible."""

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Return provider metadata such as name, version, capabilities."""


_REGISTRY: MutableMapping[str, Type[LLMProvider]] = {}


def register_provider(name: str, cls: Type[LLMProvider]) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("Provider name must be non-empty")
    if key in _REGISTRY:
        raise ValueError(f"Provider already registered: {name}")
    if not issubclass(cls, LLMProvider):
        raise TypeError("cls must be a subclass of LLMProvider")
    _REGISTRY[key] = cls


def get_provider_class(name: str) -> Type[LLMProvider]:
    key = name.strip().lower()
    try:
        return _REGISTRY[key]
    except KeyError as e:
        raise KeyError(f"Provider not found: {name}") from e


def create_provider(name: str, **init_kwargs: Any) -> LLMProvider:
    cls = get_provider_class(name)
    provider = cls()  # type: ignore[call-arg]
    provider.initialize(**init_kwargs)
    return provider


def list_providers() -> List[str]:
    return sorted(_REGISTRY.keys())
