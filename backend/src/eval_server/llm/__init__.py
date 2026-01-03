"""LLM provider registry and adapters.

Import common providers to trigger auto-registration.
"""

# Ensure providers are imported so they register themselves
# noqa: F401 - imported for side effects
from .providers import (
    dummy_provider,  # noqa: F401
    openai_provider,  # noqa: F401
    azure_openai_provider,  # noqa: F401
    google_provider,  # noqa: F401
    aws_bedrock_provider,  # noqa: F401
    ollama_provider,  # noqa: F401
)

from .provider_registry import (
    LLMProvider,
    register_provider,
    get_provider_class,
    create_provider,
    list_providers,
)

__all__ = [
    "LLMProvider",
    "register_provider",
    "get_provider_class",
    "create_provider",
    "list_providers",
]
