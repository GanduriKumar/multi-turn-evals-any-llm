from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
from datetime import datetime

@dataclass
class ProviderRequest:
    model: str
    messages: List[Dict[str, str]]  # [{role, content}]
    metadata: Dict[str, Any]

@dataclass
class ProviderResponse:
    ok: bool
    content: str
    latency_ms: int
    provider_meta: Dict[str, Any]
    error: Optional[str] = None
    created_at: str = datetime.utcnow().isoformat() + 'Z'
