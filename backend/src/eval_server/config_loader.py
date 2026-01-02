from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


class ModelSettings(BaseModel):
    provider: str = Field(description="LLM provider registry name")
    model_id: str = Field(description="Model identifier/version for the provider")
    temperature: float = Field(ge=0.0, le=1.0, description="Sampling temperature 0..1")
    max_tokens: int = Field(gt=0, description="Maximum tokens for generation")
    timeout_sec: int = Field(gt=0, description="Request timeout in seconds")


class DatasetSettings(BaseModel):
    root_dir: str = Field(description="Default dataset root directory")
    allowed_extensions: list[str] = Field(default_factory=list, description="Permitted file extensions")

    @field_validator("allowed_extensions")
    @classmethod
    def _normalize_exts(cls, v: list[str]) -> list[str]:
        # Ensure extensions are normalized like '.json'
        norm = []
        for e in v:
            e = e.strip()
            if not e:
                continue
            if not e.startswith('.'):
                e = '.' + e
            norm.append(e.lower())
        return norm


class ScoringWeights(BaseModel):
    correctness: float = 0.6
    consistency: float = 0.2
    adherence: float = 0.15
    hallucination: float = 0.05

    @field_validator("correctness", "consistency", "adherence", "hallucination")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Weights must be non-negative")
        return v


class ScoringSettings(BaseModel):
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    pass_threshold: float = Field(ge=0.0, le=1.0, default=0.8)
    aggregation: Literal["mean", "min", "weighted"] = "mean"


class ReportSettings(BaseModel):
    output_dir: str = Field(description="Directory for report artifacts")
    formats: list[Literal["html", "json", "csv", "md"]] = Field(default_factory=lambda: ["html", "json"]) 
    include_raw: bool = True


class UISettings(BaseModel):
    enabled: bool = True
    theme: Literal["light", "dark", "system"] = "system"
    page_size: int = Field(gt=0, default=25)


class Settings(BaseModel):
    version: int = 1
    models: ModelSettings
    datasets: DatasetSettings
    scoring: ScoringSettings
    report: ReportSettings
    ui: UISettings


def _default_settings_path() -> Path:
    # Resolve repository root based on this file location:
    # <repo>/backend/src/eval_server/config_loader.py -> parents[3] = <repo>
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "configs" / "settings.yaml"


def load_settings(file_path: Optional[str | Path] = None) -> Settings:
    """Load settings.yaml into a typed Settings object.

    If file_path is None, loads from the repository default path
    '<repo>/configs/settings.yaml'.
    """
    path = Path(file_path) if file_path else _default_settings_path()
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    try:
        return Settings.model_validate(data)
    except ValidationError as e:
        # Re-raise for callers/tests to assert
        raise


__all__ = [
    "Settings",
    "ModelSettings",
    "DatasetSettings",
    "ScoringSettings",
    "ScoringWeights",
    "ReportSettings",
    "UISettings",
    "load_settings",
]
