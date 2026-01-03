from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True, order=True)
class SemVer:
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


def parse_version(s: str) -> SemVer:
    s = s.strip()
    parts = s.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"Invalid semantic version: '{s}'. Expected 'MAJOR.MINOR.PATCH'")
    major, minor, patch = (int(p) for p in parts)
    if major < 0 or minor < 0 or patch < 0:
        raise ValueError("Version numbers must be non-negative")
    return SemVer(major, minor, patch)


def compare_versions(a: str, b: str) -> int:
    va, vb = parse_version(a), parse_version(b)
    if va < vb:
        return -1
    if va > vb:
        return 1
    return 0


def is_version_greater(a: str, b: str) -> bool:
    return compare_versions(a, b) > 0


def versioned_name(base: str, version: str, *, sep: str = "-") -> str:
    # Generate versioned artifact names like results-1.0.0
    _ = parse_version(version)  # validate
    return f"{base}{sep}{version}"
