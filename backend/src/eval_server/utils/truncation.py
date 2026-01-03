from __future__ import annotations

from typing import Callable, List, Sequence


Truncator = Callable[[List[str]], List[str]]


def full_history() -> Truncator:
    def _apply(lines: List[str]) -> List[str]:
        return list(lines)

    return _apply


def windowed_history(window_size: int) -> Truncator:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")

    def _apply(lines: List[str]) -> List[str]:
        if len(lines) <= window_size:
            return list(lines)
        return list(lines[-window_size:])

    return _apply


def summarized_history(max_items: int = 5) -> Truncator:
    if max_items <= 0:
        raise ValueError("max_items must be > 0")

    def _apply(lines: List[str]) -> List[str]:
        # Keep only the last N items and coalesce into a single summary line
        last = lines[-max_items:] if len(lines) > max_items else list(lines)
        compact = " | ".join(last)
        return [f"SUMMARY: {compact}"]

    return _apply


def make_truncator(policy: str, **params: int) -> Truncator:
    name = (policy or "").strip().lower()
    if name in ("full", "none", "all"):
        return full_history()
    if name in ("window", "windowed", "windowed_history"):
        return windowed_history(int(params.get("window_size", 10)))
    if name in ("summary", "summarized", "summarized_history"):
        return summarized_history(int(params.get("summary_max_items", 5)))
    # default to full
    return full_history()


__all__ = [
    "Truncator",
    "full_history",
    "windowed_history",
    "summarized_history",
    "make_truncator",
]
