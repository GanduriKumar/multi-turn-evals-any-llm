from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence


ConversationDict = Mapping[str, Any]


def _get_tags(item: ConversationDict, *, case_insensitive: bool) -> List[str]:
    md = item.get("metadata", {}) if isinstance(item, Mapping) else {}
    tags = md.get("tags", []) if isinstance(md, Mapping) else []
    result: List[str] = []
    for t in tags or []:
        if not isinstance(t, str):
            continue
        result.append(t.lower() if case_insensitive else t)
    return result


def _get_difficulty(item: ConversationDict, *, case_insensitive: bool) -> Optional[str]:
    md = item.get("metadata", {}) if isinstance(item, Mapping) else {}
    diff = md.get("difficulty") if isinstance(md, Mapping) else None
    if isinstance(diff, str):
        return diff.lower() if case_insensitive else diff
    return None


def filter_by_tags(
    items: Sequence[ConversationDict],
    *,
    include_any: Optional[Iterable[str]] = None,
    include_all: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    case_insensitive: bool = True,
) -> List[ConversationDict]:
    """Filter conversations by metadata.tags.

    - include_any: keep items having any of these tags.
    - include_all: keep items having all of these tags.
    - exclude: drop items having any of these tags.
    - Matching is case-insensitive by default.
    """
    any_set = {t.lower() for t in include_any} if include_any else set()
    all_set = {t.lower() for t in include_all} if include_all else set()
    exc_set = {t.lower() for t in exclude} if exclude else set()

    out: List[ConversationDict] = []
    for it in items:
        tags = _get_tags(it, case_insensitive=case_insensitive)
        tag_set = set(tags)

        if exc_set and tag_set & exc_set:
            continue
        if any_set and not (tag_set & any_set):
            continue
        if all_set and not all_set.issubset(tag_set):
            continue
        out.append(it)
    return out


def filter_by_difficulty(
    items: Sequence[ConversationDict],
    *,
    allowed: Optional[Iterable[str]] = None,
    case_insensitive: bool = True,
) -> List[ConversationDict]:
    """Filter conversations by metadata.difficulty.

    - allowed: iterable of allowed difficulty values (e.g., ["easy", "hard"]).
    - If allowed is None or empty, returns items unchanged.
    - Matching is case-insensitive by default.
    """
    if not allowed:
        return list(items)
    allow_set = {d.lower() for d in allowed}
    out: List[ConversationDict] = []
    for it in items:
        diff = _get_difficulty(it, case_insensitive=case_insensitive)
        if diff is None:
            continue
        if diff in allow_set:
            out.append(it)
    return out


def filter_datasets(
    items: Sequence[ConversationDict],
    *,
    tags_any: Optional[Iterable[str]] = None,
    tags_all: Optional[Iterable[str]] = None,
    exclude_tags: Optional[Iterable[str]] = None,
    difficulties: Optional[Iterable[str]] = None,
    case_insensitive: bool = True,
) -> List[ConversationDict]:
    """Filter helper combining tags and difficulty criteria.

    Applies tag filters first, then difficulty filtering.
    """
    tmp = filter_by_tags(
        items,
        include_any=tags_any,
        include_all=tags_all,
        exclude=exclude_tags,
        case_insensitive=case_insensitive,
    )
    return filter_by_difficulty(tmp, allowed=difficulties, case_insensitive=case_insensitive)


__all__ = [
    "ConversationDict",
    "filter_by_tags",
    "filter_by_difficulty",
    "filter_datasets",
]
