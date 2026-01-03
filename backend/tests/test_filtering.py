from __future__ import annotations

from eval_server.data.filtering import (
    filter_by_difficulty,
    filter_by_tags,
    filter_datasets,
)


ITEMS = [
    {
        "conversation_id": "c1",
        "metadata": {"tags": ["search", "support"], "difficulty": "easy"},
    },
    {
        "conversation_id": "c2",
        "metadata": {"tags": ["checkout"], "difficulty": "hard"},
    },
    {
        "conversation_id": "c3",
        "metadata": {"tags": ["returns"], "difficulty": "medium"},
    },
    {
        "conversation_id": "c4",
        "metadata": {"tags": ["SUPPORT"], "difficulty": "HARD"},
    },
]


def ids(items):
    return [it["conversation_id"] for it in items]


def test_filter_by_tags_any_all_exclude():
    # include any of {support, returns}
    r1 = filter_by_tags(ITEMS, include_any=["support", "returns"])
    assert set(ids(r1)) == {"c1", "c3", "c4"}

    # include all: requires both search and support
    r2 = filter_by_tags(ITEMS, include_all=["search", "support"])
    assert ids(r2) == ["c1"]

    # exclude checkout
    r3 = filter_by_tags(ITEMS, exclude=["checkout"])
    assert set(ids(r3)) == {"c1", "c3", "c4"}

    # combine any + exclude
    r4 = filter_by_tags(ITEMS, include_any=["support"], exclude=["returns"])
    assert set(ids(r4)) == {"c1", "c4"}


def test_filter_by_difficulty():
    r1 = filter_by_difficulty(ITEMS, allowed=["easy", "medium"])  # case-insensitive
    assert set(ids(r1)) == {"c1", "c3"}

    r2 = filter_by_difficulty(ITEMS, allowed=["hard"])  # picks c2,c4
    assert set(ids(r2)) == {"c2", "c4"}


def test_filter_combined():
    r = filter_datasets(
        ITEMS,
        tags_any=["support"],
        exclude_tags=["returns"],
        difficulties=["hard"],
    )
    assert set(ids(r)) == {"c4"}
