"""Shared utility functions for text normalization and scored ranking."""

from __future__ import annotations

from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")


def normalize_genres(genres: Iterable[str]) -> set[str]:
    """Normalize an iterable of genre strings to a lowercase, stripped set."""
    return {genre.strip().lower() for genre in genres if genre.strip()}


def normalize_title(title: str) -> str:
    """Normalize a title string by lowercasing and stripping whitespace."""
    return title.lower().strip()


def top_n_by_score(
    scored_items: Sequence[tuple[float, T]],
    n: int,
    tiebreakers: tuple[str, ...] = ("rating", "year"),
) -> list[T]:
    """Sort (score, item) pairs descending and return the top N items.

    Tiebreakers are resolved by looking up attributes on the item in order.
    """

    def sort_key(item: tuple[float, T]) -> tuple[float, ...]:
        score, obj = item
        keys: list[float] = [score]
        for attr in tiebreakers:
            keys.append(getattr(obj, attr, 0))
        return tuple(keys)

    sorted_items = sorted(scored_items, key=sort_key, reverse=True)
    return [item for _, item in sorted_items[:n]]


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two string sets."""
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return len(set_a & set_b) / union
