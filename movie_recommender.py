"""Simple movie recommendation system.

This module provides a lightweight recommendation engine that uses
content-based filtering (genres, release year, and community rating)
for suggesting movies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from utils import jaccard_similarity, normalize_genres, normalize_title, top_n_by_score


@dataclass(frozen=True)
class Movie:
    title: str
    genres: tuple[str, ...]
    year: int
    rating: float


MOVIE_CATALOG: tuple[Movie, ...] = (
    Movie("Inception", ("Sci-Fi", "Action", "Thriller"), 2010, 8.8),
    Movie("The Dark Knight", ("Action", "Crime", "Drama"), 2008, 9.0),
    Movie("Interstellar", ("Sci-Fi", "Drama", "Adventure"), 2014, 8.7),
    Movie("The Matrix", ("Sci-Fi", "Action"), 1999, 8.7),
    Movie("Parasite", ("Thriller", "Drama"), 2019, 8.5),
    Movie("La La Land", ("Romance", "Drama", "Music"), 2016, 8.0),
    Movie("Mad Max: Fury Road", ("Action", "Adventure", "Sci-Fi"), 2015, 8.1),
    Movie("The Grand Budapest Hotel", ("Comedy", "Adventure", "Drama"), 2014, 8.1),
    Movie("Spirited Away", ("Animation", "Fantasy", "Adventure"), 2001, 8.6),
    Movie("Everything Everywhere All at Once", ("Sci-Fi", "Comedy", "Action"), 2022, 7.8),
    Movie("Whiplash", ("Drama", "Music"), 2014, 8.5),
    Movie("Get Out", ("Horror", "Mystery", "Thriller"), 2017, 7.7),
)


class MovieRecommender:
    """Recommend movies from a fixed catalog using weighted relevance scoring."""

    def __init__(self, catalog: Sequence[Movie] | None = None) -> None:
        self.catalog = tuple(catalog or MOVIE_CATALOG)
        self._titles = {normalize_title(movie.title): movie for movie in self.catalog}

    def recommend(
        self,
        preferred_genres: Iterable[str],
        *,
        min_year: int | None = None,
        min_rating: float | None = None,
        top_n: int = 5,
        exclude_titles: Iterable[str] | None = None,
    ) -> list[Movie]:
        """Return top-N recommendations ranked by relevance score."""
        normalized = normalize_genres(preferred_genres)
        if not normalized:
            raise ValueError("preferred_genres must contain at least one genre")

        excluded = {normalize_title(t) for t in (exclude_titles or []) if t.strip()}

        scored: list[tuple[float, Movie]] = []
        for movie in self.catalog:
            if normalize_title(movie.title) in excluded:
                continue
            if min_year is not None and movie.year < min_year:
                continue
            if min_rating is not None and movie.rating < min_rating:
                continue

            score = self._score_movie(movie, normalized)
            if score > 0:
                scored.append((score, movie))

        return top_n_by_score(scored, top_n, tiebreakers=("rating", "year"))

    def recommend_similar(self, title: str, top_n: int = 5) -> list[Movie]:
        """Recommend movies that share the strongest genre similarity with *title*."""
        source = self._titles.get(normalize_title(title))
        if source is None:
            raise ValueError(f"Unknown title: {title!r}")

        similarities: list[tuple[float, Movie]] = []
        source_genres = normalize_genres(source.genres)
        for movie in self.catalog:
            if movie.title == source.title:
                continue

            target_genres = normalize_genres(movie.genres)
            jaccard = jaccard_similarity(source_genres, target_genres)
            weighted = jaccard * 0.8 + (movie.rating / 10) * 0.2
            similarities.append((weighted, movie))

        return top_n_by_score(similarities, top_n, tiebreakers=("rating",))

    @staticmethod
    def _score_movie(movie: Movie, preferred_genres: set[str]) -> float:
        movie_genres = normalize_genres(movie.genres)
        genre_overlap = len(movie_genres & preferred_genres)
        if genre_overlap == 0:
            return 0.0

        overlap_score = genre_overlap / max(len(preferred_genres), 1)
        rating_score = movie.rating / 10
        recency_score = max(min((movie.year - 1990) / 40, 1.0), 0.0)
        return overlap_score * 0.6 + rating_score * 0.3 + recency_score * 0.1


def _format_movies(movies: Sequence[Movie]) -> str:
    lines = []
    for index, movie in enumerate(movies, start=1):
        genres = ", ".join(movie.genres)
        lines.append(f"{index}. {movie.title} ({movie.year}) — {genres} — rating {movie.rating}")
    return "\n".join(lines) if lines else "No recommendations found."


def main() -> None:
    recommender = MovieRecommender()
    print("Movie recommendation system")
    genre_input = input("Enter preferred genres (comma-separated): ").strip()
    genres = [item.strip() for item in genre_input.split(",")]
    min_year_raw = input("Minimum release year (optional): ").strip()
    min_rating_raw = input("Minimum rating out of 10 (optional): ").strip()

    min_year = int(min_year_raw) if min_year_raw else None
    min_rating = float(min_rating_raw) if min_rating_raw else None

    recommendations = recommender.recommend(
        genres,
        min_year=min_year,
        min_rating=min_rating,
        top_n=5,
    )

    print("\nTop recommendations:")
    print(_format_movies(recommendations))


if __name__ == "__main__":
    main()
