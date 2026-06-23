import unittest
from unittest.mock import patch

from movie_recommender import (
    MOVIE_CATALOG,
    Movie,
    MovieRecommender,
    _format_movies,
    main,
)


class MovieRecommenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recommender = MovieRecommender()

    def test_recommend_returns_ranked_matches(self) -> None:
        results = self.recommender.recommend(["Sci-Fi", "Action"], top_n=3)
        self.assertEqual(len(results), 3)
        self.assertIn("Inception", [movie.title for movie in results])

    def test_recommend_supports_filters(self) -> None:
        results = self.recommender.recommend(["Drama"], min_year=2015, min_rating=8.0, top_n=10)
        self.assertTrue(all(movie.year >= 2015 for movie in results))
        self.assertTrue(all(movie.rating >= 8.0 for movie in results))

    def test_recommend_rejects_empty_preferences(self) -> None:
        with self.assertRaises(ValueError):
            self.recommender.recommend([])

    def test_recommend_similar_raises_for_unknown_title(self) -> None:
        with self.assertRaises(ValueError):
            self.recommender.recommend_similar("Unknown Movie")

    def test_recommend_similar_returns_expected_top_match(self) -> None:
        results = self.recommender.recommend_similar("Inception", top_n=1)
        self.assertEqual(results[0].title, "The Matrix")


# ---------------------------------------------------------------------------
# Tests for uncovered / under-covered code paths
# ---------------------------------------------------------------------------


class TestMovieDataclass(unittest.TestCase):
    def test_movie_is_frozen(self) -> None:
        movie = Movie("Test", ("Action",), 2020, 7.0)
        with self.assertRaises(AttributeError):
            movie.title = "Changed"

    def test_movie_equality(self) -> None:
        a = Movie("Test", ("Action",), 2020, 7.0)
        b = Movie("Test", ("Action",), 2020, 7.0)
        self.assertEqual(a, b)

    def test_movie_catalog_is_populated(self) -> None:
        self.assertGreater(len(MOVIE_CATALOG), 0)
        for movie in MOVIE_CATALOG:
            self.assertIsInstance(movie, Movie)


class TestRecommenderInit(unittest.TestCase):
    def test_default_catalog(self) -> None:
        rec = MovieRecommender()
        self.assertEqual(rec.catalog, MOVIE_CATALOG)

    def test_custom_catalog(self) -> None:
        custom = [Movie("A", ("Drama",), 2000, 5.0)]
        rec = MovieRecommender(catalog=custom)
        self.assertEqual(len(rec.catalog), 1)
        self.assertEqual(rec.catalog[0].title, "A")

    def test_empty_catalog_falls_back_to_default(self) -> None:
        rec = MovieRecommender(catalog=[])
        self.assertEqual(rec.catalog, MOVIE_CATALOG)


class TestRecommendExcludeTitles(unittest.TestCase):
    """Covers the exclude_titles branch (line 64)."""

    def setUp(self) -> None:
        self.recommender = MovieRecommender()

    def test_exclude_titles_removes_movies(self) -> None:
        results = self.recommender.recommend(
            ["Sci-Fi", "Action"],
            exclude_titles=["Inception"],
            top_n=10,
        )
        titles = [m.title for m in results]
        self.assertNotIn("Inception", titles)

    def test_exclude_titles_case_insensitive(self) -> None:
        results = self.recommender.recommend(
            ["Sci-Fi"],
            exclude_titles=["inception", "THE MATRIX"],
            top_n=10,
        )
        titles = [m.title for m in results]
        self.assertNotIn("Inception", titles)
        self.assertNotIn("The Matrix", titles)

    def test_exclude_titles_with_whitespace(self) -> None:
        results = self.recommender.recommend(
            ["Sci-Fi"],
            exclude_titles=["  Inception  "],
            top_n=10,
        )
        self.assertNotIn("Inception", [m.title for m in results])

    def test_exclude_titles_empty_strings_ignored(self) -> None:
        all_results = self.recommender.recommend(["Sci-Fi"], top_n=10)
        results = self.recommender.recommend(
            ["Sci-Fi"],
            exclude_titles=["", "  "],
            top_n=10,
        )
        self.assertEqual(len(results), len(all_results))


class TestRecommendEdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        self.recommender = MovieRecommender()

    def test_whitespace_only_genres_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.recommender.recommend(["  ", ""])

    def test_genre_matching_is_case_insensitive(self) -> None:
        lower = self.recommender.recommend(["sci-fi"], top_n=5)
        upper = self.recommender.recommend(["SCI-FI"], top_n=5)
        self.assertEqual(
            [m.title for m in lower],
            [m.title for m in upper],
        )

    def test_top_n_zero_returns_empty(self) -> None:
        results = self.recommender.recommend(["Action"], top_n=0)
        self.assertEqual(results, [])

    def test_no_genre_overlap_returns_empty(self) -> None:
        results = self.recommender.recommend(["Western"], top_n=10)
        self.assertEqual(results, [])

    def test_min_year_filters_correctly(self) -> None:
        results = self.recommender.recommend(["Action"], min_year=2020, top_n=10)
        self.assertTrue(all(m.year >= 2020 for m in results))

    def test_min_rating_filters_correctly(self) -> None:
        results = self.recommender.recommend(["Action"], min_rating=9.0, top_n=10)
        self.assertTrue(all(m.rating >= 9.0 for m in results))


class TestRecommendSimilarEdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        self.recommender = MovieRecommender()

    def test_recommend_similar_excludes_source(self) -> None:
        results = self.recommender.recommend_similar("Inception", top_n=20)
        self.assertNotIn("Inception", [m.title for m in results])

    def test_recommend_similar_top_n_limits_output(self) -> None:
        results = self.recommender.recommend_similar("Inception", top_n=2)
        self.assertLessEqual(len(results), 2)

    def test_recommend_similar_whitespace_title(self) -> None:
        results = self.recommender.recommend_similar("  inception  ", top_n=1)
        self.assertEqual(len(results), 1)

    def test_recommend_similar_with_no_genre_overlap(self) -> None:
        catalog = [
            Movie("A", ("X",), 2000, 5.0),
            Movie("B", ("Y",), 2001, 6.0),
        ]
        rec = MovieRecommender(catalog=catalog)
        results = rec.recommend_similar("A", top_n=5)
        self.assertEqual(len(results), 1)


class TestScoreMovie(unittest.TestCase):
    def test_zero_overlap_returns_zero(self) -> None:
        movie = Movie("Test", ("Horror",), 2010, 8.0)
        score = MovieRecommender._score_movie(movie, {"comedy"})
        self.assertEqual(score, 0.0)

    def test_full_overlap_high_score(self) -> None:
        movie = Movie("Test", ("Action",), 2020, 9.0)
        score = MovieRecommender._score_movie(movie, {"action"})
        self.assertGreater(score, 0.0)

    def test_recency_score_clamped_to_zero_for_old_movies(self) -> None:
        movie = Movie("Old", ("Drama",), 1980, 8.0)
        score = MovieRecommender._score_movie(movie, {"drama"})
        # recency_score = max(min((1980 - 1990) / 40, 1.0), 0.0) = 0.0
        expected = 1.0 * 0.6 + 0.8 * 0.3 + 0.0 * 0.1
        self.assertAlmostEqual(score, expected)

    def test_recency_score_clamped_to_one_for_far_future(self) -> None:
        movie = Movie("Future", ("Drama",), 2050, 10.0)
        score = MovieRecommender._score_movie(movie, {"drama"})
        # recency_score = max(min((2050 - 1990) / 40, 1.0), 0.0) = 1.0
        expected = 1.0 * 0.6 + 1.0 * 0.3 + 1.0 * 0.1
        self.assertAlmostEqual(score, expected)

    def test_partial_genre_overlap(self) -> None:
        movie = Movie("Mix", ("Action", "Drama"), 2010, 8.0)
        score = MovieRecommender._score_movie(movie, {"action", "comedy", "horror"})
        # overlap = 1/3
        overlap_score = 1 / 3
        rating_score = 0.8
        recency_score = (2010 - 1990) / 40  # = 0.5
        expected = overlap_score * 0.6 + rating_score * 0.3 + recency_score * 0.1
        self.assertAlmostEqual(score, expected)


class TestFormatMovies(unittest.TestCase):
    """Covers _format_movies (lines 112-117)."""

    def test_format_single_movie(self) -> None:
        movies = [Movie("Test", ("Action", "Drama"), 2020, 8.5)]
        result = _format_movies(movies)
        self.assertIn("1.", result)
        self.assertIn("Test", result)
        self.assertIn("2020", result)
        self.assertIn("Action, Drama", result)
        self.assertIn("8.5", result)

    def test_format_multiple_movies(self) -> None:
        movies = [
            Movie("A", ("Drama",), 2000, 7.0),
            Movie("B", ("Comedy",), 2010, 8.0),
        ]
        result = _format_movies(movies)
        lines = result.split("\n")
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("1."))
        self.assertTrue(lines[1].startswith("2."))

    def test_format_empty_list(self) -> None:
        result = _format_movies([])
        self.assertEqual(result, "No recommendations found.")


class TestMain(unittest.TestCase):
    """Covers main() (lines 120-139, 143) via mocked I/O."""

    @patch("builtins.input", side_effect=["Sci-Fi, Action", "", ""])
    @patch("builtins.print")
    def test_main_no_optional_filters(self, mock_print, mock_input) -> None:
        main()
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("Movie recommendation system", printed)
        self.assertIn("Top recommendations", printed)

    @patch("builtins.input", side_effect=["Drama", "2015", "8.0"])
    @patch("builtins.print")
    def test_main_with_filters(self, mock_print, mock_input) -> None:
        main()
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("Top recommendations", printed)

    @patch("builtins.input", side_effect=["Western", "", ""])
    @patch("builtins.print")
    def test_main_no_matching_genre(self, mock_print, mock_input) -> None:
        main()
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("No recommendations found.", printed)


if __name__ == "__main__":
    unittest.main()
