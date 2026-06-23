import unittest

from movie_recommender import MovieRecommender, _parse_year, _parse_rating


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

    def test_recommend_rejects_non_positive_top_n(self) -> None:
        with self.assertRaises(ValueError):
            self.recommender.recommend(["Sci-Fi"], top_n=0)
        with self.assertRaises(ValueError):
            self.recommender.recommend(["Sci-Fi"], top_n=-1)

    def test_recommend_similar_rejects_non_positive_top_n(self) -> None:
        with self.assertRaises(ValueError):
            self.recommender.recommend_similar("Inception", top_n=0)


class ParseYearTests(unittest.TestCase):
    def test_empty_returns_none(self) -> None:
        self.assertIsNone(_parse_year(""))

    def test_valid_year(self) -> None:
        self.assertEqual(_parse_year("2020"), 2020)

    def test_non_numeric_exits(self) -> None:
        with self.assertRaises(SystemExit):
            _parse_year("abc")

    def test_out_of_range_exits(self) -> None:
        with self.assertRaises(SystemExit):
            _parse_year("1800")
        with self.assertRaises(SystemExit):
            _parse_year("2200")


class ParseRatingTests(unittest.TestCase):
    def test_empty_returns_none(self) -> None:
        self.assertIsNone(_parse_rating(""))

    def test_valid_rating(self) -> None:
        self.assertAlmostEqual(_parse_rating("8.5"), 8.5)

    def test_non_numeric_exits(self) -> None:
        with self.assertRaises(SystemExit):
            _parse_rating("high")

    def test_out_of_range_exits(self) -> None:
        with self.assertRaises(SystemExit):
            _parse_rating("-1")
        with self.assertRaises(SystemExit):
            _parse_rating("11")


if __name__ == "__main__":
    unittest.main()
