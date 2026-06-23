import unittest

from movie_recommender import MovieRecommender


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


if __name__ == "__main__":
    unittest.main()
