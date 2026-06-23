import unittest

from utils import jaccard_similarity, normalize_genres, normalize_title, top_n_by_score


class NormalizeGenresTests(unittest.TestCase):
    def test_lowercases_and_strips(self) -> None:
        result = normalize_genres(["  Sci-Fi ", "ACTION", " drama"])
        self.assertEqual(result, {"sci-fi", "action", "drama"})

    def test_filters_empty_strings(self) -> None:
        result = normalize_genres(["Action", "", "  ", "Drama"])
        self.assertEqual(result, {"action", "drama"})

    def test_returns_empty_set_for_no_valid_input(self) -> None:
        result = normalize_genres(["", "  "])
        self.assertEqual(result, set())


class NormalizeTitleTests(unittest.TestCase):
    def test_lowercases_and_strips(self) -> None:
        self.assertEqual(normalize_title("  The Matrix  "), "the matrix")

    def test_empty_string(self) -> None:
        self.assertEqual(normalize_title(""), "")


class TopNByScoreTests(unittest.TestCase):
    def test_returns_top_n_items(self) -> None:

        class Item:
            def __init__(self, name: str, rating: float, year: int) -> None:
                self.name = name
                self.rating = rating
                self.year = year

        items = [
            (0.5, Item("a", 7.0, 2010)),
            (0.9, Item("b", 8.0, 2015)),
            (0.7, Item("c", 9.0, 2020)),
        ]
        result = top_n_by_score(items, 2, tiebreakers=("rating", "year"))
        self.assertEqual(result[0].name, "b")
        self.assertEqual(result[1].name, "c")
        self.assertEqual(len(result), 2)

    def test_handles_empty_input(self) -> None:
        self.assertEqual(top_n_by_score([], 5), [])


class JaccardSimilarityTests(unittest.TestCase):
    def test_identical_sets(self) -> None:
        self.assertAlmostEqual(jaccard_similarity({"a", "b"}, {"a", "b"}), 1.0)

    def test_disjoint_sets(self) -> None:
        self.assertAlmostEqual(jaccard_similarity({"a"}, {"b"}), 0.0)

    def test_partial_overlap(self) -> None:
        self.assertAlmostEqual(
            jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"}), 0.5
        )

    def test_empty_sets(self) -> None:
        self.assertAlmostEqual(jaccard_similarity(set(), set()), 0.0)


if __name__ == "__main__":
    unittest.main()
