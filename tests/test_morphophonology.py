"""
Unit tests for morphophonological sound changes in Uzbek:
1. Q -> G' alternation (qishloq -> qishlog'i, o'rtoq -> o'rtog'im, buloq -> bulog'i)
2. K -> G alternation (yurak -> yuragi, tilak -> tilagim)
3. Vowel syncope (shahar -> shahri, singil -> singlim, burun -> burnim, o'g'il -> o'g'li)
"""
import unittest
from uztoken.dictionary import Dictionary
from uztoken.morphology import MorphAnalyzer


class TestMorphophonology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dict = Dictionary()
        cls.analyzer = MorphAnalyzer(cls.dict, max_depth=10)

    def test_q_to_g_alternation(self):
        """Test Q -> G' alternation when vowel suffix is attached."""
        test_cases = [
            ("qishlog'i", "qishloq"),
            ("qishlog'imiz", "qishloq"),
            ("qishlog'imga", "qishloq"),
            ("o'rtog'im", "o'rtoq"),
            ("tayog'i", "tayoq"),
            ("bulog'i", "buloq"),
            ("qulog'im", "quloq"),
            ("oyog'i", "oyoq"),
        ]
        for word, expected_root in test_cases:
            with self.subTest(word=word):
                result = self.analyzer.analyze(word)
                self.assertTrue(result.is_found, f"Failed to analyze '{word}'")
                self.assertEqual(
                    result.root.replace("\u02BB", "'").replace("\u02BC", "'"),
                    expected_root.replace("\u02BB", "'").replace("\u02BC", "'"),
                    f"Word '{word}' should have root '{expected_root}', got '{result.root}'"
                )

    def test_k_to_g_alternation(self):
        """Test K -> G alternation when vowel suffix is attached."""
        test_cases = [
            ("yuragi", "yurak"),
            ("yuragim", "yurak"),
            ("yuragimizda", "yurak"),
            ("tilagim", "tilak"),
            ("tilagingiz", "tilak"),
        ]
        for word, expected_root in test_cases:
            with self.subTest(word=word):
                result = self.analyzer.analyze(word)
                self.assertTrue(result.is_found, f"Failed to analyze '{word}'")
                self.assertEqual(
                    result.root, expected_root,
                    f"Word '{word}' should have root '{expected_root}', got '{result.root}'"
                )

    def test_vowel_syncope(self):
        """Test vowel syncope (dropping of unstressed vowel) in specific noun stems."""
        test_cases = [
            ("shahri", "shahar"),
            ("shahrimizda", "shahar"),
            ("singlim", "singil"),
            ("singlingiz", "singil"),
            ("burni", "burun"),
            ("burnim", "burun"),
            ("o'g'li", "o'g'il"),
            ("o'g'lim", "o'g'il"),
            ("bag'rim", "bag'ir"),
            ("ko'nglim", "ko'ngil"),
            ("qorni", "qorin"),
            ("qornim", "qorin"),
            ("bo'yni", "bo'yin"),
            ("og'zi", "og'iz"),
        ]
        for word, expected_root in test_cases:
            with self.subTest(word=word):
                result = self.analyzer.analyze(word)
                self.assertTrue(result.is_found, f"Failed to analyze '{word}'")
                self.assertEqual(
                    result.root.replace("\u02BB", "'").replace("\u02BC", "'"),
                    expected_root.replace("\u02BB", "'").replace("\u02BC", "'"),
                    f"Word '{word}' should have root '{expected_root}', got '{result.root}'"
                )

    def test_no_false_positive_on_natural_g_or_g_prime(self):
        """Ensure words naturally ending in g' or g are not falsely alternated."""
        # 'bog'' is in dictionary, so 'bog'i' -> root should be 'bog'', not 'boq'
        result = self.analyzer.analyze("bog'i")
        self.assertTrue(result.is_found)
        self.assertEqual(result.root.replace("\u02BB", "'").replace("\u02BC", "'"), "bog'")

        result = self.analyzer.analyze("tog'imiz")
        self.assertTrue(result.is_found)
        self.assertEqual(result.root.replace("\u02BB", "'").replace("\u02BC", "'"), "tog'")


if __name__ == "__main__":
    unittest.main()
