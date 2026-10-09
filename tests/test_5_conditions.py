"""
Unit test asserting all 5 core tokenizer criteria:
1. Unicode Apostrophe normalization
2. Morphophonological sound changes (Q/G', K/G, Vowel syncope)
3. Word boundary markers (Ġroot, ##suffix)
4. Unified vocabulary (vocab.json) and numeric encoding/decoding
5. High-performance caching and Trie structures
"""
import unittest
from uztoken import UzTokenizer, normalize_uzbek, SuffixTrie, DictionaryTrie


class TestFiveConditions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = UzTokenizer(max_depth=12)

    def test_condition_1_unicode_apostrophe_normalization(self):
        """1-shart: Unicode apostroflarni standartlashtirish va bo'shliqlarni tozalash."""
        sample_raw = "sho’r Oʻzbekiston g‘alaba bog` o  '  tgan"
        sample_clean = normalize_uzbek(sample_raw)

        # Hech qanday g'ayrioddiy apostrof yoki belgi qolmasligi shart
        self.assertNotIn("’", sample_clean)
        self.assertNotIn("ʻ", sample_clean)
        self.assertNotIn("‘", sample_clean)
        self.assertNotIn("`", sample_clean)
        self.assertIn("o'zbekiston", sample_clean)
        self.assertIn("g'alaba", sample_clean)
        self.assertIn("o'tgan", sample_clean)

    def test_condition_2_morphophonology(self):
        """2-shart: Morfonologik tovush o'zgarishlari (Q/G', K/G va unli tushishi)."""
        cases = [
            ("qishlog'imiz", "qishloq"),
            ("yuragimda", "yurak"),
            ("shahrimiz", "shahar"),
            ("singlimga", "singil"),
            ("burni", "burun"),
        ]
        for word, expected_root in cases:
            res = self.tok.analyze(word)
            self.assertTrue(res.is_found, f"Failed for {word}")
            self.assertEqual(res.root, expected_root)

    def test_condition_3_word_boundary_markers(self):
        """3-shart: So'z chegarasi belgilari (Ġ va ##)."""
        subwords = self.tok.tokenize_with_boundaries("kitoblarimizga", style="hybrid")
        self.assertEqual(subwords, ["Ġkitob", "##lar", "##imiz", "##ga"])

        # BERT usuli
        subwords_bert = self.tok.tokenize_with_boundaries("kitoblarimizga", style="bert")
        self.assertEqual(subwords_bert, ["kitob", "##lar", "##imiz", "##ga"])

    def test_condition_4_unified_vocab_and_encoding(self):
        """4-shart: Yagona vocab.json va raqamli ID'larga kodlash/dekodlash."""
        vocab = self.tok.build_unified_vocab()

        # Special tokenlar mavjudligi
        self.assertEqual(vocab["<pad>"], 0)
        self.assertEqual(vocab["<unk>"], 1)
        self.assertEqual(vocab["<s>"], 2)

        # O'zak va qo'shimchalar raqamlanganligi
        self.assertIn("Ġkitob", vocab)
        self.assertIn("##lar", vocab)
        self.assertIn("##imiz", vocab)

        # Encode va Decode
        sample_text = "Maktablarimizda kitob o'qiymiz."
        ids = self.tok.encode(sample_text)
        self.assertIsInstance(ids, list)
        self.assertTrue(all(isinstance(i, int) for i in ids))

        decoded = self.tok.decode(ids)
        self.assertIn("maktablarimizda", decoded)
        self.assertIn("kitob", decoded)

    def test_condition_5_speed_and_caching(self):
        """5-shart: Trie tuzilmasi va LRU kesh orqali tezkor ishlash."""
        # SuffixTrie mavjud va to'la
        self.assertGreater(len(self.tok._analyzer._suffix_trie), 50)

        # Kesh ishlashini tekshirish
        self.tok._analyzer.clear_cache()
        self.assertEqual(len(self.tok._analyzer._cache), 0)

        self.tok.analyze("maktablarimizda")
        self.assertIn("maktablarimizda", self.tok._analyzer._cache)


if __name__ == "__main__":
    unittest.main()
