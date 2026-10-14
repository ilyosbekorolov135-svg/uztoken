"""Unit tests for uztoken.trie module."""

import unittest
import numpy as np

from uztoken.trie import (
    TrieNode,
    SuffixTrie,
    DictionaryTrie,
    batch_lookup,
    batch_find_suffixes,
    HAS_NUMPY,
)


class TestTrieNode(unittest.TestCase):
    def test_node_initialization(self):
        node = TrieNode()
        self.assertFalse(node.is_end)
        # data is a list (supports multiple payloads per suffix)
        self.assertEqual(node.data, [])
        self.assertEqual(node.children, {})

    def test_node_slots(self):
        node = TrieNode()
        # Should have __slots__ and not allow dynamic attributes
        with self.assertRaises(AttributeError):
            node.extra_field = 123  # type: ignore[attr-defined]

    def test_node_custom_data(self):
        data = [{"pos": "noun", "case": "locative"}]
        node = TrieNode(data=data, is_end=True)
        self.assertTrue(node.is_end)
        self.assertEqual(node.data, data)


class TestSuffixTrie(unittest.TestCase):
    def setUp(self):
        self.trie = SuffixTrie()
        # Common Uzbek suffixes
        self.suffixes = [
            ("da", {"type": "case", "name": "o'rin-payt"}),
            ("dan", {"type": "case", "name": "chiqish"}),
            ("ga", {"type": "case", "name": "jo'nalish"}),
            ("ni", {"type": "case", "name": "tushum"}),
            ("ning", {"type": "case", "name": "qaratqich"}),
            ("lar", {"type": "plural", "name": "ko'plik"}),
            ("imiz", {"type": "possessive", "person": "1pl"}),
            ("imizda", {"type": "compound", "parts": ["imiz", "da"]}),
            ("larimizda", {"type": "compound", "parts": ["lar", "imiz", "da"]}),
            ("roq", {"type": "comparative", "name": "qiyosiy"}),
            ("cha", {"type": "diminutive", "name": "kichraytirish"}),
        ]
        for s, d in self.suffixes:
            self.trie.insert(s, d)

    def test_len(self):
        self.assertEqual(len(self.trie), len(self.suffixes))
        # Re-insert existing suffix should not increase length
        self.trie.insert("da", {"type": "updated"})
        self.assertEqual(len(self.trie), len(self.suffixes))

    def test_contains_and_search(self):
        self.assertTrue("lar" in self.trie)
        self.assertTrue("imizda" in self.trie)
        self.assertFalse("mas" in self.trie)
        self.assertFalse("kitob" in self.trie)
        self.assertTrue(self.trie.search("dan"))
        self.assertFalse(self.trie.search("nonexistent"))

    def test_get(self):
        # get() returns list of dicts (multiple payloads per suffix)
        data = self.trie.get("da")
        self.assertIsNotNone(data)
        self.assertIsInstance(data, list)
        self.assertEqual(data[0]["type"], "case")
        self.assertIsNone(self.trie.get("nonexistent"))
        self.assertEqual(self.trie.get("nonexistent", default=[]), [])

    def test_find_suffixes_ordering_longest_first(self):
        # "kitoblarimizda" has suffixes: "da", "imizda", "larimizda"
        matches = self.trie.find_suffixes("kitoblarimizda")
        matched_strings = [m[0] for m in matches]
        self.assertEqual(matched_strings, ["larimizda", "imizda", "da"])

        # Check lengths are strictly descending
        lengths = [len(m[0]) for m in matches]
        self.assertEqual(lengths, sorted(lengths, reverse=True))

        # Check associated data (each match is (suffix_str, payload_dict))
        self.assertEqual(matches[0][1]["type"], "compound")
        self.assertEqual(matches[2][1]["name"], "o'rin-payt")

    def test_find_suffixes_single_and_none(self):
        matches = self.trie.find_suffixes("maktabda")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0], "da")

        no_matches = self.trie.find_suffixes("kitob")
        self.assertEqual(no_matches, [])

    def test_find_suffixes_edge_cases(self):
        self.assertEqual(self.trie.find_suffixes(""), [])
        self.assertEqual(self.trie.find_suffixes(None), [])  # type: ignore[arg-type]

    def test_bulk_insert(self):
        st = SuffixTrie()
        added = st.bulk_insert([
            ("lar", {"cat": "pl"}),
            ("dan", {"cat": "abl"}),
            "dek",  # plain string without data dict
        ])
        self.assertEqual(added, 3)
        self.assertEqual(len(st), 3)
        self.assertTrue("dek" in st)
        # get() returns list; empty-dict payload = [{}]
        result = st.get("dek")
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], {})

    def test_iteration_and_items(self):
        st = SuffixTrie([("lar", {"id": 1}), ("da", {"id": 2})])
        all_suffixes = set(st)
        self.assertEqual(all_suffixes, {"lar", "da"})

        items = dict(st.items())
        # items() returns (suffix, list_of_dicts)
        self.assertEqual(items["lar"][0], {"id": 1})
        self.assertEqual(items["da"][0], {"id": 2})

    def test_type_error_handling(self):
        with self.assertRaises(TypeError):
            self.trie.insert(123, {})  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            self.trie.insert("lar", "invalid_data")  # type: ignore[arg-type]


class TestDictionaryTrie(unittest.TestCase):
    def setUp(self):
        self.words = [
            "kitob",
            "kitobxon",
            "kitobxona",
            "maktab",
            "maktabdosh",
            "daftar",
            "qalam",
            "qalamdon",
            "ona",
            "ona-vatan",
            "o'quvchi",
            "g'olib",
        ]
        self.trie = DictionaryTrie(self.words)

    def test_len(self):
        self.assertEqual(len(self.trie), len(self.words))
        self.trie.insert("kitob")  # duplicate insert
        self.assertEqual(len(self.trie), len(self.words))

    def test_search_and_contains(self):
        self.assertTrue("kitob" in self.trie)
        self.assertTrue(self.trie.search("maktab"))
        self.assertTrue("o'quvchi" in self.trie)
        self.assertTrue("g'olib" in self.trie)
        self.assertFalse("shahar" in self.trie)
        self.assertFalse(self.trie.search("kit"))

    def test_starts_with(self):
        self.assertTrue(self.trie.starts_with("kit"))
        self.assertTrue(self.trie.starts_with("kitob"))
        self.assertTrue(self.trie.starts_with("maktab"))
        self.assertTrue(self.trie.starts_with("o'quv"))
        self.assertTrue(self.trie.starts_with("g'"))
        self.assertFalse(self.trie.starts_with("xyz"))
        self.assertTrue(self.trie.starts_with(""))

        empty_trie = DictionaryTrie()
        self.assertFalse(empty_trie.starts_with(""))

    def test_get_words_with_prefix(self):
        # Prefix "kitob" should match "kitob", "kitobxon", "kitobxona" in alphabetical order
        matches = self.trie.get_words_with_prefix("kitob")
        self.assertEqual(matches, ["kitob", "kitobxon", "kitobxona"])

        # Test limit
        matches_limited = self.trie.get_words_with_prefix("kitob", limit=2)
        self.assertEqual(matches_limited, ["kitob", "kitobxon"])

        # Nonexistent prefix
        self.assertEqual(self.trie.get_words_with_prefix("z"), [])

        # Limit 0
        self.assertEqual(self.trie.get_words_with_prefix("kitob", limit=0), [])

    def test_bulk_insert(self):
        dt = DictionaryTrie()
        dt.bulk_insert(["olma", "anor", "anjir", "olxo'ri"])
        self.assertEqual(len(dt), 4)
        self.assertTrue("olma" in dt)
        self.assertTrue(dt.starts_with("an"))

    def test_find_prefixes_and_longest_prefix(self):
        # "kitobxonlik" has prefix "kitob" and "kitobxon" in trie
        prefixes = self.trie.find_prefixes("kitobxoni")
        self.assertEqual([p[0] for p in prefixes], ["kitobxon", "kitob"])
        self.assertEqual(self.trie.longest_prefix("kitobxoni"), "kitobxon")
        self.assertEqual(self.trie.longest_prefix("kitob"), "kitob")
        self.assertIsNone(self.trie.longest_prefix("nonexistent"))

    def test_iteration(self):
        words = sorted(list(self.trie))
        self.assertEqual(words, sorted(self.words))


class TestBatchOperations(unittest.TestCase):
    def setUp(self):
        self.dt = DictionaryTrie(["kitob", "maktab", "daftar", "qalam"])
        self.st = SuffixTrie([
            ("da", {"case": "loc"}),
            ("dan", {"case": "abl"}),
            ("lar", {"cat": "pl"}),
            ("larimizda", {"cat": "comp"}),
        ])

    def test_batch_lookup(self):
        queries = ["kitob", "nonexistent", "daftar", "qalamcha"]
        results = batch_lookup(self.dt, queries)
        self.assertIsInstance(results, np.ndarray)
        self.assertEqual(results.dtype, bool)
        np.testing.assert_array_equal(results, np.array([True, False, True, False]))

        # Method version on trie instance
        results_method = self.dt.batch_lookup(queries)
        np.testing.assert_array_equal(results_method, results)

    def test_batch_lookup_empty(self):
        results = batch_lookup(self.dt, [])
        self.assertEqual(len(results), 0)
        self.assertEqual(results.dtype, bool)

    def test_batch_find_suffixes(self):
        queries = ["kitobda", "maktab", "daftarimlarimizda"]
        results = batch_find_suffixes(self.st, queries)
        self.assertEqual(len(results), 3)

        # 1. "kitobda" -> [("da", {"case": "loc"})]
        self.assertEqual(len(results[0]), 1)
        self.assertEqual(results[0][0][0], "da")

        # 2. "maktab" -> []
        self.assertEqual(results[1], [])

        # 3. "daftarimlarimizda" -> matches "larimizda" and "da"
        self.assertEqual(len(results[2]), 2)
        self.assertEqual(results[2][0][0], "larimizda")
        self.assertEqual(results[2][1][0], "da")

        # Method version on suffix trie instance
        results_method = self.st.batch_find_suffixes(queries)
        self.assertEqual(results_method, results)


if __name__ == "__main__":
    unittest.main()
