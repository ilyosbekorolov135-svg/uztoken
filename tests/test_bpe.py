"""Unit tests for uztoken.bpe (Byte Pair Encoding tokenizer)."""

import os
import shutil
import tempfile
import unittest

from uztoken.bpe import (
    BPETokenizer,
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    DEFAULT_SPECIAL_TOKENS,
    HAS_NUMPY,
)


class TestBPETokenizer(unittest.TestCase):
    """Test suite for BPETokenizer."""

    def setUp(self):
        """Set up temporary directory and sample data for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_corpus = [
            "Salom! O'zbekiston - go'zal va mehmondo'st yurt!",
            "Maktablarimizda o'quvchilar va o'qituvchilar ko'p.",
            "Bog'da 100% yangi ko'chatlar va daraxtlar o'smoqda.",
            "Kitoblarimiz va daftarlarimiz maktabda qoldi.",
            "To'g'ri so'z - a'lo fazilat, ma'no va san'at.",
            "Mexanizatsiyalashtirish bo'yicha ishlar boshlandi.",
            "O'zbekiston tarixi juda qadimiy va boydir.",
            "2026-yilda yangi marralar zabt etiladi.",
        ]

    def tearDown(self):
        """Clean up temporary test directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_special_tokens_initialization(self):
        """Verify standard special tokens (<pad>, <unk>, <bos>, <eos>) and IDs."""
        bpe = BPETokenizer(vocab_size=100)
        self.assertEqual(bpe.vocab[PAD_TOKEN], 0)
        self.assertEqual(bpe.vocab[UNK_TOKEN], 1)
        self.assertEqual(bpe.vocab[BOS_TOKEN], 2)
        self.assertEqual(bpe.vocab[EOS_TOKEN], 3)

        self.assertEqual(bpe.pad_token_id, 0)
        self.assertEqual(bpe.unk_token_id, 1)
        self.assertEqual(bpe.bos_token_id, 2)
        self.assertEqual(bpe.eos_token_id, 3)

        self.assertEqual(bpe.inverse_vocab[0], PAD_TOKEN)
        self.assertEqual(bpe.inverse_vocab[1], UNK_TOKEN)
        self.assertEqual(bpe.inverse_vocab[2], BOS_TOKEN)
        self.assertEqual(bpe.inverse_vocab[3], EOS_TOKEN)

    def test_pre_tokenization_uzbek_orthography(self):
        """Verify pre-tokenization preserves o', g', tutuq belgisi, and splits punctuation."""
        bpe = BPETokenizer()

        # 1. Digraph letters: o', g'
        tokens = bpe.pre_tokenize("O'zbekiston go'zal bog' to'g'ri")
        self.assertEqual([t.replace("ʻ", "'") for t in tokens], ["O'zbekiston", "go'zal", "bog'", "to'g'ri"])

        # 2. Tutuq belgisi (internal apostrophe)
        tokens = bpe.pre_tokenize("Ma'no va san'at, a'lo ta'lim!")
        self.assertEqual([t.replace("ʻ", "'").replace("ʼ", "'") for t in tokens], ["Ma'no", "va", "san'at", ",", "a'lo", "ta'lim", "!"])

        # 3. Normalization of apostrophe variants (ʻ, ʼ, ’, etc.)
        tokens_variant = bpe.pre_tokenize("Oʻzbekiston g‘o‘za maʼno")
        self.assertEqual([t.replace("ʻ", "'").replace("ʼ", "'") for t in tokens_variant], ["O'zbekiston", "g'o'za", "ma'no"])

        # 4. Quotation marks separated from words
        tokens_quotes = bpe.pre_tokenize("'maktab' va 'bog''")
        self.assertEqual([t.replace("ʻ", "'") for t in tokens_quotes], ["'", "maktab", "'", "va", "'", "bog'", "'"])

        # 5. Numbers and punctuation
        tokens_num = bpe.pre_tokenize("100% kelishdi, 2026-yil.")
        self.assertEqual(tokens_num, ["100", "%", "kelishdi", ",", "2026", "-", "yil", "."])

    def test_training_and_vocabulary_expansion(self):
        """Verify BPE training expands vocabulary and creates merge rules."""
        bpe = BPETokenizer(vocab_size=150)
        bpe.train(self.sample_corpus, min_frequency=2)

        self.assertGreater(len(bpe.merges), 0)
        self.assertGreater(len(bpe.vocab), len(DEFAULT_SPECIAL_TOKENS))
        self.assertLessEqual(len(bpe.vocab), 150)

        # Check inverse vocab integrity
        for tok, idx in bpe.vocab.items():
            self.assertEqual(bpe.inverse_vocab[idx], tok)

    def test_min_frequency_threshold(self):
        """Verify pairs with frequency below min_frequency are not merged."""
        corpus = ["kamkamkam noyobsoz12345"]
        bpe = BPETokenizer(vocab_size=100)
        bpe.train(corpus, min_frequency=5)

        # 'noyobsoz12345' appears only once, pairs should not be merged
        for pair in bpe.merges:
            merged = pair[0] + pair[1]
            self.assertNotIn("12345", merged)

    def test_tokenization_subwords(self):
        """Verify subword tokenization splits unseen or known words cleanly."""
        bpe = BPETokenizer(vocab_size=80)
        bpe.train(self.sample_corpus, min_frequency=2)

        tokens = bpe.tokenize("maktablarimizda")
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        self.assertEqual("".join(tokens), "maktablarimizda")

        # Empty string
        self.assertEqual(bpe.tokenize(""), [])

    def test_encode_and_decode_roundtrip(self):
        """Verify lossless roundtrip encoding and decoding."""
        bpe = BPETokenizer(vocab_size=150)
        bpe.train(self.sample_corpus, min_frequency=2)

        test_sentences = [
            "O'zbekiston - go'zal yurt!",
            "Maktablarimizda o'quvchilar ko'p.",
            "Bog'da yangi ko'chatlar o'smoqda.",
            "To'g'ri so'z - a'lo fazilat.",
            "100% to'g'ri.",
        ]

        for s in test_sentences:
            ids = bpe.encode(s)
            self.assertIsInstance(ids, list)
            decoded = bpe.decode(ids)
            self.assertEqual(decoded.replace("ʻ", "'").replace("ʼ", "'"), s.replace("ʻ", "'").replace("ʼ", "'"))

    def test_encode_with_special_tokens(self):
        """Verify encode with add_bos and add_eos flags."""
        bpe = BPETokenizer(vocab_size=100)
        bpe.train(self.sample_corpus, min_frequency=2)

        ids = bpe.encode("Salom", add_bos=True, add_eos=True)
        self.assertEqual(ids[0], bpe.bos_token_id)
        self.assertEqual(ids[-1], bpe.eos_token_id)

        # Decoding skipping special tokens
        decoded = bpe.decode(ids, skip_special_tokens=True)
        self.assertEqual(decoded, "Salom")

        # Decoding preserving special tokens
        decoded_raw = bpe.decode(ids, skip_special_tokens=False)
        self.assertTrue(decoded_raw.startswith(BOS_TOKEN))
        self.assertTrue(decoded_raw.endswith(EOS_TOKEN))

    def test_encode_batch(self):
        """Verify batch encoding with padding and truncation."""
        bpe = BPETokenizer(vocab_size=100)
        bpe.train(self.sample_corpus, min_frequency=2)

        texts = ["Kitob", "O'zbekiston go'zal yurt"]
        batch = bpe.encode_batch(texts, pad_to_max=True, return_numpy=False)

        self.assertEqual(len(batch), 2)
        self.assertEqual(len(batch[0]), len(batch[1]))
        # First sequence is shorter, should have padding at the end
        self.assertEqual(batch[0][-1], bpe.pad_token_id)

        if HAS_NUMPY:
            np_batch = bpe.encode_batch(texts, pad_to_max=True, return_numpy=True)
            import numpy as np
            self.assertIsInstance(np_batch, np.ndarray)
            self.assertEqual(np_batch.shape[0], 2)
            self.assertEqual(np_batch.shape[1], len(batch[0]))

    def test_save_and_load(self):
        """Verify model serialization to JSON and reloading."""
        bpe = BPETokenizer(vocab_size=100)
        bpe.train(self.sample_corpus, min_frequency=2)

        model_path = os.path.join(self.temp_dir, "bpe_model.json")
        bpe.save(model_path)
        self.assertTrue(os.path.exists(model_path))

        # Load into a new instance
        loaded_bpe = BPETokenizer.from_file(model_path)
        self.assertEqual(len(loaded_bpe.vocab), len(bpe.vocab))
        self.assertEqual(len(loaded_bpe.merges), len(bpe.merges))
        self.assertEqual(loaded_bpe.merges, bpe.merges)

        # Tokenization & encoding consistency
        sample_text = "O'zbekiston - go'zal yurt!"
        self.assertEqual(bpe.tokenize(sample_text), loaded_bpe.tokenize(sample_text))
        self.assertEqual(bpe.encode(sample_text), loaded_bpe.encode(sample_text))
        self.assertEqual(
            loaded_bpe.decode(loaded_bpe.encode(sample_text)).replace("ʻ", "'"), sample_text.replace("ʻ", "'")
        )

    def test_magic_methods(self):
        """Verify __len__, __contains__, __getitem__, and __repr__."""
        bpe = BPETokenizer(vocab_size=100)
        self.assertEqual(len(bpe), 4)
        self.assertIn(PAD_TOKEN, bpe)
        self.assertNotIn("noma'lum_token", bpe)

        self.assertEqual(bpe[PAD_TOKEN], 0)
        self.assertEqual(bpe[0], PAD_TOKEN)

        with self.assertRaises(KeyError):
            _ = bpe["missing_token_xyz"]

        rep = repr(bpe)
        self.assertIn("BPETokenizer", rep)

    def test_edge_cases_and_error_handling(self):
        """Verify edge cases such as empty input, unknown characters, and errors."""
        bpe = BPETokenizer(vocab_size=100)

        # Empty texts for train
        with self.assertRaises(ValueError):
            bpe.train([])

        # Invalid min_frequency
        with self.assertRaises(ValueError):
            bpe.train(["salom"], min_frequency=0)

        # Load non-existent file
        with self.assertRaises(FileNotFoundError):
            bpe.load("non_existent_path.json")

        # Unknown character encoding
        bpe.train(["salom"], min_frequency=1)
        unknown_char = "§"
        encoded = bpe.encode(unknown_char)
        self.assertEqual(encoded, [bpe.unk_token_id])


if __name__ == "__main__":
    unittest.main()
