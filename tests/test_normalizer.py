"""Unit tests for uztoken.normalizer module."""

import unittest
import sys
import os

# Add parent directory to sys.path to enable direct importing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from uztoken.normalizer import (
    normalize_apostrophes,
    normalize_uzbek,
    cyrillic_to_latin,
    is_uzbek_word,
    clean_text,
    has_cyrillic,
    detect_script,
    APOSTROPHE_VARIANTS,
)


class TestNormalizeApostrophes(unittest.TestCase):
    """Tests for normalize_apostrophes function."""

    def test_empty_and_types(self):
        self.assertEqual(normalize_apostrophes(""), "")
        with self.assertRaises(TypeError):
            normalize_apostrophes(None)  # type: ignore
        with self.assertRaises(TypeError):
            normalize_apostrophes(123)  # type: ignore

    def test_all_specified_apostrophe_variants(self):
        # Specific variants required: U+02BB, U+02BC, U+2019, U+2018, U+0060, U+02B9, U+2032, U+02BF, and regular '
        variants = {
            "\u02BB": "MODIFIER LETTER TURNED COMMA (official Uzbek Oʻ, Gʻ)",
            "\u02BC": "MODIFIER LETTER APOSTROPHE (official Uzbek tutuq belgisi)",
            "\u2019": "RIGHT SINGLE QUOTATION MARK",
            "\u2018": "LEFT SINGLE QUOTATION MARK",
            "\u0060": "GRAVE ACCENT",
            "\u02B9": "MODIFIER LETTER PRIME",
            "\u2032": "PRIME",
            "\u02BF": "MODIFIER LETTER LEFT HALF RING",
            "\u0027": "ASCII APOSTROPHE",
            # Additional variants
            "\u00B4": "ACUTE ACCENT",
            "\u02BE": "MODIFIER LETTER RIGHT HALF RING",
            "\u201B": "SINGLE HIGH-REVERSED-9 QUOTATION MARK",
            "\u2035": "REVERSED PRIME",
            "\uFF07": "FULLWIDTH APOSTROPHE",
        }

        for char, desc in variants.items():
            input_text = f"o{char}zbek g{char}isht ta{char}lim"
            expected = "o'zbek g'isht ta'lim"
            self.assertEqual(
                normalize_apostrophes(input_text),
                expected,
                f"Failed for {desc} ({hex(ord(char))})"
            )

    def test_real_world_phrases(self):
        self.assertEqual(normalize_apostrophes("Oʻzbekiston Respublikasi"), "O'zbekiston Respublikasi")
        self.assertEqual(normalize_apostrophes("Gʻalaba bogʻi"), "G'alaba bog'i")
        self.assertEqual(normalize_apostrophes("ko`ngil va ta’lim"), "ko'ngil va ta'lim")
        self.assertEqual(normalize_apostrophes("maʼlumotlar ombori"), "ma'lumotlar ombori")
        self.assertEqual(normalize_apostrophes("surʿat va san′at"), "sur'at va san'at")


class TestNormalizeUzbek(unittest.TestCase):
    """Tests for normalize_uzbek pipeline."""

    def test_empty_and_types(self):
        self.assertEqual(normalize_uzbek(""), "")
        with self.assertRaises(TypeError):
            normalize_uzbek(None)  # type: ignore

    def test_full_pipeline(self):
        # 1. NFC, 2. apostrophes, 3. lower, 4. strip & collapse extra whitespace
        input_text = "  OʻZBEKISTON   Respublikasi   Taʼlim  Vazirligi  \n\t"
        expected = "o'zbekiston respublikasi ta'lim vazirligi"
        self.assertEqual(normalize_uzbek(input_text), expected)

    def test_orphaned_apostrophe_handling(self):
        # Poorly typed spaces between letter and apostrophe
        self.assertEqual(normalize_uzbek("o ' zbekiston"), "o'zbekiston")
        self.assertEqual(normalize_uzbek("g ' alaba"), "g'alaba")

    def test_unicode_nfc_composition(self):
        text = "ko\u02BBrinish"
        self.assertEqual(normalize_uzbek(text), "ko'rinish")


class TestCyrillicToLatin(unittest.TestCase):
    """Tests for cyrillic_to_latin transliteration."""

    def test_empty_and_types(self):
        self.assertEqual(cyrillic_to_latin(""), "")
        with self.assertRaises(TypeError):
            cyrillic_to_latin(None)  # type: ignore

    def test_all_mapped_alphabet_characters(self):
        # Mapping from prompt:
        # А->A, Б->B, В->V, Г->G, Д->D, Е->E (Ye at start of word), Ё->Yo, Ж->J, З->Z, И->I,
        # Й->Y, К->K, Л->L, М->M, Н->N, О->O, П->P, Р->R, С->S, Т->T, У->U, Ф->F, Х->X,
        # Ц->Ts, Ч->Ch, Ш->Sh, Щ->Sh, Ъ->' (apostrophe), Ы->I, Ь->(empty), Э->E, Ю->Yu,
        # Я->Ya, Ў->O', Қ->Q, Ғ->G', Ҳ->H
        cyr_chars = (
            "А Б В Г Д Ж З И Й К Л М Н О П Р С Т У Ф Х Ц Ч Ш Щ Ъ Ы Ь Э Ю Я Ў Қ Ғ Ҳ "
            "а б в г д ж з и й к л м н о п р с т у ф х ц ч ш щ ъ ы ь э ю я ў қ ғ ҳ"
        )
        expected_lat = (
            "A B V G D J Z I Y K L M N O P R S T U F X Ts Ch Sh Sh ' I  E Yu Ya O' Q G' H "
            "a b v g d j z i y k l m n o p r s t u f x ts ch sh sh ' i  e yu ya o' q g' h"
        )
        self.assertEqual(cyrillic_to_latin(cyr_chars), expected_lat)

    def test_contextual_e_rule(self):
        # At start of word: Ye / ye
        self.assertEqual(cyrillic_to_latin("Ер"), "Yer")
        self.assertEqual(cyrillic_to_latin("ер"), "yer")
        self.assertEqual(cyrillic_to_latin("ЕР"), "YER")
        self.assertEqual(cyrillic_to_latin("Европа"), "Yevropa")
        self.assertEqual(cyrillic_to_latin("елка"), "yelka")
        self.assertEqual(cyrillic_to_latin("Бу ер"), "Bu yer")

        # After consonant: E / e
        self.assertEqual(cyrillic_to_latin("керак"), "kerak")
        self.assertEqual(cyrillic_to_latin("КЕРАК"), "KERAK")
        self.assertEqual(cyrillic_to_latin("Берлин"), "Berlin")
        self.assertEqual(cyrillic_to_latin("девор"), "devor")
        self.assertEqual(cyrillic_to_latin("мен"), "men")

        # After vowel: Ye / ye
        self.assertEqual(cyrillic_to_latin("поезд"), "poyezd")
        self.assertEqual(cyrillic_to_latin("ПОЕЗД"), "POYEZD")
        self.assertEqual(cyrillic_to_latin("бояемиш"), "boyayemish")

        # After hard sign ъ: 'ye
        self.assertEqual(cyrillic_to_latin("съезд"), "s'yezd")
        self.assertEqual(cyrillic_to_latin("разъезд"), "raz'yezd")

        # After soft sign ь: ye (with ь dropped)
        self.assertEqual(cyrillic_to_latin("пьеса"), "pyesa")
        self.assertEqual(cyrillic_to_latin("премьера"), "premyera")

    def test_special_uzbek_letters(self):
        self.assertEqual(cyrillic_to_latin("Ўзбекистон"), "O'zbekiston")
        self.assertEqual(cyrillic_to_latin("ўзбек"), "o'zbek")
        self.assertEqual(cyrillic_to_latin("ЎЗБЕКИСТОН"), "O'ZBEKISTON")
        self.assertEqual(cyrillic_to_latin("Қорақалпоғистон"), "Qoraqalpog'iston")
        self.assertEqual(cyrillic_to_latin("Ғалаба"), "G'alaba")
        self.assertEqual(cyrillic_to_latin("ҒАЛАБА"), "G'ALABA")
        self.assertEqual(cyrillic_to_latin("ҳурматли"), "hurmatli")

    def test_digraph_case_preservation(self):
        self.assertEqual(cyrillic_to_latin("Шаҳар"), "Shahar")
        self.assertEqual(cyrillic_to_latin("ШАҲАР"), "SHAHAR")
        self.assertEqual(cyrillic_to_latin("Чойхона"), "Choyxona")
        self.assertEqual(cyrillic_to_latin("ЧОЙ"), "CHOY")
        self.assertEqual(cyrillic_to_latin("КУЧ"), "KUCH")
        self.assertEqual(cyrillic_to_latin("Куч"), "Kuch")
        self.assertEqual(cyrillic_to_latin("Цех"), "Tsex")
        self.assertEqual(cyrillic_to_latin("ЦЕХ"), "TSEX")
        self.assertEqual(cyrillic_to_latin("Яхши"), "Yaxshi")
        self.assertEqual(cyrillic_to_latin("ЯХШИ"), "YAXSHI")
        self.assertEqual(cyrillic_to_latin("Юлдуз"), "Yulduz")
        self.assertEqual(cyrillic_to_latin("ЮЛДУЗ"), "YULDUZ")
        self.assertEqual(cyrillic_to_latin("Ёшлик"), "Yoshlik")
        self.assertEqual(cyrillic_to_latin("ЁШЛИК"), "YOSHLIK")

    def test_tutuq_and_soft_sign(self):
        self.assertEqual(cyrillic_to_latin("маъно"), "ma'no")
        self.assertEqual(cyrillic_to_latin("эълон"), "e'lon")
        self.assertEqual(cyrillic_to_latin("фильм"), "film")
        self.assertEqual(cyrillic_to_latin("лагерь"), "lager")
        self.assertEqual(cyrillic_to_latin("компьютер"), "kompyuter")


class TestIsUzbekWord(unittest.TestCase):
    """Tests for is_uzbek_word validator."""

    def test_valid_uzbek_words(self):
        valid = [
            "kitob", "maktab", "dasturlash", "til", "inson",
            "o'quvchi", "o‘quvchi", "O'zbekiston", "o'g'il",
            "g'alaba", "g'isht", "g'oya",
            "shahar", "shoshilmoq", "tosh",
            "choy", "chala", "achchiq", "kecha",
            "a'lo", "ma'no", "in'om", "fe'l", "sur'at",
            "ko'z", "to'g'ri", "so'z",
            "o'", "g'", "ch", "sh", "ng",
            "CHOY", "SHAHAR", "G'AFUR", "O'ZBEK",
        ]
        for w in valid:
            self.assertTrue(is_uzbek_word(w), f"Failed for valid word {w!r}")

    def test_invalid_uzbek_words(self):
        invalid = [
            "",               # empty
            "   ",            # whitespace
            "'",              # apostrophe alone
            "''",             # double apostrophe
            "123",            # digits
            "kitob1",         # word + digits
            "китоб",          # Cyrillic
            "web",            # 'w' not in Uzbek alphabet
            "world",          # 'w' not in Uzbek alphabet
            "cat",            # 'c' without 'h'
            "cofe",           # 'c' without 'h'
            "action",         # 'c' without 'h'
            "'salom",         # leading apostrophe
            "kitob'",         # trailing apostrophe not after o/g
            "san'at'",        # trailing apostrophe not after o/g
            "salom''dunyo",   # double apostrophe
            "salom dunyo",    # multiple words with space
            "salom!",         # punctuation
            "kitob-daftar",   # hyphenated
        ]
        for w in invalid:
            self.assertFalse(is_uzbek_word(w), f"Expected False for invalid word {w!r}")

    def test_general_mode(self):
        # With strict=False, loanwords with 'w' or 'c' are accepted as Latin words
        self.assertTrue(is_uzbek_word("web", strict=False))
        self.assertTrue(is_uzbek_word("cat", strict=False))
        # But digits, Cyrillic, invalid apostrophes are still rejected
        self.assertFalse(is_uzbek_word("123", strict=False))
        self.assertFalse(is_uzbek_word("китоб", strict=False))
        self.assertFalse(is_uzbek_word("'salom", strict=False))


class TestCleanText(unittest.TestCase):
    """Tests for clean_text cleaning pipeline."""

    def test_empty_and_types(self):
        self.assertEqual(clean_text(""), "")
        with self.assertRaises(TypeError):
            clean_text(None)  # type: ignore

    def test_html_tag_and_entity_removal(self):
        text = "<p>Salom <b>dunyo</b>! &amp; hush kelibsiz!<br/></p>"
        expected = "Salom dunyo! & hush kelibsiz!"
        self.assertEqual(clean_text(text), expected)

    def test_url_removal(self):
        text = "Batafsil https://kun.uz/news/123 yoki www.gov.uz orqali o'qing."
        expected = "Batafsil yoki orqali o'qing."
        self.assertEqual(clean_text(text), expected)

    def test_email_removal(self):
        text = "Murojaat uchun: support@uztoken.uz yoki admin-help@corp.org yozing."
        expected = "Murojaat uchun: yoki yozing."
        self.assertEqual(clean_text(text), expected)

    def test_excessive_punctuation(self):
        text = "Zo'r yangilik!!!!! Rostdanmi???? Voy-dod..... Qoyil ***"
        expected = "Zo'r yangilik! Rostdanmi? Voy-dod... Qoyil"
        self.assertEqual(clean_text(text), expected)

    def test_mixed_punctuation(self):
        text = "Bu qanday bo'ldi!?!?!? Tushunmadim..."
        expected = "Bu qanday bo'ldi?! Tushunmadim..."
        self.assertEqual(clean_text(text), expected)

    def test_uzbek_apostrophe_preservation(self):
        # Uzbek words with apostrophes MUST NOT be destroyed
        text = "<h1>O'zbekiston</h1> <a href='https://daryo.uz'>ko'ring</a>: g'alaba va e'lon!"
        expected = "O'zbekiston ko'ring: g'alaba va e'lon!"
        self.assertEqual(clean_text(text), expected)


class TestAuxiliaryFunctions(unittest.TestCase):
    """Tests for auxiliary helper functions."""

    def test_has_cyrillic(self):
        self.assertTrue(has_cyrillic("Ўзбекистон"))
        self.assertFalse(has_cyrillic("O'zbekiston"))
        self.assertTrue(has_cyrillic("iPhone 15 ни сотиб олдим"))
        self.assertFalse(has_cyrillic(""))

    def test_detect_script(self):
        self.assertEqual(detect_script("Ўзбекистон Республикаси"), "cyrillic")
        self.assertEqual(detect_script("O'zbekiston Республикаси"), "mixed")
        self.assertEqual(detect_script("O'zbekiston Respublikasi"), "latin")
        self.assertEqual(detect_script("123 456"), "other")
        self.assertEqual(detect_script(""), "empty")


if __name__ == "__main__":
    unittest.main()
