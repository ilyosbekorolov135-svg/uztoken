"""
Integration test for UzToken library.
Tests the full pipeline: dictionary loading -> morphological analysis -> output.
"""
import sys
import os
import time

# Add parent to path for testing without install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uztoken import UzTokenizer, Dictionary, normalize_uzbek


def test_full_pipeline():
    print("=" * 70)
    print("  UZTOKEN INTEGRATSIYA TESTI")
    print("=" * 70)

    # 1. Create tokenizer with built-in dictionary only (no download)
    print("\n[1] Tokenizer yaratilmoqda (ichki lug'at)...")
    start = time.time()
    tok = UzTokenizer(max_depth=15)
    elapsed = time.time() - start
    print(f"    Tayyor: {elapsed:.3f}s, Lug'at: {tok.dictionary_size} ta o'zak")

    # 2. Test basic morphological analysis
    print("\n[2] Morfologik tahlil testlari:")
    print("-" * 70)

    test_words = [
        ("maktablarimizda", "maktab", True),
        ("ishlaydigan", "ishla", True),
        ("kitoblarni", "kitob", True),
        ("o'qituvchilar", "o'qit", True),
        ("yozilmagan", "yoz", True),
        ("kodda", "kod", True),
        ("funksiyasi", "funksiya", True),
        ("sizga", "siz", True),
        ("faylda", "fayl", True),
        ("qishlog'i", "qishloq", True),
        ("yuragim", "yurak", True),
        ("shahri", "shahar", True),
        ("singlim", "singil", True),
        ("burni", "burun", True),
    ]

    passed = 0
    failed = 0

    for word, expected_root, expect_found in test_words:
        result = tok.analyze(word)
        root_ok = result.root == expected_root if expect_found else True
        found_ok = result.is_found == expect_found

        if root_ok and found_ok:
            status = "OK"
            passed += 1
        else:
            status = "FAIL"
            failed += 1

        chain = result.suffix_chain if result.suffix_chain else "(qo'shimchasiz)"
        print(f"    [{status}] '{word}' -> o'zak: '{result.root}' {chain}")
        if not root_ok:
            print(f"         Kutilgan o'zak: '{expected_root}', Topilgan: '{result.root}'")

    print(f"\n    Natija: {passed}/{passed+failed} ta o'tdi")

    # 3. Test full tokenization
    print("\n[3] To'liq tokenizatsiya testi:")
    print("-" * 70)

    test_text = "Maktablarimizda yangi kitoblar bor. Biz har kuni o'qiymiz."
    tokens = tok.tokenize(test_text)

    for t in tokens:
        if t.token_type == "word":
            if t.is_morph and t.morph and t.morph.is_found:
                chain = t.morph.suffix_chain
                if chain:
                    print(f"    [MORPH] '{t.text}' -> {t.morph.root} {chain}")
                else:
                    print(f"    [MORPH] '{t.text}' -> {t.morph.root} (bare)")
            else:
                print(f"    [????]  '{t.text}' -> noma'lum")
        elif t.token_type == "punctuation":
            print(f"    [PUNCT] '{t.text}'")

    # 4. Test string tokenization output
    print("\n[4] String tokenizatsiya:")
    print("-" * 70)
    str_tokens = tok.tokenize_to_strings("maktablarimizda o'qiymiz")
    print(f"    Kirish:  'maktablarimizda o'qiymiz'")
    print(f"    Chiqish: {str_tokens}")

    # 5. Test stats
    print("\n[5] Statistika:")
    print("-" * 70)
    stats = tok.stats
    for k, v in stats.items():
        print(f"    {k}: {v}")

    # 6. Performance benchmark
    print("\n[6] Tezlik benchmarki:")
    print("-" * 70)

    bench_text = "Bugun biz maktabda yangi darslarni o'rganamiz va kitoblar o'qiymiz"
    iterations = 1000

    start = time.time()
    for _ in range(iterations):
        tok.tokenize(bench_text)
    elapsed = time.time() - start

    words_per_sec = (iterations * len(bench_text.split())) / elapsed
    print(f"    {iterations} marta tokenizatsiya: {elapsed:.3f}s")
    print(f"    Tezlik: ~{words_per_sec:.0f} so'z/soniya")

    # 7. Murakkab so'zlar
    print("\n[7] Murakkab so'zlar testi:")
    print("-" * 70)

    complex_words = [
        "internatsionalizatsiyalashtirolmayotganliklaringizdandir",
        "mexanizatsiyalashtirilmaganliklaringizdandirda",
        "muvaffaqiyatsizlik",
        "kompyuterizatsiyalashtirish",
    ]

    for word in complex_words:
        result = tok.analyze(word)
        if result.is_found:
            print(f"    [OK]   '{word}'")
            print(f"           O'zak: {result.root}")
            print(f"           Zanjir: {result.suffix_chain}")
            print(f"           Ishonch: {result.confidence:.2f}")
        else:
            print(f"    [FAIL] '{word}' -> topilmadi")

    print("\n" + "=" * 70)
    print("  TEST YAKUNLANDI")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
