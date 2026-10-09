"""
Test script to verify lossless tokenization and correct morphology.
"""

from uztoken.normalizer import normalize_apostrophes_lossless, restore_apostrophes_lossless
from uztoken.morphology import create_analyzer

text = "O'zbekistonning go'zal qishlog'imizdan o'qiyaptilar-ku!"
print(f"Original: {text}")

# 1. Lossless Normalization
norm_text, offsets = normalize_apostrophes_lossless(text)
print(f"Normalized: {norm_text}")

# 2. Reversal test
restored = restore_apostrophes_lossless(norm_text, offsets)
print(f"Restored: {restored}")
print(f"Lossless Match: {restored == text}")

# 3. Morphological analysis
analyzer = create_analyzer()
words = norm_text.split()
for word in words:
    # simple word split for demo, assuming no punctuation attached
    clean_word = word.strip("!,-")
    res = analyzer.analyze(clean_word)
    print(f"{clean_word} -> {res.root} + {[s.text for s in res.suffixes]}")

