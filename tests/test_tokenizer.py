import pytest
from uztoken.tokenizer import create_tokenizer
from uztoken.normalizer import normalize_apostrophes_lossless, restore_apostrophes_lossless

@pytest.fixture(scope="module")
def tokenizer():
    return create_tokenizer(load_dict=True)

def test_apostrophe_lossless():
    original = "O'zbekistonning go'zal qishlog'i va sanʼat"
    norm, offsets = normalize_apostrophes_lossless(original)
    
    assert "ʻ" in norm  # O' -> Oʻ, go' -> goʻ
    assert "ʼ" in norm  # san'at -> sanʼat
    
    restored = restore_apostrophes_lossless(norm, offsets)
    assert restored == original

def test_morphology_basic(tokenizer):
    tokens = list(tokenizer.tokenize("maktablarimizda"))
    assert len(tokens) == 1
    t = tokens[0]
    
    assert t.is_morph is True
    assert t.morph is not None
    assert t.morph.root == "maktab"
    assert len(t.morph.suffixes) == 3
    assert [s.text for s in t.morph.suffixes] == ["lar", "imiz", "da"]

def test_morphophonology(tokenizer):
    tokens = list(tokenizer.tokenize("qishlog'imizdan"))
    t = tokens[0]
    
    assert t.morph.root == "qishloq"
    assert [s.text for s in t.morph.suffixes] == ["imiz", "dan"]

def test_verb_morphology(tokenizer):
    tokens = list(tokenizer.tokenize("ishlaydigan"))
    t = tokens[0]
    
    assert t.morph.root == "ishla"
    assert [s.text for s in t.morph.suffixes] == ["ydigan"]

def test_oov_fallback(tokenizer):
    tokens = list(tokenizer.tokenize("Pythonda"))
    t = tokens[0]
    
    # Python is not in dictionary, but 'da' is a valid suffix
    assert t.is_morph is True
    assert t.morph.root == "python"
    assert [s.text for s in t.morph.suffixes] == ["da"]

def test_capitalization_preservation(tokenizer):
    # Ensure that hybrid tokenization keeps the case of the root
    subwords = tokenizer.tokenize_to_strings("Maktablarimizda", style="hybrid")
    
    # Depending on BPE logic, the root might be 'Maktab' or 'ĠMaktab'
    assert any("Maktab" in sw for sw in subwords)
    
    subwords_lower = tokenizer.tokenize_to_strings("maktablarimizda", style="hybrid")
    assert any("maktab" in sw for sw in subwords_lower)
