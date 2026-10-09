from uztoken.tokenizer import create_tokenizer
from uztoken.morphology import create_analyzer

text = "O'zbekistonning go'zal qishlog'imizdan o'qiyaptilar-ku!"
print("MATN:", text)

tok = create_tokenizer(load_dict=True)
analyzer = tok._analyzer

# Just do a raw trace of what analyze() returns
for word in text.split():
    clean_word = word.strip("!,-")
    res = analyzer.analyze(clean_word)
    print(f"[{clean_word}] -> Root: {res.root}, Suffixes: {[s.text for s in res.suffixes]}")

