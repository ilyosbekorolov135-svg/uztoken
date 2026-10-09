from uztoken.tokenizer import create_tokenizer

text = "Maktablarimizda yangi darsliklar keldi! Pythonda ishlaydigan dastur."

tok = create_tokenizer(load_dict=True)

# Hybrid BPE tokens (Morphology first, then BPE for OOV)
tokens = tok.tokenize_to_strings(text, style="hybrid")

for t in tok.tokenize(text):
    print(t)

print("\n--- Yakuniy Subwords ---")
print(tokens)
