import os
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Punctuation, Sequence
from tokenizers.normalizers import Lowercase

def train_bpe(corpus_path: str, vocab_size: int, output_path: str):
    print(f"Training BPE model with vocab_size={vocab_size} on {corpus_path}...")
    
    # Initialize a Byte-Level BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # Simple pre-tokenization
    tokenizer.pre_tokenizer = Sequence([Whitespace(), Punctuation()])
    tokenizer.normalizer = Lowercase()
    
    trainer = BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress=True
    )
    
    tokenizer.train(files=[corpus_path], trainer=trainer)
    tokenizer.save(output_path)
    print(f"BPE model successfully saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BPE model for UzToken")
    parser.add_argument("--corpus", type=str, required=True, help="Path to the training text file")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--output", type=str, default="uztoken/data/uz_bpe.json", help="Output JSON path")
    
    args = parser.parse_args()
    train_bpe(args.corpus, args.vocab_size, args.output)
