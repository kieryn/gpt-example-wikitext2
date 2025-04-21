"""
Tokenizer training and loading using HuggingFace Tokenizers.
"""

import os
from tokenizers import ByteLevelBPETokenizer

def train_tokenizer(data_path="data", vocab_size=32000, min_frequency=2, special_tokens=None):
    if special_tokens is None:
        special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
    os.makedirs("tokenizer", exist_ok=True)
    tokenizer = ByteLevelBPETokenizer()
    files = []
    for split in ("train", "validation", "test"):
        file_path = os.path.join(data_path, f"wikitext2_{split}.txt")
        files.append(file_path)
    tokenizer.train(files=files, vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens)
    tokenizer.save_model("tokenizer")
    return tokenizer

def load_tokenizer(tokenizer_path="tokenizer"):
    # Initialize tokenizer from trained vocabulary and merges
    tokenizer = ByteLevelBPETokenizer(
        os.path.join(tokenizer_path, "vocab.json"),
        os.path.join(tokenizer_path, "merges.txt")
    )
    return tokenizer
