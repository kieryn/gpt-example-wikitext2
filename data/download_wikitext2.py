"""
Script to download WikiText-2 and save raw text splits.
"""

from datasets import load_dataset
import os

def main():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    os.makedirs("data", exist_ok=True)
    for split in ["train", "validation", "test"]:
        texts = dataset[split]["text"]
        out_file = os.path.join("data", f"wikitext2_{split}.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            for line in texts:
                if line.strip():
                    f.write(line + "\n")
    print("Saved train, validation, and test text files in data/")

if __name__ == "__main__":
    main()
