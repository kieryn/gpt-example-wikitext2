"""
Training loop for the Transformer language model.
"""

import os
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints
from flax import linen as nn
from datasets import load_dataset
import yaml
import numpy as np

from model import TransformerDecoder
from tokenizer import load_tokenizer

def create_train_state(rng, config):
    model = TransformerDecoder(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_length=config["max_seq_length"],
        dropout_rate=config.get("dropout_rate", 0.1)
    )
    params = model.init(rng, jnp.ones((1, config["max_seq_length"]), dtype=jnp.int32))["params"]
    tx = optax.adamw(config["learning_rate"])
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def prepare_datasets(tokenizer, config):
    """Load raw text splits, tokenize, and group into fixed-length sequences."""
    data_files = {
        "train": os.path.join("data", "wikitext2_train.txt"),
        "validation": os.path.join("data", "wikitext2_validation.txt"),
        "test": os.path.join("data", "wikitext2_test.txt"),
    }
    raw_datasets = load_dataset("text", data_files=data_files)
    def tokenize_fn(examples):
        encodings = tokenizer.encode_batch(examples["text"])
        return {"input_ids": [enc.ids for enc in encodings]}
    tokenized = raw_datasets.map(tokenize_fn, batched=True, remove_columns=["text"])
    block_size = config["max_seq_length"]
    def group_fn(examples):
        all_ids = sum(examples["input_ids"], [])
        total_len = (len(all_ids) // block_size) * block_size
        all_ids = all_ids[:total_len]
        chunks = [all_ids[i:i+block_size] for i in range(0, total_len, block_size)]
        return {"input_ids": chunks, "labels": chunks.copy()}
    lm_datasets = tokenized.map(group_fn, batched=True)
    lm_datasets.set_format(type="numpy", columns=["input_ids", "labels"])
    return lm_datasets

def data_loader(dataset, batch_size, shuffle=True):
    """Yield batches of data as dicts of JAX arrays."""
    input_ids = dataset["input_ids"]
    labels = dataset["labels"]
    data_size = len(input_ids)
    indices = np.arange(data_size)
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, data_size, batch_size):
            batch_idx = indices[start:start+batch_size]
            batch_inputs = np.stack([input_ids[i] for i in batch_idx])
            batch_labels = np.stack([labels[i] for i in batch_idx])
            yield {
                "input_ids": jnp.array(batch_inputs, dtype=jnp.int32),
                "labels": jnp.array(batch_labels, dtype=jnp.int32),
            }

def train(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    tokenizer = load_tokenizer(config.get("tokenizer_path", "tokenizer"))

    # Prepare data pipeline: load raw text, tokenize, and batch into fixed-length sequences
    lm_datasets = prepare_datasets(tokenizer, config)
    train_ds = lm_datasets["train"]
    val_ds = lm_datasets.get("validation") or lm_datasets.get("valid")

    train_loader = data_loader(train_ds, config["batch_size"], shuffle=True)
    val_loader = data_loader(val_ds, config["batch_size"], shuffle=False)

    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, config)

    # Quick check: fetch one batch
    batch = next(train_loader)
    print(f"Loaded a batch of input_ids shape {batch['input_ids'].shape}")
    # TODO: implement the full training loop using state, batches, and checkpoints

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
