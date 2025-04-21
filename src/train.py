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

def train(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    tokenizer = load_tokenizer(config.get("tokenizer_path", "tokenizer"))

    # Load dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # TODO: tokenize, batch, and prefetch data

    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, config)

    # TODO: implement training loop
    print("Training stub complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
