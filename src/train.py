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
import time

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

@jax.jit
def train_step(state, batch, dropout_rng):
    """Performs a single training step and returns the updated state and loss."""
    def loss_fn(params):
        inputs = batch['input_ids'][:, :-1]
        targets = batch['labels'][:, 1:]
        logits = state.apply_fn({'params': params}, inputs, train=True, rngs={'dropout': dropout_rng})
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
        return loss
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

@jax.jit
def eval_step(state, batch):
    """Computes loss on a batch without updating parameters."""
    inputs = batch['input_ids'][:, :-1]
    targets = batch['labels'][:, 1:]
    logits = state.apply_fn({'params': state.params}, inputs, train=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    return loss

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
    # Optional smoke-test settings
    max_steps = config.get("max_steps_per_epoch", None)
    skip_validation = config.get("skip_validation", False)

    # Prepare data pipeline: load raw text, tokenize, and batch into fixed-length sequences
    lm_datasets = prepare_datasets(tokenizer, config)
    train_ds = lm_datasets["train"]
    val_ds = lm_datasets.get("validation") or lm_datasets.get("valid")

    train_loader = data_loader(train_ds, config["batch_size"], shuffle=True)
    val_loader = data_loader(val_ds, config["batch_size"], shuffle=False)

    # Initialize model state
    rng = jax.random.PRNGKey(config.get("seed", 0))
    state = create_train_state(rng, config)

    # Training loop
    dropout_rng = jax.random.PRNGKey(config.get("seed", 1))
    batch_size = config["batch_size"]
    total_steps = len(train_ds["input_ids"]) // batch_size
    num_epochs = config.get("num_epochs", 1)
    log_every = config.get("log_every", 100)
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()
        step_count = 0
        for step in range(total_steps):
            if max_steps is not None and step_count >= max_steps:
                break
            batch = next(train_loader)
            dropout_rng, subkey = jax.random.split(dropout_rng)
            state, loss = train_step(state, batch, subkey)
            epoch_loss += loss
            step_count += 1
            if step_count % log_every == 0 or (max_steps is not None and step_count == max_steps):
                print(f"Epoch {epoch} step {step_count}/{total_steps} loss {loss:.4f}")
        avg_loss = (epoch_loss / step_count) if step_count > 0 else float('nan')
        t1 = time.time()
        print(f"Epoch {epoch} completed in {t1 - t0:.2f}s, steps {step_count}, avg loss {avg_loss:.4f}")

        # Validation
        if (not skip_validation) and val_ds is not None:
            val_loss = 0.0
            val_steps = len(val_ds["input_ids"]) // batch_size
            val_count = 0
            for _ in range(val_steps):
                batch = next(val_loader)
                val_loss += eval_step(state, batch)
                val_count += 1
            avg_val_loss = (val_loss / val_count) if val_count > 0 else float('nan')
            print(f"Validation loss: {avg_val_loss:.4f}")

        # Save checkpoint (use absolute path)
        ckpt_dir = os.path.abspath(config.get("checkpoint_dir", "logs"))
        os.makedirs(ckpt_dir, exist_ok=True)
        checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=state.params, step=epoch, overwrite=True)
    print("Training complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)
