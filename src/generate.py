"""
Interactive text generation with the trained Transformer model.
"""
import os
import sys
import argparse
# Early parse to set CPU/GPU before JAX imports
_early_parser = argparse.ArgumentParser(add_help=False)
_early_parser.add_argument("--no_gpu", action="store_true")
_early_args, _ = _early_parser.parse_known_args()
if _early_args.no_gpu:
    # Force JAX to use CPU backend
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
import yaml
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

from model import TransformerDecoder
from tokenizer import load_tokenizer

def top_k_top_p_filter(logits, top_k=0, top_p=1.0):
    """Filter logits with top-k and/or nucleus (top-p) filtering."""
    # logits: 1D numpy array
    if top_k > 0:
        # Remove all tokens not in top-k
        indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
        logits[indices_to_remove] = -np.inf
    if top_p < 1.0:
        sorted_indices = np.argsort(-logits)
        sorted_logits = logits[sorted_indices]
        exp_logits = np.exp(sorted_logits - np.max(sorted_logits))
        probs = exp_logits / np.sum(exp_logits)
        cumulative_probs = np.cumsum(probs)
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[0] = False
        remove_indices = sorted_indices[sorted_indices_to_remove]
        logits[remove_indices] = -np.inf
    return logits

def generate_text(params, tokenizer, config, prompt, length=20,
                  temperature=1.0, top_k=0, top_p=1.0):
    """
    Autoregressively generate `length` tokens given a text prompt.
    Supports temperature, top-k, and top-p sampling.
    """
    # Tokenize prompt
    encoding = tokenizer.encode(prompt)
    seq = encoding.ids.copy()
    max_len = config["max_seq_length"]
    # Instantiate model for inference
    model = TransformerDecoder(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_length=max_len,
        dropout_rate=config.get("dropout_rate", 0.1)
    )
    for _ in range(length):
        context = seq[-max_len:]
        input_arr = np.array([context], dtype=np.int32)
        logits = model.apply({'params': params}, input_arr, train=False)
        # Get logits for last position and convert to numpy
        logits = np.array(logits[0, -1], dtype=np.float64)
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        # Apply top-k and top-p filtering
        logits = top_k_top_p_filter(logits, top_k, top_p)
        # Convert to probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        # Sample next token id
        next_id = int(np.random.choice(len(probs), p=probs))
        seq.append(next_id)
    return tokenizer.decode(seq)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate text with a trained Transformer model.")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to training config file.")
    parser.add_argument("--checkpoint_dir", default="logs",
                        help="Directory where checkpoints are saved.")
    parser.add_argument("--length", type=int, default=20,
                        help="Number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (1.0 = greedy).")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-k sampling (0 = disabled).")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus (top-p) sampling (1.0 = disabled).")
    parser.add_argument("--no_gpu", action="store_true",
                        help="Force CPU for generation.")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set device
    if args.no_gpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"

    # Load tokenizer
    tokenizer = load_tokenizer(config.get("tokenizer_path", "tokenizer"))

    # Restore trained parameters
    ckpt_dir = os.path.abspath(args.checkpoint_dir)
    # Initialize model variables to get parameter structure for restore
    model = TransformerDecoder(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_seq_length=config["max_seq_length"],
        dropout_rate=config.get("dropout_rate", 0.1)
    )
    # Create dummy input for init
    rng = jax.random.PRNGKey(config.get("seed", 0))
    dummy_input = jnp.ones((1, config["max_seq_length"]), dtype=jnp.int32)
    variables = model.init(rng, dummy_input)
    init_params = variables.get("params")
    # Restore checkpoint into parameters
    params = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=init_params)

    # Interactive prompt loop
    print("Enter prompt (empty to exit):")
    while True:
        try:
            prompt = input(">>> ")
        except EOFError:
            break
        if not prompt.strip():
            break
        output = generate_text(
            params,
            tokenizer,
            config,
            prompt,
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(output)

if __name__ == "__main__":
    main()