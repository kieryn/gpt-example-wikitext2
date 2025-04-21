"""
Interactive text generation with the trained Transformer model.
"""
import os
import yaml
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

from model import TransformerDecoder
from tokenizer import load_tokenizer

def generate_text(params, tokenizer, config, prompt, length=20):
    """
    Autoregressively generate `length` tokens given a text prompt.
    Uses greedy sampling (argmax).
    """
    # Tokenize the prompt
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
    # Generate tokens
    for _ in range(length):
        # Prepare input (truncate to last max_len tokens)
        context = seq[-max_len:]
        input_arr = np.array([context], dtype=np.int32)
        # Get logits for next token
        logits = model.apply({'params': params}, input_arr, train=False)
        logits = np.array(logits[0, -1])  # (vocab_size,)
        next_id = int(np.argmax(logits))
        seq.append(next_id)
    # Decode full sequence
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
    params = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None)

    # Interactive prompt loop
    print("Enter prompt (empty to exit):")
    while True:
        try:
            prompt = input(">>> ")
        except EOFError:
            break
        if not prompt.strip():
            break
        output = generate_text(params, tokenizer, config, prompt, length=args.length)
        print(output)

if __name__ == "__main__":
    main()