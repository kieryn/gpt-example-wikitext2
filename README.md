# Small Transformer LM with Flax

This project trains a Transformer decoder-only language model on WikiText-2 using JAX/Flax.

## Setup

1. Create and activate a virtual environment:

   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Upgrade pip and install dependencies:

   ```
   pip install --upgrade pip
   pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   pip install -r requirements.txt
   ```

3. Download the WikiText-2 data:

   ```
   python data/download_wikitext2.py
   ```

4. Train the model:

   ```
   python src/train.py --config configs/config.yaml
   ```

## Directory Structure

```
.
├── .venv/                  # Python virtual environment
├── data/                   # Scripts and processed data
│   └── download_wikitext2.py
├── configs/                # Configuration files
│   └── config.yaml
├── logs/                   # Monitoring & checkpoint logs
├── requirements.txt        # Python dependencies
└── src/                    # Source code
    ├── tokenizer.py
    ├── model.py
    └── train.py
```
