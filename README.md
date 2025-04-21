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
   
  **Note for WSL2 GPU users**: Ensure you have both the NVIDIA CUDA toolkit and cuDNN libraries installed inside WSL2. For example:
  ```bash
  sudo apt update
  # Install CUDA toolkit (version may vary, e.g. 12.8) from the local CUDA repo
  sudo apt update
  sudo apt install -y cuda-12-8
  # (Optional) To enable deeper GPU support, install cuDNN for your CUDA version:
  # 1. Visit https://developer.nvidia.com/rdp/cudnn-download and download the cuDNN Debian package matching your CUDA version and OS.
  # 2. Install the local repository package, e.g.:
  #    wget https://developer.download.nvidia.com/compute/cudnn/<version>/local_installers/cudnn-local-repo-*-deb
  #    sudo dpkg -i cudnn-local-repo-*.deb
  #    sudo cp /var/cudnn-local-repo-*/cudnn-local.keyring /usr/share/keyrings/
  #    sudo apt update
  #    sudo apt install libcudnn8 libcudnn8-dev
  ```
  Then add the driver library path to the loader:
  ```bash
  echo "/usr/lib/wsl/lib" | sudo tee /etc/ld.so.conf.d/nvidia.conf
  sudo ldconfig
  ```
  Verify with:
  ```bash
  python - << 'EOF'
  import jax; import jax.numpy as jnp
  print("Devices:", jax.devices())
  EOF
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
    ├── train.py
    └── generate.py          # Interactive text generation
```

## Generation

After training, you can generate text interactively:

```bash
source .venv/bin/activate
# (optional) force CPU for generation
JAX_PLATFORM_NAME=cpu python src/generate.py --config configs/config.yaml --length 20
```

At the prompt, enter a starting text (empty input to exit) and the model will produce the next tokens.
