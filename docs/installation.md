# Installation

## Requirements

- Python 3.8 or higher
- pip

That's all you need to get started. Heavy dependencies like PyTorch and Diffusers are pulled in on first use, not at install time.

---

## Install from PyPI

```bash
pip install vllama
```

Verify the install:

```bash
vllama --help
```

---

## System-specific Notes

=== "Windows"
    Install works normally on Windows. For Speech commands (`vllama stt`), you may need to install PortAudio:

    ```bash
    pip install pipwin
    pipwin install pyaudio
    ```

=== "macOS"
    PortAudio is required for speech features:

    ```bash
    brew install portaudio
    pip install vllama
    ```

=== "Linux"
    ```bash
    sudo apt-get install portaudio19-dev  # for speech features
    pip install vllama
    ```

---

## Virtual Environment (Recommended)

Running in a virtual environment keeps your global Python clean:

```bash
python -m venv vllama-env

# Activate
source vllama-env/bin/activate   # Linux / macOS
vllama-env\Scripts\activate      # Windows

pip install vllama
```

---

## GPU Support (Optional but Faster)

Vllama works on CPU out of the box. For GPU acceleration, install PyTorch with CUDA support **before** installing vllama:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install vllama
pip install vllama
```

!!! tip "No GPU locally?"
    Use Vllama's built-in [Kaggle GPU offload](guides/no-gpu.md) to run heavy models for free without any local GPU.

---

## VS Code Extension

To use the VS Code chat integration:

1. Open VS Code
2. Press `Ctrl+Shift+X` (or `Cmd+Shift+X` on macOS)
3. Search for **Vllama**
4. Click **Install**

The extension connects to a local LLM server you start with `vllama run_llm`. See the [Local LLM in VS Code guide](guides/local-llm-vscode.md).

---

## Updating

```bash
pip install --upgrade vllama
```

---

## Uninstalling

```bash
pip uninstall vllama
```

Model weights cached by HuggingFace are stored in `~/.cache/huggingface`. Remove that directory to free disk space.
