# Vllama

Vllama is a powerful yet simple command-line tool designed to make running vision models (like Stable Diffusion) easy for everyone. Whether you have a powerful local GPU or need to offload the heavy lifting to the cloud (Kaggle), Vllama handles it seamlessly.

It also includes autonomous data preprocessing tools to help you clean and prepare your datasets for machine learning with a single command.

## Features

*   **🚀 Run Locally**: Generate high-quality images on your own machine using models like `stabilityai/sd-turbo`.
*   **☁️ Cloud Execution**: Seamlessly offload generation to Kaggle GPUs if your local hardware is limited.
*   **💬 Interactive Mode**: Keep the model loaded and generate multiple images in a chat-like session for faster results.
*   **🧹 Autonomous Data Cleaning**: Automatically handle missing values, encode categories, scale features, and detect outliers in your datasets.
*   **📦 Model Management**: Easily download, install, and manage different vision models.
*   **🔐 Secure & Private**: Your credentials and data stay with you.

## Installation

You can install Vllama directly from the source:

```bash
git clone https://github.com/ManvithGopu13/Vllama.git
cd Vllama
pip install -r requirements.txt
```

## Quick Start

**1. Generate an image locally:**
```bash
vllama run stabilityai/sd-turbo --prompt "A cyberpunk city at night"
```

**2. Generate an image on Kaggle (requires login):**

Requires Kaggle API credentials.

```bash
vllama login --service kaggle --username YOUR_USER --key YOUR_KEY
vllama run stabilityai/sd-turbo --prompt "A cat in space" --service kaggle
```

**3. Clean a dataset:**
```bash
vllama data --path my_data.csv --target price
```

## Configuration Guide (Kaggle Setup)

To use Vllama with Kaggle, you need your API credentials.

1.  Go to your **Kaggle Account Settings**.
2.  Scroll down to the **API** section.
3.  Click **Create New Token** to download `kaggle.json`.

You can then configure Vllama in two ways:

**Option A: Login Command (Recommended)**
Run the following command to securely store your credentials:
```bash
vllama login --service kaggle --username <your_username> --key <your_api_key>
```

**Option B: Manual Setup**
Place your `kaggle.json` file in the default location:
*   **Windows**: `C:\Users\<User>\.kaggle\kaggle.json`
*   **Linux/Mac**: `~/.kaggle/kaggle.json`

## Command Reference

Here is the full list of commands you can use with Vllama:

### Core Commands

| Command | Description | Example |
| :--- | :--- | :--- |
| `vllama run <model>` | Run a model. Add `--prompt` for a single image, or leave empty for interactive mode. | `vllama run stabilityai/sd-turbo` |
| `vllama install <model>` | Download and install a model for local use. | `vllama install stabilityai/sd-turbo` |
| `vllama show models` | List all available models you can use. | `vllama show models` |
| `vllama stop` | Stop the currently running model session to free up memory. | `vllama stop` |

### Remote & Cloud Commands

| Command | Description | Example |
| :--- | :--- | :--- |
| `vllama login` | Log in to a cloud service. Options: `--service`, `--username`, `--key`. | `vllama login --service kaggle` |
| `vllama init gpu` | Initialize a GPU session on a remote service. | `vllama init gpu --service kaggle` |
| `vllama logout` | Log out and remove stored credentials. | `vllama logout` |

### Data Tools

| Command | Description | Example |
| :--- | :--- | :--- |
| `vllama data` | Preprocess a dataset. Options: `--path`, `--target`, `--test_size`, `--output_dir`. | `vllama data --path data.csv` |

### Arguments Guide

*   `--prompt`, `-p`: The text description for the image you want to generate.
*   `--service`, `-s`: The cloud service to use (currently supports `kaggle`).
*   `--output_dir`, `-o`: Where to save the generated images or processed data.
*   `--test_size`, `-t`: The proportion of the dataset to include in the test split (e.g., `0.2` for 20%).

## Troubleshooting

| Issue | Possible Cause | Solution |
| :--- | :--- | :--- |
| **`CUDA out of memory`** | Your GPU doesn't have enough VRAM. | Try running with `--service kaggle` to use cloud GPUs. |
| **`403 Forbidden` (Kaggle)** | Invalid or expired API credentials. | Run `vllama logout` then `vllama login` with new keys. |
| **`Model not found`** | The model name is incorrect or not installed. | Check spelling or run `vllama install <model>`. |
| **`ImportError`** | Missing dependencies. | Run `pip install -r requirements.txt`. |

## Security Best Practices

*   **Protect your API Keys**: Never share your `kaggle.json` or commit it to public repositories.
*   **Review Prompts**: When using remote execution, avoid sending sensitive personal information in prompts.
*   **Check Permissions**: Ensure your output directories have appropriate permissions if working on a shared machine.
*   **Report Vulnerabilities**: See [SECURITY.md](SECURITY.md) for how to report security issues.

## Future Roadmap

We are actively working on these exciting new features:

*   **Model Management**: Commands to list installed models and remove old ones.
*   **Progress Bars**: Visual indicators for downloads and long-running tasks.
*   **Configuration**: Save your preferences (like default model) in a config file.
*   **Batch Processing**: Generate hundreds of images from a list of prompts in one go.
*   **Advanced Editing**: Image-to-Image generation and Inpainting support.
*   **Web UI**: A beautiful browser-based interface for those who prefer not to use the terminal.
*   **Negative Prompts**: Specify what you *don't* want in your images (e.g., "blurry", "low quality").

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to get started.

## License

This project is open source.
