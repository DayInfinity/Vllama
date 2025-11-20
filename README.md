# vllama 🦙

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful CLI tool to run vision models like Stable Diffusion locally or on cloud GPUs (Kaggle). Generate stunning images from text prompts with ease!

## ✨ Features

- 🖼️ **Text-to-Image Generation**: Create images from text prompts using Stable Diffusion
- ☁️ **Cloud GPU Support**: Offload computation to Kaggle GPUs when local resources are limited
- 💻 **Local & Remote**: Run models locally or remotely with a single command
- 🎨 **Interactive Mode**: Enter prompts interactively for rapid experimentation
- ⚡ **Optimized**: Automatic VRAM detection and optimization for low-memory GPUs
- 🔒 **Secure**: Input validation and credential management built-in

## 📦 Installation

### From PyPI (Recommended)
```bash
pip install vllama
```

### From Source
```bash
git clone https://github.com/ManvithGopu13/Vllama.git
cd Vllama
pip install -e .
```

## 🚀 Quick Start

### 1. Install a Model
```bash
vllama install stabilityai/sd-turbo
```

### 2. Generate an Image (Local)
```bash
vllama run stabilityai/sd-turbo --prompt "a beautiful sunset over mountains"
```

### 3. Generate Using Kaggle GPU
```bash
# First, login to Kaggle
vllama login --service kaggle --username YOUR_USERNAME --key YOUR_API_KEY

# Run on Kaggle GPU
vllama run stabilityai/sd-turbo --service kaggle --prompt "a futuristic city"
```

## 📖 Usage

### Available Commands

#### Show Available Models
```bash
vllama show models
```

#### Install a Model
```bash
vllama install <model-name>
# Example:
vllama install stabilityai/sd-turbo
```

#### Run a Model

**Single Prompt:**
```bash
vllama run <model-name> --prompt "your prompt here" [--output_dir ./outputs]
```

**Interactive Mode:**
```bash
vllama run <model-name>
# Then enter prompts interactively
# Type 'exit' or 'quit' to stop
```

**Remote Execution (Kaggle):**
```bash
vllama run <model-name> --service kaggle --prompt "your prompt"
```

#### Cloud GPU Commands

**Login to Kaggle:**
```bash
vllama login --service kaggle --username YOUR_USERNAME --key YOUR_API_KEY
```

**Initialize GPU Session:**
```bash
vllama init gpu --service kaggle
```

**Logout:**
```bash
vllama logout
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in your project directory (use `.env.example` as template):

```bash
# Kaggle API Credentials
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# Optional: Custom output directory
VLLAMA_OUTPUT_DIR=./outputs
```

### Kaggle API Setup

1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings/account)
2. Scroll to "API" section
3. Click "Create New API Token"
4. Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
5. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## 🎯 Examples

### Basic Image Generation
```bash
vllama run stabilityai/sd-turbo --prompt "a serene lake at dawn"
```

### Custom Output Directory
```bash
vllama run stabilityai/sd-turbo --prompt "cyberpunk street" --output_dir ./my_images
```

### Using Kaggle GPU
```bash
# Login once
vllama login --service kaggle

# Generate images on Kaggle GPU
vllama run stabilityai/sd-turbo --service kaggle --prompt "fantasy castle"
```

### Interactive Session
```bash
vllama run stabilityai/sd-turbo
# Prompt> a magical forest
# Prompt> a steampunk robot
# Prompt> exit
```

## 🛠️ Supported Models

Currently supported models:
- `stabilityai/sd-turbo` - Fast Stable Diffusion model (recommended)

More models coming soon!

## 💡 Tips

- **Low VRAM?** The tool automatically detects and optimizes for GPUs with ≤3GB VRAM
- **No GPU?** Falls back to CPU (slower but works)
- **Kaggle GPU** is free and provides powerful T4/P100 GPUs
- **Output files** are named with timestamps: `vllama_output_<timestamp>.png`

## 🐛 Troubleshooting

### "CUDA out of memory" error
- The tool automatically reduces parameters for low VRAM
- Try using Kaggle GPU: `--service kaggle`
- Close other GPU-intensive applications

### "Kaggle credentials not found"
- Ensure `~/.kaggle/kaggle.json` exists with correct permissions (600)
- Or use: `vllama login --service kaggle --username <user> --key <key>`

### "Model not found"
- First install the model: `vllama install <model-name>`
- Check model name spelling

## 🔒 Security

- Never commit `.env` or `kaggle.json` to version control
- Keep API keys secure and rotate regularly
- The tool validates inputs to prevent injection attacks
- Credential files are automatically set to secure permissions (600)

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

- **Author**: Gopu Manvith
- **Email**: manvithgopu1394@gmail.com
- **GitHub**: [@ManvithGopu13](https://github.com/ManvithGopu13)

## 🙏 Acknowledgments

- Built with [Diffusers](https://github.com/huggingface/diffusers) by Hugging Face
- Powered by [Stable Diffusion](https://stability.ai/)
- Cloud GPU support via [Kaggle](https://www.kaggle.com/)

---

⭐ If you find this project useful, please consider giving it a star!
