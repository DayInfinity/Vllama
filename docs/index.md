# Vllama

<div class="hero" markdown>

# Vllama 🦙

**One CLI for everything AI — locally or on free cloud GPUs.**

Image generation · AutoML · Local LLMs · Speech · Object Detection · 3D · VS Code

<div class="hero-badges" markdown>
[![PyPI](https://img.shields.io/pypi/v/vllama?color=7c3aed&label=PyPI)](https://pypi.org/project/vllama/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Stars](https://img.shields.io/github/stars/DayInfinity/Vllama?style=social)](https://github.com/DayInfinity/Vllama)
</div>

```bash
pip install vllama
```

[Get Started](installation.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/DayInfinity/Vllama){ .md-button }

</div>

---

## What is Vllama?

Vllama is a single CLI tool that puts state-of-the-art AI at your fingertips — without needing a powerful GPU or writing any code.

!!! tip "No GPU? No problem."
    Vllama can offload heavy models like Stable Diffusion to **Kaggle's free GPU** and automatically download the results to your machine. All you need is a Kaggle account.

---

## Features

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
<div class="icon">🎨</div>
**Image Generation**

Run Stable Diffusion and other diffusion models locally or on Kaggle GPUs.
</div>

<div class="feature-card" markdown>
<div class="icon">🤖</div>
**Local LLMs**

Spin up any HuggingFace chat model as a local REST API server in one command.
</div>

<div class="feature-card" markdown>
<div class="icon">🆚</div>
**VS Code Extension**

Chat with your locally running LLM directly inside VS Code's native chat panel — zero API keys.
</div>

<div class="feature-card" markdown>
<div class="icon">📷</div>
**Object Detection**

Run YOLO on images and videos from the terminal instantly.
</div>

<div class="feature-card" markdown>
<div class="icon">🎬</div>
**Video Generation**

Generate videos from text prompts using text-to-video models.
</div>

<div class="feature-card" markdown>
<div class="icon">🔊</div>
**Speech**

Text-to-speech and speech-to-text using local models — no cloud API needed.
</div>

<div class="feature-card" markdown>
<div class="icon">🖼️</div>
**Image/Video to 3D**

Generate 3D `.ply` models from images or videos via Kaggle GPU.
</div>

<div class="feature-card" markdown>
<div class="icon">🏆</div>
**AutoML**

Preprocess any CSV and auto-train 9+ ML models with hyperparameter tuning in two commands.
</div>

</div>

---

## 5-Minute Examples

=== "Image Generation"
    ```bash
    # Login to Kaggle once (free account)
    vllama login --service kaggle --username YOU --key YOUR_KEY

    # Generate an image on free Kaggle GPU
    vllama run stabilityai/sd-turbo --service kaggle --prompt "A cyberpunk city at night"
    ```

=== "AutoML"
    ```bash
    # Step 1: Preprocess your CSV
    vllama data --path housing.csv --target price

    # Step 2: Train 9 models and get a leaderboard
    vllama train --path ./output_folder_YYYYMMDD_HHMMSS --target price
    ```

=== "Local LLM"
    ```bash
    # Terminal 1: Start local LLM server
    vllama run_llm Qwen/Qwen2.5-Coder-0.5B-Instruct

    # Terminal 2: Chat with it
    vllama chat_llm
    ```

=== "Object Detection"
    ```bash
    # Detect objects in a photo
    vllama detect_image --path photo.jpg

    # Detect objects in a video
    vllama detect_video --path video.mp4
    ```

---

## Installation

```bash
pip install vllama
```

That's it. See [Installation](installation.md) for environment-specific setup and optional dependency groups.

---

## Next Steps

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
[**📦 Installation →**](installation.md)

Full install guide including optional dependencies and environment setup.
</div>

<div class="feature-card" markdown>
[**🚀 Quickstart →**](quickstart.md)

Five hands-on examples to get you running in minutes.
</div>

<div class="feature-card" markdown>
[**☁️ No GPU Guide →**](guides/no-gpu.md)

Run heavy models for free using Kaggle's GPU — Vllama's standout feature.
</div>

<div class="feature-card" markdown>
[**📚 Command Reference →**](commands/index.md)

Every command, every flag, every output explained.
</div>

</div>
