# Quickstart

Five working examples to get you productive in under five minutes.

---

## 1. Generate an Image (No GPU Needed)

If you have a [Kaggle account](https://www.kaggle.com/) (free), you can offload image generation to their GPU:

```bash
# One-time setup: connect your Kaggle account
vllama login --service kaggle --username YOUR_USERNAME --key YOUR_API_KEY

# Generate an image — runs on Kaggle's free T4 GPU, downloads to your machine
vllama run stabilityai/sd-turbo --service kaggle --prompt "A neon-lit city street at night"
```

The image is saved as `vllama_kaggle_<timestamp>.png` in your current directory.

!!! info "Where to find your Kaggle API key"
    Go to [kaggle.com](https://www.kaggle.com) → Account → API → **Create New Token**. This downloads `kaggle.json`. Your username and key are in that file.

---

## 2. AutoML: Train Models on Your CSV

Two commands to go from a raw CSV to a trained model leaderboard:

```bash
# Step 1: Preprocess your dataset
vllama data --path my_data.csv --target target_column

# Step 2: Train all models on the preprocessed data
# (replace the folder name with the one created in Step 1)
vllama train --path ./output_folder_20240101_120000 --target target_column
```

After `vllama train`, open `results/report.html` in your browser for a full interactive leaderboard.

**Don't have a dataset handy?** Grab the Titanic dataset:

```bash
# Download the Titanic dataset
curl -O https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

vllama data --path titanic.csv --target Survived
vllama train --path ./output_folder_* --target Survived
```

---

## 3. Run a Local LLM and Chat With It

```bash
# Terminal 1: Start a local LLM server (downloads model on first run)
vllama run_llm Qwen/Qwen2.5-Coder-0.5B-Instruct
```

```bash
# Terminal 2: Chat with it
vllama chat_llm
```

```
You> Write a Python function to reverse a linked list
Assistant> Here's a function...
```

The model runs entirely on your machine. No internet after the initial download, no API keys, no usage costs.

---

## 4. Detect Objects in an Image

```bash
# From a local file
vllama detect_image --path photo.jpg

# From a URL
vllama detect_image --url https://ultralytics.com/images/bus.jpg
```

Results are saved to the current directory with bounding boxes drawn on the image.

---

## 5. Text-to-Speech

```bash
# Speak text directly
vllama tts --text "Hello from Vllama"

# Interactive mode — enter multiple lines
vllama tts
```

---

## What's Next?

- [Commands Reference](commands/index.md) — every command explained
- [No GPU Guide](guides/no-gpu.md) — full Kaggle GPU workflow
- [AutoML Pipeline Guide](guides/automl-pipeline.md) — deep dive into the ML workflow
- [Local LLM in VS Code](guides/local-llm-vscode.md) — set up the VS Code extension
