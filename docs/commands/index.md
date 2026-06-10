# Command Reference

All Vllama commands, organised by category.

---

## Quick Reference

| Command | What it does |
|---|---|
| [`vllama run`](vision.md) | Generate images with diffusion models |
| [`vllama run_video`](video.md) | Generate video from text prompts |
| [`vllama detect_image`](detection.md) | YOLO object detection on an image |
| [`vllama detect_video`](detection.md) | YOLO object detection on a video |
| [`vllama image3d`](detection.md#image3d) | Generate a 3D model from an image (Kaggle) |
| [`vllama video3d`](detection.md#video3d) | Generate a 3D model from a video (Kaggle) |
| [`vllama view3d`](viewer.md) | View a 3D model file interactively |
| [`vllama run_llm`](llm.md) | Start a local LLM as a REST API server |
| [`vllama chat_llm`](llm.md#chat_llm) | CLI chat with the running LLM server |
| [`vllama tts`](speech.md) | Convert text to speech |
| [`vllama stt`](speech.md#stt) | Transcribe speech from mic or file |
| [`vllama translate`](speech.md#translate) | Translate text using a local NLLB model |
| [`vllama login`](cloud.md) | Authenticate with Kaggle |
| [`vllama logout`](cloud.md#logout) | Remove saved credentials |
| [`vllama init gpu`](cloud.md#init-gpu) | Initialize a cloud GPU session |
| [`vllama show models`](vision.md#show-models) | List all supported vision models |
| [`vllama install`](vision.md#install) | Pre-download a model to cache |
| [`vllama list`](vision.md#list) | List downloaded models |
| [`vllama uninstall`](vision.md#uninstall) | Remove a downloaded model from cache |
| [`vllama post`](vision.md#post) | Send a prompt to a running model session |
| [`vllama stop`](vision.md#stop) | Stop the running model session |
| [`vllama data`](data.md) | Preprocess and clean a CSV/Excel/JSON dataset |
| [`vllama train`](train.md) | AutoML: train and compare multiple models |

---

## Global Flags

These flags work across all commands:

| Flag | Description |
|---|---|
| `--help` | Show help for the command |
| `--version` | Show the installed Vllama version |
