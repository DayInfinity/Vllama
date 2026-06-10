# FAQ

---

## General

**What is Vllama?**

Vllama is a CLI tool that lets you run AI models locally — or offload them to free Kaggle GPUs — from your terminal. It covers image generation, AutoML, local LLMs, object detection, speech, 3D, and more.

**Do I need a GPU?**

No. Vllama runs on CPU and automatically falls back gracefully. For heavy models like Stable Diffusion, use `--service kaggle` to run on Kaggle's free T4 GPU instead.

**Is it free?**

Yes. Vllama is open source (Apache 2.0). Kaggle GPU offload uses Kaggle's free tier (~30 GPU hours/week).

**What Python versions are supported?**

Python 3.8 and above.

---

## Installation

**`pip install vllama` fails with a dependency error**

Try upgrading pip first:

```bash
pip install --upgrade pip
pip install vllama
```

If a specific package fails, check if you need system-level dependencies (e.g., `portaudio` for speech features on macOS/Linux).

**Can I install just part of Vllama?**

Currently `pip install vllama` installs the full package. Heavy dependencies (PyTorch, Diffusers, etc.) are imported lazily and only download model weights when you first use a feature.

---

## Kaggle GPU

**How do I get my Kaggle API key?**

Log in at [kaggle.com](https://www.kaggle.com) → Settings → API → Create New Token. This downloads `kaggle.json` containing your `username` and `key`.

**My Kaggle GPU quota is exceeded**

Kaggle gives ~30 GPU hours per week for free accounts. Quota resets weekly. Check your usage at [kaggle.com/account](https://www.kaggle.com/account).

**The Kaggle run is stuck / never downloads**

Check the Kaggle kernel status at [kaggle.com/code](https://www.kaggle.com/code). If the kernel errored, check its output logs for the error message.

---

## Image Generation

**Generation is slow on CPU**

This is expected. Diffusion models are designed for GPUs. Use `--service kaggle` to run on a free Kaggle GPU, or use `stabilityai/sd-turbo` which is specifically optimized for speed.

**"CUDA out of memory"**

Vllama automatically falls back to CPU or uses memory-efficient settings for low-VRAM GPUs. If you still get OOM, try `--service kaggle`.

---

## AutoML

**"Target column not found"**

Specify the column name explicitly:

```bash
vllama data --path data.csv --target your_actual_column_name
```

Use `head data.csv` or `python -c "import pandas as pd; print(pd.read_csv('data.csv').columns.tolist())"` to check your column names.

**Training takes very long**

Large datasets with hyperparameter search take longer. For faster testing, subsample your data first. Training time scales with dataset size × number of models.

---

## Local LLMs

**"VS Code extension can't connect to local LLM"**

Make sure `vllama run_llm <model>` is running and shows `Running on http://localhost:2513` before opening VS Code chat. Check extension settings for the correct port.

**Model downloads are slow**

HuggingFace models download from the internet the first time. A 7B model is 10–14GB. Subsequent runs use the local cache at `~/.cache/huggingface`.

---

## Speech

**"No microphone found" when running `vllama stt`**

Make sure your system has a working microphone and audio drivers. On Linux, also install: `sudo apt-get install portaudio19-dev`.

**`vllama stt` requires internet?**

Yes, the STT command uses Google Speech Recognition which sends audio to Google's API. The TTS and translation commands (`vllama tts`, `vllama translate`) run fully offline.

---

## Getting Help

- **GitHub Issues**: [github.com/DayInfinity/Vllama/issues](https://github.com/DayInfinity/Vllama/issues)
- **GitHub Discussions**: [github.com/DayInfinity/Vllama/discussions](https://github.com/DayInfinity/Vllama/discussions)
- **Email**: manvithgopu1394@gmail.com
