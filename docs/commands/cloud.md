# Cloud & GPU Commands

Vllama integrates with Kaggle to let you offload compute-heavy tasks to their **free GPU** — no payment required, just a Kaggle account.

---

## `vllama login` — Authenticate with Kaggle { #login }

Save your Kaggle credentials so Vllama can submit kernels on your behalf.

### Syntax

```bash
vllama login --service kaggle [--username <user>] [--key <api_key>]
```

### How to Get Your Kaggle API Key

1. Go to [kaggle.com](https://www.kaggle.com) and log in
2. Click your profile picture → **Settings**
3. Scroll to **API** section → **Create New Token**
4. A `kaggle.json` file downloads containing your `username` and `key`

### Examples

```bash
# Provide credentials directly
vllama login --service kaggle --username manvith --key abcdef1234567890

# Use existing kaggle.json at ~/.kaggle/kaggle.json
vllama login --service kaggle
```

---

## `vllama logout` — Remove Credentials { #logout }

```bash
vllama logout
```

Removes saved Kaggle credentials from your machine.

---

## `vllama init gpu` — Initialize a GPU Session { #init-gpu }

Prepares a GPU-enabled Kaggle kernel for use.

```bash
vllama init gpu --service kaggle
```

This is typically called automatically when you run commands with `--service kaggle`. You can run it manually to verify connectivity.

---

## Using Kaggle GPU with Any Command

Add `--service kaggle` to these commands to offload to Kaggle's free T4 GPU:

```bash
# Image generation on Kaggle GPU
vllama run stabilityai/sd-turbo --service kaggle --prompt "..."

# Video generation on Kaggle GPU
vllama run_video damo-vilab/text-to-video-ms-1.7b --service kaggle --prompt "..."

# Image to 3D on Kaggle GPU
vllama image3d --path photo.jpg --service kaggle

# Video to 3D on Kaggle GPU
vllama video3d --path clip.mp4 --service kaggle
```

---

## How It Works

When you add `--service kaggle`:

1. Vllama creates a Kaggle notebook kernel via the Kaggle API
2. The kernel runs with a free T4 GPU enabled
3. Vllama installs the required dependencies inside the kernel
4. The model runs and generates the output
5. The output file is downloaded to your local machine automatically

!!! info "Kaggle GPU limits"
    Kaggle provides approximately **30 GPU hours per week** for free accounts. Each image generation takes a few minutes.

---

## Environment Variables

Alternatively, set credentials as environment variables instead of using `vllama login`:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

Or create a `.env` file in your project directory:

```bash
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
HF_HOME=/path/to/model/cache        # optional: custom model cache dir
HF_TOKEN=your_huggingface_token     # optional: for gated models like Llama 2
```

For a detailed step-by-step walkthrough, see the [No GPU Guide](../guides/no-gpu.md).
