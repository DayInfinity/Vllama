# Image Generation Commands

Commands for running diffusion models to generate images, managing downloaded models, and interacting with running model sessions.

---

## `vllama run` — Generate Images { #run }

Run a diffusion model to generate images from text prompts.

### Syntax

```bash
vllama run <model_name> [--prompt <text>] [--service <service>] [--output_dir <dir>]
```

### Parameters

| Parameter | Short | Default | Description |
|---|---|---|---|
| `model_name` | | required | HuggingFace model ID |
| `--prompt` | `-p` | interactive | Text prompt for generation |
| `--service` | `-s` | local | Cloud service to offload to (`kaggle`) |
| `--output_dir` | `-o` | current dir | Where to save generated images |

### Examples

```bash
# Generate locally (requires GPU or patience on CPU)
vllama run stabilityai/sd-turbo --prompt "A futuristic cityscape at dusk"

# Interactive mode — enter multiple prompts
vllama run stabilityai/sd-turbo
# Prompt> A mountain lake at sunset
# Prompt> A robot in a garden
# Prompt> exit

# Offload to Kaggle's free GPU
vllama run stabilityai/sd-turbo --service kaggle --prompt "A neon dragon"

# Save to specific folder
vllama run stabilityai/sd-turbo -p "Cherry blossoms" -o ./images
```

### Output

Images are saved as:

- `vllama_output_<timestamp>.png` — local generation
- `vllama_kaggle_<timestamp>.png` — Kaggle generation

### GPU Optimization

Vllama automatically adjusts settings based on your hardware:

| Hardware | Precision | Resolution | Notes |
|---|---|---|---|
| GPU > 3GB VRAM | float16 | 512×512 | Full quality |
| GPU ≤ 3GB VRAM | float32 | Reduced steps | Memory-efficient attention enabled |
| CPU | float32 | Reduced steps | Works but slow |

!!! tip "No GPU? Use Kaggle"
    Add `--service kaggle` to any `vllama run` command to run on Kaggle's free T4 GPU. See the [No GPU Guide](../guides/no-gpu.md).

---

## `vllama show models` — List Supported Models { #show-models }

```bash
vllama show models
```

Lists all vision models supported by Vllama with descriptions.

---

## `vllama install` — Download a Model { #install }

Pre-download a model's weights to local cache so the first `run` is instant:

```bash
vllama install stabilityai/sd-turbo
vllama install CompVis/stable-diffusion-v1-4
```

Model weights are cached in `~/.cache/huggingface`.

---

## `vllama list` — List Downloaded Models { #list }

```bash
vllama list models
```

Shows all models currently cached on your machine.

---

## `vllama uninstall` — Remove a Model { #uninstall }

```bash
vllama uninstall stabilityai/sd-turbo
```

Removes the model from local cache to free disk space.

---

## `vllama post` — Send a Prompt to a Running Session { #post }

If you have a model session already running in the background, send a prompt without entering interactive mode:

```bash
vllama post "A waterfall in a rainforest" --output_dir ./outputs
```

---

## `vllama stop` — Stop a Running Session { #stop }

```bash
vllama stop
```

Stops the currently running model session and frees GPU memory.
