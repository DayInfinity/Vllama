# `vllama run_video` — Video Generation

Generate short videos from text prompts using text-to-video models.

---

## Syntax

```bash
vllama run_video <model_name> [--prompt <text>] [--service <service>] [--output_dir <dir>]
```

---

## Parameters

| Parameter | Short | Default | Description |
|---|---|---|---|
| `model_name` | | required | HuggingFace model ID |
| `--prompt` | `-p` | interactive | Text prompt for generation |
| `--service` | `-s` | local | Cloud service (`kaggle`) |
| `--output_dir` | `-o` | current dir | Where to save the output video |

---

## Examples

```bash
# Generate video locally
vllama run_video damo-vilab/text-to-video-ms-1.7b --prompt "A cat playing piano"

# Interactive mode
vllama run_video damo-vilab/text-to-video-ms-1.7b
# Prompt> A sunset over the ocean
# Prompt> exit

# Offload to Kaggle GPU (recommended — video gen is VRAM-heavy)
vllama run_video damo-vilab/text-to-video-ms-1.7b --service kaggle --prompt "A robot dancing"

# Custom output folder
vllama run_video damo-vilab/text-to-video-ms-1.7b -p "Waves crashing on shore" -o ./videos
```

---

## Recommended Model

| Model | Description |
|---|---|
| `damo-vilab/text-to-video-ms-1.7b` | ModelScope's 1.7B text-to-video model — solid quality for short clips |

---

## Notes

!!! warning "Video generation is VRAM-heavy"
    Text-to-video models typically need 8GB+ VRAM for comfortable local use. If you have less, use `--service kaggle` to offload to Kaggle's free T4 GPU.

Output videos are saved as `.mp4` files in the specified output directory.
