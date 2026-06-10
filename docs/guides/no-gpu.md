# Run Heavy AI Models for Free — No GPU Needed

Vllama has a built-in bridge to **Kaggle's free GPU** (NVIDIA T4, ~16GB VRAM). You submit a task from your laptop, it runs on Kaggle's infrastructure, and the output downloads back to your machine automatically.

This means you can run Stable Diffusion, text-to-video models, and 3D reconstruction — even on a 4GB RAM laptop — for free.

---

## What You Need

- A [Kaggle account](https://www.kaggle.com) (free, takes 2 minutes)
- Phone number for verification (Kaggle requires this for GPU access)
- Vllama installed: `pip install vllama`

---

## Step 1: Get Your Kaggle API Key

1. Log in at [kaggle.com](https://www.kaggle.com)
2. Click your profile icon (top right) → **Settings**
3. Scroll to the **API** section
4. Click **Create New Token** — this downloads `kaggle.json`

Open the file. It looks like:

```json
{
  "username": "yourname",
  "key": "abc123yourkeyhere"
}
```

---

## Step 2: Connect Vllama to Kaggle

```bash
vllama login --service kaggle --username yourname --key abc123yourkeyhere
```

If your `kaggle.json` is already at `~/.kaggle/kaggle.json` (default Kaggle CLI location), just run:

```bash
vllama login --service kaggle
```

---

## Step 3: Run Something

Now just add `--service kaggle` to any supported command.

### Generate an Image

```bash
vllama run stabilityai/sd-turbo --service kaggle --prompt "A cyberpunk street at night, neon lights, rain"
```

Vllama will:
1. Create a Kaggle kernel with GPU
2. Install dependencies
3. Run the model
4. Download `vllama_kaggle_<timestamp>.png` to your current directory

Takes about 3–5 minutes total. The actual generation is under 10 seconds on the T4.

### Generate a Video

```bash
vllama run_video damo-vilab/text-to-video-ms-1.7b --service kaggle --prompt "A sunset over the ocean, cinematic"
```

### Generate a 3D Model from an Image

```bash
vllama image3d --path my_object.jpg --service kaggle -o ./3d_output
```

---

## Troubleshooting

**"Kaggle API credentials not found"**

```bash
vllama login --service kaggle --username YOUR_USERNAME --key YOUR_KEY
```

**"GPU quota exceeded"**

Kaggle gives ~30 GPU hours/week for free accounts. If you hit the limit, wait until your quota resets (weekly), or check quota at [kaggle.com/account](https://www.kaggle.com/account).

**Kernel takes too long / times out**

Kaggle notebooks have a 12-hour session limit. Individual inference tasks take 3–10 minutes, well within this.

**First run is slow**

The first run installs dependencies inside the Kaggle kernel. Subsequent runs reuse the environment and are faster.

---

## Kaggle GPU Limits (Free Tier)

| Resource | Limit |
|---|---|
| GPU type | NVIDIA T4 (16GB VRAM) |
| Weekly GPU hours | ~30 hours |
| Session max duration | 12 hours |
| Disk space | 73GB |
| Internet | Enabled |

For most use cases — generating images, videos, 3D models — you'll stay well within the weekly quota.
