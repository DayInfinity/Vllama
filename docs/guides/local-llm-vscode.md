# Local LLM in VS Code

Use Vllama to chat with a local language model directly inside VS Code's native chat panel — no API keys, no subscriptions, no data sent to the cloud.

---

## How It Works

Vllama runs a local LLM as an HTTP server on your machine. The Vllama VS Code extension connects to that server and plugs it into VS Code's native **Chat with AI** panel. Your model, your hardware, your data.

```
Your Machine
┌────────────────────────────────────────────┐
│  vllama run_llm (Flask server :2513)        │
│         ↕  HTTP                             │
│  VS Code Vllama Extension                  │
│         ↕  VS Code Chat API                 │
│  VS Code Chat panel                         │
└────────────────────────────────────────────┘
```

---

## Step 1: Install the VS Code Extension

1. Open VS Code
2. Press `Ctrl+Shift+X` (Windows/Linux) or `Cmd+Shift+X` (macOS)
3. Search for **Vllama**
4. Click **Install**
5. Reload VS Code when prompted

---

## Step 2: Start the Local LLM Server

In your terminal:

```bash
vllama run_llm Qwen/Qwen2.5-Coder-0.5B-Instruct
```

On first run, this downloads the model weights (~1GB for Qwen 0.5B). Wait for:

```
* Running on http://localhost:2513
```

Keep this terminal open — closing it stops the server.

---

## Step 3: Chat in VS Code

1. In VS Code, open the Chat panel: **View → Chat with AI** (or `Ctrl+Alt+I`)
2. Select your Vllama local model from the model dropdown
3. Start chatting

---

## Recommended Models by Use Case

| Goal | Model | Download Size |
|---|---|---|
| Code help, fast | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | ~1 GB |
| General assistant (low RAM) | `microsoft/DialoGPT-medium` | ~1.5 GB |
| Better quality (needs 8GB+ RAM) | `meta-llama/Llama-2-7b-chat-hf` | ~14 GB |

---

## Troubleshooting

**"VS Code extension can't connect"**

- Make sure `vllama run_llm` is running and shows `Running on http://localhost:2513`
- Check VS Code extension settings — the default port is `2513`
- On Linux, check if a firewall is blocking localhost connections: `sudo ufw status`

**Model responses are slow**

Smaller models (0.5B–1B parameters) respond in seconds even on CPU. Larger models (7B+) need a GPU or are slow on CPU. Try `Qwen/Qwen2.5-Coder-0.5B-Instruct` for fast responses.

**"Model not found"**

Verify the model ID is valid on [huggingface.co](https://huggingface.co). Gated models (Llama 2) require a HuggingFace token:

```bash
export HF_TOKEN=your_huggingface_token
vllama run_llm meta-llama/Llama-2-7b-chat-hf
```

---

## Also: Chat from the Terminal

You don't need VS Code — you can also chat directly from a second terminal window:

```bash
# Terminal 2 (while server is running in Terminal 1)
vllama chat_llm
```

Both the CLI chat and VS Code extension can connect to the same server simultaneously.
