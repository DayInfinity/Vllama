# Local LLM Commands

Run any HuggingFace chat model as a local REST API and chat with it — no API keys, no internet after the first download, no usage costs.

---

## `vllama run_llm` — Start a Local LLM Server { #run-llm }

Downloads and starts any HuggingFace chat model as a REST API server on `localhost:2513`.

### Syntax

```bash
vllama run_llm <model_name>
```

### Examples

```bash
# Qwen Coder (small, fast, great for code)
vllama run_llm Qwen/Qwen2.5-Coder-0.5B-Instruct

# Llama 2 Chat (requires HuggingFace token for gated access)
vllama run_llm meta-llama/Llama-2-7b-chat-hf

# Microsoft DialoGPT
vllama run_llm microsoft/DialoGPT-medium
```

### What Starts Up

Once running, the server:

- Listens on `http://localhost:2513`
- Exposes a `/chat` endpoint
- Maintains conversation history across turns
- Works with `vllama chat_llm` and the VS Code extension

### API Endpoint

You can also talk to the server directly via HTTP:

```bash
curl -X POST http://localhost:2513/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Write a Python function to sort a list"}'
```

Response:

```json
{
  "response": "Here's a Python function..."
}
```

### Notes

- The server runs in the foreground. Press `Ctrl+C` to stop it.
- Models are cached in `~/.cache/huggingface` after first download.
- Gated models (like Llama 2) require a HuggingFace token: set `HF_TOKEN=your_token` in your environment.

---

## `vllama chat_llm` — CLI Chat { #chat-llm }

Connect to a running LLM server and chat interactively in the terminal.

### Syntax

```bash
vllama chat_llm
```

### Usage

First, start the server in one terminal:

```bash
# Terminal 1
vllama run_llm Qwen/Qwen2.5-Coder-0.5B-Instruct
```

Then chat in another:

```bash
# Terminal 2
vllama chat_llm
```

```
You> What is a closure in Python?
Assistant> A closure is a function that remembers the values from its enclosing scope...

You> Give me a code example
Assistant> Sure! Here's a simple closure example...

You> exit
```

Type `exit` or `quit` to end the session.

### Notes

- Requires a running `vllama run_llm` server
- Conversation context is maintained for the full session
- The VS Code extension uses the same server — both can connect simultaneously

---

## VS Code Integration

The Vllama VS Code extension connects to the same `localhost:2513` server. See the [Local LLM in VS Code guide](../guides/local-llm-vscode.md) for the full setup walkthrough.

---

## Model Recommendations

| Use Case | Recommended Model | Size |
|---|---|---|
| Code generation | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | ~1GB |
| General chat | `microsoft/DialoGPT-medium` | ~1.5GB |
| Capable general | `meta-llama/Llama-2-7b-chat-hf` | ~14GB (gated) |
| Low-spec machine | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | ~1GB |
