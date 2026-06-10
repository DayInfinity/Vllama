# Speech Commands

Text-to-speech, speech-to-text, and text translation — all running locally.

---

## `vllama tts` — Text to Speech { #tts }

Convert text to speech using a local TTS engine. Audio plays through your system speakers.

### Syntax

```bash
vllama tts [--text <text>]
```

### Examples

```bash
# Speak a string directly
vllama tts --text "Hello from Vllama. This is a test."

# Interactive mode — enter text at the prompt
vllama tts
# Enter text: Hello world
```

---

## `vllama stt` — Speech to Text { #stt }

Transcribe speech from your microphone or an audio file using Google Speech Recognition.

### Syntax

```bash
vllama stt [--path <audio_file>] [--language <lang_code>]
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--path` | microphone | Path to an audio file to transcribe |
| `--language` | `en-US` | Language code for recognition |

### Examples

```bash
# Transcribe from microphone
vllama stt
# Output: Listening... Speak now!
# Output: Transcribed: Hello world

# Transcribe from file
vllama stt --path recording.wav

# Specify language
vllama stt --language hi-IN   # Hindi
vllama stt --language te-IN   # Telugu
```

### Notes

- Microphone input requires a working audio input device
- Uses Google Speech Recognition under the hood — requires internet for the actual transcription
- `--path` supports WAV and other common audio formats

---

## `vllama translate` — Text Translation { #translate }

Translate text between languages using a local NLLB (No Language Left Behind) model from Meta. Runs entirely offline after the first model download.

### Syntax

```bash
vllama translate --text <text> [--src <lang>] [--tgt <lang>] [--model <model_id>]
```

### Parameters

| Parameter | Short | Default | Description |
|---|---|---|---|
| `--text` | | required | Text to translate |
| `--src` | | `en` | Source language code |
| `--tgt` | | `fr` | Target language code |
| `--model` | | NLLB default | HuggingFace model ID to use |

### Examples

```bash
# English to French
vllama translate --text "Hello, how are you?" --src en --tgt fr

# French to English
vllama translate --text "Bonjour le monde" --src fr --tgt en

# English to Hindi
vllama translate --text "Good morning" --src en --tgt hi

# English to Telugu
vllama translate --text "Welcome to Vllama" --src en --tgt te
```

### Language Codes

NLLB uses BCP-47 style codes. Common ones:

| Language | Code |
|---|---|
| English | `en` |
| Hindi | `hi` |
| Telugu | `te` |
| Tamil | `ta` |
| French | `fr` |
| Spanish | `es` |
| German | `de` |
| Chinese (Simplified) | `zh` |
| Arabic | `ar` |
| Portuguese | `pt` |

For the full list, refer to the [NLLB language list](https://github.com/facebookresearch/flores/blob/main/flores200/README.md).
