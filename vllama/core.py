import os
import time
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline, DPMSolverMultistepScheduler
from huggingface_hub import scan_cache_dir
from huggingface_hub.constants import HF_HUB_CACHE
import numpy as np
import requests
import imageio
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, pipeline
import soundfile as sf
import re

_pipeline = None


_SUPPORTED_MODELS = {
    "stabilityai/sd-turbo": "StabilityAI SD-Turbo (distilled Stable Diffusion 2.1, fast text-to-image model)",
    "damo-vilab/text-to-video-ms-1.7b": "DAMO VILAB Text-to-Video MS 1.7B (text-to-video generation model)",
}


# Show available Models
def show_models():
    """List available models for installation."""
    print("Supported models:")
    for name, desc in _SUPPORTED_MODELS.items():
        print(f"- {name}: {desc}")


def _get_hf_cache_dir() -> str:
    """
    Resolve the Hugging Face cache dir, respecting env vars if set.
    """
    env_cache = os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("HF_HOME")
    if env_cache:
        return env_cache
    return HF_HUB_CACHE


def list_downloads():
    """
    List all Hugging Face *model* repos that are already downloaded
    in the local cache (including Stable Diffusion models).
    """
    try:
        cache_dir = _get_hf_cache_dir()
        cache_info = scan_cache_dir(cache_dir=cache_dir)
    except Exception as e:
        print(f"Error scanning local Hugging Face cache: {e}")
        return

    # Collect only model repos (ignoring datasets/spaces)
    models = sorted({repo for repo in cache_info.repos if repo.repo_type == "model"})

    if not models:
        print("No downloaded models found in the local Hugging Face cache.")
        return

    print("Downloaded models in Hugging Face cache:")
    for m in models:
        print(f" - {m.repo_id} ---  size: {m.size_on_disk/(1024**3):.2f} GB")


# Install Model
def install_model(model_name:str):
    """Download the model weights for the given model from Hugging Face."""
    print(f"Installing model '{model_name}'...")
    try:
        # This will download the model and cache it.
        _ = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        print(f"Model '{model_name}' downloaded successfully.")
    except Exception as e:
        print(f"Error downloading model {model_name}: {e}")


# Uninstall model from cache
def uninstall_model(model_name: str):
    """Remove a previously downloaded model from the local Hugging Face cache."""
    print(f"Uninstalling model '{model_name}'...")

    cache_info = scan_cache_dir()

    # Find all cached repos matching this model id (e.g. "stabilityai/sd-turbo")
    matching_repos = [r for r in cache_info.repos if r.repo_id == model_name]

    if not matching_repos:
        print(f"Model '{model_name}' was not found in the local cache.")
        return

    # Collect all revision hashes for this repo
    revision_hashes = []
    for repo in matching_repos:
        for rev in repo.revisions:
            revision_hashes.append(rev.commit_hash)

    if not revision_hashes:
        print(f"No cached revisions found for '{model_name}'.")
        return

    # Plan deletion and show how much space we’ll free
    delete_strategy = cache_info.delete_revisions(*revision_hashes)
    if delete_strategy.expected_freed_size == 0:
        print(f"No files to delete for '{model_name}'.")
        return

    print(f"Freeing {delete_strategy.expected_freed_size_str} of disk space...")
    delete_strategy.execute()
    print(f"Model '{model_name}' removed from local Hugging Face cache.")


# Run Model
def run_model(model_name: str, prompt: str = None, output_dir: str = "."):
    """
    Run the specified model. If prompt is given, generate an image for it.
    If prompt is None, enter interactive mode to accept prompts repeatedly.
    """
    global _pipeline

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"CUDA device: {props.name}, VRAM: {vram_gb:.2f} GB")

        if vram_gb <= 3:
            device = "cuda"
            dtype = torch.float32
            low_vram = True
        else:
            device = "cuda"
            dtype = torch.float16
            low_vram = False
        
        print("CUDA device detected. Using GPU for inference.")

    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
        low_vram = False
        print("MPS device detected. Using GPU for inference.")

    else:
        device = "cpu"
        dtype = torch.float32
        low_vram = True
        print("No CUDA device detected. Using CPU for inference (may be slow).")


    # Load the model pipeline if not already loaded or if a different model is requested
    if _pipeline is None or getattr(_pipeline, 'model_name', None) != model_name:
        print(f"Loading model '{model_name}' on {device} with dtype = {dtype} ...")
        try:
            _pipeline = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
                safety_checker=None,
            )
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            return
        # Move pipeline to GPU if available
        # if torch.cuda.is_available():
        #     _pipeline = _pipeline.to("cuda")
        _pipeline = _pipeline.to(device)
        _pipeline.low_vram = low_vram
        if device == "cuda":
            try:
                _pipeline.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"Failed to enable xformers memory efficient attention: {e}")
        
        _pipeline.enable_attention_slicing()
        _pipeline.enable_vae_tiling()

        # Store model_name as an attribute for reference (not a built-in property of pipeline, we add it)
        _pipeline.model_name = model_name
        print(f"Model loaded. (Model: {model_name})")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if prompt is not None:
        # Single prompt mode
        _generate_image(prompt, output_dir)
    else:
        # Interactive mode
        print("Entering interactive prompt mode. Type 'exit' or 'quit' to stop.")
        try:
            while True:
                user_input = input("Prompt> ")
                if user_input.strip().lower() in {"exit", "quit"}:
                    break
                if user_input.strip() == "":
                    continue  # skip empty prompts
                _generate_image(user_input, output_dir)
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nSession terminated by user.")
        finally:
            # Optionally, unload the model from memory if needed
            # _pipeline = None
            print("Interactive session ended.")


# Run Video Model
def run_video_model(model_name: str, prompt: str = None, output_dir: str = "."):
    """Run the specified video model. If prompt is given, generate a video for it.
    If prompt is None, enter interactive mode to accept prompts repeatedly.
    """
    global _pipeline

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"CUDA device: {props.name}, VRAM: {vram_gb:.2f} GB")

        if vram_gb <= 3:
            device = "cuda"
            dtype = torch.float32
            fp = "fp32"
            low_vram = True
        else:
            device = "cuda"
            dtype = torch.float16
            fp = "fp16"
            low_vram = False
        print("CUDA device detected. Using GPU for inference.")

    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
        fp = "fp16"
        low_vram = False
        print("MPS device detected. Using GPU for inference.")

    else:
        device = "cpu"
        dtype = torch.float32
        fp = "fp32"
        low_vram = True
        print("No CUDA device detected. Using CPU for inference (may be slow).")


    # Load the model pipeline if not already loaded or if a different model is requested
    if _pipeline is None or getattr(_pipeline, 'model_name', None) != model_name:
        print(f"Loading model '{model_name}' on {device} with dtype = {dtype} ...")
        try:
            _pipeline = DiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
                variant = fp,
                safety_checker=None,
            )
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            return
        # Move pipeline to GPU if available
        # if torch.cuda.is_available():
        #     _pipeline = _pipeline.to("cuda")
        _pipeline.scheduler = DPMSolverMultistepScheduler.from_config(_pipeline.scheduler.config)
        _pipeline = _pipeline.to(device)
        _pipeline.low_vram = low_vram

        if device == "cuda":
            try:
                _pipeline.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(f"Failed to enable xformers memory efficient attention: {e}")
                pass

        if hasattr(_pipeline, "enable_attention_slicing"):
            _pipeline.enable_attention_slicing()
        if hasattr(_pipeline, "enable_vae_tiling"):
            _pipeline.enable_vae_tiling()

        # Store model_name as an attribute for reference (not a built-in property of pipeline, we add it)
        _pipeline.model_name = model_name
        print(f"Model loaded. (Model: {model_name})")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if prompt is not None:
        # Single prompt mode
        _generate_video(prompt, output_dir)
    else:
        # Interactive mode
        print("Entering interactive prompt mode. Type 'exit' or 'quit' to stop.")
        try:
            while True:
                user_input = input("Prompt> ")
                if user_input.strip().lower() in {"exit", "quit"}:
                    break
                if user_input.strip() == "":
                    continue  # skip empty prompts
                _generate_video(user_input, output_dir)
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nSession terminated by user.")
        finally:
            # Optionally, unload the model from memory if needed
            # _pipeline = None
            print("Interactive session ended.")


# Generate Video
def _generate_video(prompt: str, output_dir: str):
    """Helper to generate a video from the global pipeline and save to output_dir."""
    global _pipeline
    if _pipeline is None:
        print("Error: No model loaded.")
        return
    print(f"Generating video for prompt: \"{prompt}\"...")

    steps = 60

    low_vram = getattr(_pipeline, 'low_vram', False)

    if not low_vram:
        steps = 200
        print("High VRAM mode: using maximum inference steps for quality.")

    result = _pipeline(
        prompt,
        num_inference_steps=steps,
        guidance_scale=7.5,
        height=512,
        width=512,
    )
    frames = result.frames

    if hasattr(result, "frames"):
        frames = result.frames
    elif isinstance(result, (list, tuple)):
        frames = result
    elif isinstance(result, dict) and "frames" in result:
        frames = result["frames"]
    else:
        print(f"Unexpected pipeline output type: {type(result)}")
        return

    if not frames:
        print("Error: pipeline returned 0 frames.")
        return

    print("Number of frames:", len(frames))
    print("Single frame shape:", np.array(frames[0]).shape)

    video_path = export_to_video(frames, os.path.join(output_dir, "result.mp4"))
    print("Video saved at:", video_path)


# Run Local LLM
def run_local_llm(
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    host: str = "0.0.0.0",
    port: int = 2513,
):
    """
    Download the specified LLM and run it as a local REST API server.

    - Supports any HF chat/instruct model that has a chat template (Qwen, Llama, etc).
    - Keeps conversation history on the server side.
    - Exposes POST /chat with JSON: { "message": "<user text>" }.
    """

    # Choose dtype based on GPU availability (float16 on GPU, float32 on CPU)
    if torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32  # safer for CPU

    print(f"Loading model '{model_name}'... (first time will download weights)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",  # use GPU if available, else CPU
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    # 2. Flask web server
    
    app = Flask(__name__)

    SYSTEM_PROMPT = "You are a helpful, honest coding assistant."
    # Chat history as a list of {role, content}
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    def build_prompt():
        """
        Build the text prompt for the current conversation using the model's
        chat template if available; otherwise fall back to a simple format.
        """
        # If the tokenizer knows how to build chat prompts, use that
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,  # add assistant turn at end
            )

        # Fallback: simple generic prompt format
        lines = []
        for msg in conversation:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                lines.append(f"SYSTEM: {content}")
            elif role == "user":
                lines.append(f"USER: {content}")
            elif role == "assistant":
                lines.append(f"ASSISTANT: {content}")
        # Add a final assistant prefix
        lines.append("ASSISTANT:")
        return "\n".join(lines)

    @app.route("/chat", methods=["POST"])
    def chat():
        nonlocal conversation
        data = request.get_json(force=True) or {}
        user_message = data.get("message", "")

        if not isinstance(user_message, str) or not user_message.strip():
            return jsonify(error="No 'message' provided or message is empty"), 400

        print("\n=== New Chat Request ===")
        print(f"User: {user_message}")

        # Append new user turn
        conversation.append({"role": "user", "content": user_message})

        # Build prompt for this turn
        prompt_text = build_prompt()
        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        print(">>> Calling model.generate()...")
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2048,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        print(">>> model.generate() finished.")

        # Only decode the new tokens (beyond the prompt)
        generated_ids = output_ids[0][input_ids.shape[-1] :]
        assistant_reply = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        print(f"LLM: {assistant_reply}\n")

        # Append assistant turn to conversation for future context
        conversation.append({"role": "assistant", "content": assistant_reply})

        return jsonify({"response": assistant_reply})

    print(f"Model loaded. Serving at http://{host}:{port}/chat")
    app.run(host=host, port=port)


# Export to Video
def export_to_video(frames, output_path="output.mp4", fps=8):
    out = []
    for frame in frames:
        f = np.array(frame)
        # f should now be (H, W, 3)
        if f.dtype != np.uint8:
            f = (255 * np.clip(f, 0, 1)).astype(np.uint8)
        out.append(f)

    imageio.mimsave(
        output_path,
        out,
        fps=fps,
        quality=8,
        macro_block_size=1,  # avoid the resizing warning
    )
    return output_path


# Generate Image
def _generate_image(prompt: str, output_dir: str):
    """Helper to generate an image from the global pipeline and save to output_dir."""
    global _pipeline
    if _pipeline is None:
        print("Error: No model loaded.")
        return
    print(f"Generating image for prompt: \"{prompt}\"...")
    # Inference: we can set some default generation parameters or expose via CLI

    steps = 50
    height = 512
    width = 512
    guidance = 7.5

    low_vram = getattr(_pipeline, 'low_vram', False)
    model_name = getattr(_pipeline, 'model_name', '')

    if low_vram:
        steps = min(steps, 30)
        print("Low VRAM mode: reducing inference steps for performance.")
        height = width = 512
        if "sd-turbo" in model_name:
            guidance = 5

    elif torch.backends.mps.is_available():
        steps = 200
        height = width = 512
        guidance = 7.5
        print("MPS device detected. Using GPU for inference.")

    # if torch.cuda.is_available():
    #     vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    #     if vram <= 3:
    #         steps = 25
    #         height = width = 512
    #         guidance = 7.5
    #         print("Low VRAM detected. Adjusting generation parameters for performance.")

    try:
        # Use the pipeline to generate image
        result = _pipeline(
            prompt, 
            num_inference_steps=steps, 
            guidance_scale=guidance,
            height=height, 
            width=width
        )
        # Diffusers pipeline returns an object with `.images`
        image = result.images[0]
    except Exception as e:
        print(f"Error during generation: {e}")
        return
    # Save image to file
    timestamp = int(time.time())
    out_path = os.path.join(output_dir, f"vllama_output_{timestamp}.png")
    try:
        image.save(out_path)
        print(f"Image saved to {out_path}")
    except Exception as e:
        print(f"Could not save image: {e}")


# Send Prompt
def send_prompt(prompt: str, output_dir: str = "."):
    """Send a prompt to an already running model (expects _pipeline to be loaded)."""
    if _pipeline is None:
        print("No model is currently running. Use `vllama run <model>` first.")
    else:
        os.makedirs(output_dir, exist_ok=True)
        _generate_image(prompt, output_dir)


# Chat with Local LLM
def chat_with_local_llm(host: str = "http://127.0.0.1", port: int = 2513):
    url = f"{host}:{port}/chat"
    print("Connected to local LLM. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            msg = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break

        if not msg:
            continue
        if msg.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break

        try:
            r = requests.post(url, json={"message": msg}, timeout=300)
            r.raise_for_status()
            data = r.json()
            reply = data.get("response", "").strip()
        except Exception as e:
            print(f"[Error] {e}")
            continue

        print(f"LLM: {reply}\n")


# Text to Speech 
def text_to_speech(text: str = None):
    """Convert the given text to speech using pyttsx3."""
    try:
        import pyttsx3
    except ImportError:
        print("pyttsx3 is not installed. Please install it to use text-to-speech functionality.")
        return

    engine = pyttsx3.init()
    
    while True:
        if text is None:
            try:
                text = input("Enter text to convert to speech (or 'exit' to quit): ")     
            except (KeyboardInterrupt, EOFError):
                print("\nExiting text-to-speech.")
                return
            if text.strip().lower() in {"exit", "quit"}:
                print("Exiting text-to-speech.")
                return
            if not text.strip():
                continue
            engine.say(text)
            engine.runAndWait()
            text = None
        else:
            if text.strip().lower() in {"exit", "quit"}:
                print("Exiting text-to-speech.")
                return
            engine.say(text)
            engine.runAndWait()
            return


# Interactive Text to Speech
def interactive_text_to_speech(text: str = None, model_id: str = "microsoft/speecht5_tts", output_dir: str = "."):
    while True:
        if text is None:
            try:
                text = input("Enter text to convert to speech (or 'exit' to quit): ")     
            except (KeyboardInterrupt, EOFError):
                print("\nExiting text-to-speech.")
                return
            if text.strip().lower() in {"exit", "quit"}:
                print("Exiting text-to-speech.")
                return
            if not text.strip():
                continue
            text_to_speech_model(text = text, model_id = model_id, output_dir = output_dir)
            text = None
        else:
            if text.strip().lower() in {"exit", "quit"}:
                print("Exiting text-to-speech.")
                return
            text_to_speech_model(text = text, model_id = model_id, output_dir = output_dir)
            return


# Split Text into Chunks
def split_text(text, max_chars=450):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) <= max_chars:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s
    if current:
        chunks.append(current.strip())

    return chunks


# Text to Speech model
def text_to_speech_model(text: str = "Hello, World!", model_id: str = "microsoft/speecht5_tts", output_dir="."):
    # Load components
    processor = SpeechT5Processor.from_pretrained(model_id)
    model = SpeechT5ForTextToSpeech.from_pretrained(model_id)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    speaker_embedding = torch.tensor(np.random.randn(1, 512), dtype=torch.float32)

    # Split into safe chunks
    chunks = split_text(text)
    print("Chunks:", len(chunks))

    all_audio = []

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")

        inputs = processor(text=chunk, return_tensors="pt")

        with torch.no_grad():
            spectrogram = model.generate_speech(
                input_ids=inputs["input_ids"],
                speaker_embeddings=speaker_embedding,
            )

        with torch.no_grad():
            audio = vocoder(spectrogram)

        audio = audio.squeeze().cpu().numpy()
        all_audio.append(audio)

    # Combine audio parts
    final_audio = np.concatenate(all_audio)

    timestamp = int(time.time())
    output_path = f"{output_dir}/output_{timestamp}.wav"

    sf.write(output_path, final_audio, 16000)
    print("Saved output.wav")
    return output_path


# Speech to Text
def speech_to_text():
    """Convert speech from microphone to text using SpeechRecognition."""
    try:
        import speech_recognition as sr
    except ImportError:
        print("SpeechRecognition is not installed. Please install it to use speech-to-text functionality.")
        return

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    while True:
        print("Listening... (say 'exit' to quit)")
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            if text.strip().lower() in {"exit", "quit"}:
                print("Exiting speech-to-text.")
                return
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            

# Transcribe from Path
def transcribe_from_path(path : str = None, model_id: str = 'openai/whisper-small', language: str = 'en'):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Whisper model
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=device,
        return_timestamps=True,
        language= language,
    )

    try:
        result = asr_pipeline(path)
        print("\nTranscription:")
        print(result["text"])
        return result["text"]
    except Exception as e:
        print(f"\nError: {e}")
        return


# Stop Session
def stop_session():
    """Stop the currently running model session, if any."""
    global _pipeline
    if _pipeline is not None:
        # (If we had a separate process, we'd terminate it here)
        _pipeline = None
        print("Model session stopped and unloaded from memory.")
    else:
        print("No model session is currently running.")


