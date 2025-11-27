import os, time
import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import scan_cache_dir


_pipeline = None


_SUPPORTED_MODELS = {
    "stabilityai/sd-turbo": "StabilityAI SD-Turbo (distilled Stable Diffusion 2.1, fast text-to-image model)",
    "damo-vilab/text-to-video-ms-1.7b": "DAMO VILAB Text-to-Video MS 1.7B (text-to-video generation model)",
}


# List Models
def list_models():
    """List available models for installation."""
    print("Supported models:")
    for name, desc in _SUPPORTED_MODELS.items():
        print(f"- {name}: {desc}")


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
                pass
        
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
    if "sd-turbo" in model_name:
        steps = 40
        guidance = 5.0

    if low_vram:
        steps = min(steps, 30)
        print("Low VRAM mode: reducing inference steps for performance.")
        height = width = 512
        if "sd-turbo" in model_name:
            guidance = 5

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


