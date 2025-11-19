import os, time
import torch
from diffusers import StableDiffusionPipeline


_pipeline = None


_SUPPORTED_MODELS = {
    "stabilityai/sd-turbo": "StabilityAI SD-Turbo (distilled Stable Diffusion 2.1, fast text-to-image model)",

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


# Run Model
def run_model(model_name: str, prompt: str = None, output_dir: str = "."):
    """
    Run the specified model. If prompt is given, generate an image for it.
    If prompt is None, enter interactive mode to accept prompts repeatedly.
    """
    global _pipeline

    # Load the model pipeline if not already loaded or if a different model is requested
    if _pipeline is None or getattr(_pipeline, 'model_name', None) != model_name:
        print(f"Loading model '{model_name}'...")
        try:
            _pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            return
        # Move pipeline to GPU if available
        if torch.cuda.is_available():
            _pipeline = _pipeline.to("cuda")
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
    try:
        # Use the pipeline to generate image
        result = _pipeline(prompt, num_inference_steps=50, guidance_scale=7.5)
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


