import os, time
import json
import subprocess
import tempfile
import shutil


# Login to Remote Service
def login(service: str, username: str = None, key: str = None):
    """Authenticate with the given service (supports 'kaggle')."""
    service = service.lower()
    if service == "kaggle":
        # Kaggle expects a ~/.kaggle/kaggle.json file with {"username": "...", "key": "..."}
        credentials_path = os.path.expanduser("~/.kaggle/kaggle.json")
        if username and key:
            # Use provided credentials
            os.makedirs(os.path.dirname(credentials_path), exist_ok=True)
            creds = {"username": username, "key": key}
            with open(credentials_path, "w") as f:
                json.dump(creds, f)
            os.chmod(credentials_path, 0o600)  # secure the file
            print("Kaggle credentials saved.")
        else:
            # If not provided, check if file already exists or prompt user
            if os.path.exists(credentials_path):
                print("Using existing Kaggle API credentials.")
            else:
                print("Kaggle API token not found. Please provide --username and --key, or place your kaggle.json in ~/.kaggle/")
                return
        # Authenticate using Kaggle API (this will read the file)
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            print("Kaggle login successful.")
        except Exception as e:
            print(f"Failed to authenticate with Kaggle: {e}")
    elif service == "colab":
        # Colab doesn't have a direct API. Perhaps we could instruct the user on how to connect.
        print("Google Colab login: Colab uses Google accounts. There's no direct CLI login; you will authenticate via browser when using Colab.")
    else:
        print(f"Service '{service}' not supported for login.")


# Initializing GPU Session
def init_gpu(service: str):
    """Initialize a GPU session on the given service (e.g., start Kaggle kernel with GPU)."""
    service = service.lower()
    if service == "kaggle":
        # We will create a minimal Kaggle kernel and push it to run.
        # Define a simple notebook (or script) that will stay alive.
        kernel_dir = tempfile.mkdtemp(prefix="vllama_kaggle_")
        script_path = os.path.join(kernel_dir, "vllama_kernel.py")
        metadata_path = os.path.join(kernel_dir, "kernel-metadata.json")
        # Write a simple script that installs diffusers and then waits for commands (placeholder)
        with open(script_path, "w") as f:
            f.write("import time\n")
            f.write("import subprocess\n")
            f.write("# Install required libraries\n")
            f.write("subprocess.run(['pip', 'install', 'diffusers', 'transformers', 'accelerate', '-q'])\n")
            f.write("subprocess.run(['pip', 'install', 'torch', '--no-cache-dir', '-q'])\n")
            f.write("print('vllama: Kernel is set up with required libraries and GPU.')\n")
            f.write("while True:\n")
            f.write("    time.sleep(60)\n")  # keep alive (does nothing, just waits)
        # Write kernel metadata with GPU enabled
        metadata = {
            "id": "<your-kaggle-username>/vllama-session",  # unique kernel slug
            "title": "vllama session",
            "code_file": os.path.basename(script_path),
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": []
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        # Use Kaggle CLI to push the kernel
        cmd = ["kaggle", "kernels", "push", "-p", kernel_dir]
        try:
            subprocess.run(cmd, check=True)
            print("Kaggle GPU kernel initiated. Check your Kaggle account for 'vllama session'.")
            print("The kernel will run and wait (idle). You can now use `vllama install` and `vllama run` commands which will execute on this kernel.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to push Kaggle kernel: {e}")
        finally:
            # Cleanup local temp files if desired
            shutil.rmtree(kernel_dir)
    elif service == "colab":
        print("Initializing Colab GPU session is not automated. Please open a Colab notebook with GPU manually.")
    else:
        print(f"Service '{service}' not supported for init.")


# Run Kaggle
def run_kaggle(model_name, prompt, output_dir):
    credentials_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(credentials_path) and not (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    ):
        print("Error: Kaggle API credentials not found. Please set up you kaggle API token in ~/.kaggle/kaggle.json.")
        return 
    
    try:
        subprocess.run(["kaggle", "--version"], stdout=subprocess.DEVNULL, check=True)
    except FileNotFoundError:
        print("Error: Kaggle CLI is not installed. Please install it with 'pip install kaggle'.")
        return

    usename = None
    if os.path.exists(credentials_path):
        try:
            with open(credentials_path, "r") as f:
                creds = json.load(f)
                username = creds.get("username")
        except Exception:
            username = None

    if not username:
        username = os.environ.get("KAGGLE_USERNAME")
    if not username:
        print("Error: Could not determine Kaggle username from credentials.")
        return

    # 2. Prepare a temporary directory with the kernel script and metadata
    kernel_dir = tempfile.mkdtemp(prefix="vllama_kaggle_")
    try:
        # Write the Kaggle kernel script that installs dependencies and runs the model
        script_path = os.path.join(kernel_dir, "vllama_kernel.py")
        model_str = json.dumps(model_name)    # safely quote the model string
        prompt_str = json.dumps(prompt)     # safely quote the prompt string
        script_code = f"""
import subprocess
# Install required packages inside Kaggle (quietly, without cache to speed up start)
subprocess.run(
    ["pip", "uninstall", "-y", "jax", "jaxlib", "flax"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
subprocess.run(['pip', 'install', '--no-cache-dir', 'diffusers[torch]==0.20.2', 
                'transformers==4.33.0', 'accelerate==0.22.0', 'xformers==0.0.20','protobuf==3.20.3', 'huggingface-hub==0.25.2' , '--quiet'])
from diffusers import StableDiffusionPipeline
import torch
pipe = StableDiffusionPipeline.from_pretrained({model_str}, torch_dtype=torch.float16)
pipe.to('cuda')
prompt = {prompt_str}
print(f"Generating image for prompt: {{prompt}}")
image = pipe(prompt).images[0]
image.save('result.png')
print("Image generation done. Output saved to result.png")
"""
        with open(script_path, 'w') as f:
            f.write(script_code.strip() + "\n")

        # Write kernel-metadata.json for Kaggle
        kernel_slug = "vllama-" + model_name.replace('/', '-')
        # Sanitize slug to meet Kaggle requirements (alphanumeric and hyphens)
        kernel_slug = "".join(ch if ch.isalnum() or ch == '-' else '-' for ch in kernel_slug.lower())
        if len(kernel_slug) > 50:  # slug length limit (if any)
            kernel_slug = kernel_slug[:50]
        title = "vllama " + model_name.replace('/', ' ')
        metadata = {
            "id": f"{username}/{kernel_slug}",
            "title": title,
            "code_file": os.path.basename(script_path),
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": []
        }
        meta_path = os.path.join(kernel_dir, "kernel-metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # 3. Push the kernel to Kaggle and trigger execution
        print(f"Pushing Kaggle kernel (model: {model_name})...")
        result = subprocess.run(["kaggle", "kernels", "push", "-p", kernel_dir], capture_output=True, text=True)
        if result.returncode != 0:
            print("Failed to push Kaggle kernel. Error output:")
            print(result.stderr or result.stdout)
            return
        print("Kernel pushed successfully. Kaggle is running the kernel...")

        # 4. Poll Kaggle for kernel status until it finishes
        kernel_ref = f"{username}/{kernel_slug}"
        start_time = time.time()
        timestamp = int(time.time())
        while True:
            time.sleep(5)  # wait 5 seconds between status checks
            status_res = subprocess.run(["kaggle", "kernels", "status", kernel_ref], capture_output=True, text=True)
            status_text = (status_res.stdout or "") + (status_res.stderr or "")
            status_lower = status_text.lower()
            if "complete" in status_lower:
                print("Kaggle kernel execution completed.")
                break
            if "error" in status_lower or "failed" in status_lower:
                print("Kaggle kernel execution failed. Please check the Kaggle notebook for errors.")
                return
            if time.time() - start_time > 300:  # timeout after 5 minutes
                print("Timed out waiting for Kaggle kernel to complete.")
                return

        # 5. Download the generated image from Kaggle
        os.makedirs(output_dir, exist_ok=True)
        print(f"Downloading output to {output_dir}...")
        out_res = subprocess.run(["kaggle", "kernels", "output", kernel_ref, "-p", output_dir],
                                 capture_output=True, text=True)
        if out_res.returncode != 0:
            print("Error downloading output from Kaggle:")
            print(out_res.stderr or out_res.stdout)
        else:
            image_path = os.path.join(output_dir, f"vllama_kaggle_{timestamp}.png")
            if os.path.exists(image_path):
                print(f"Image successfully downloaded to {image_path}")
            else:
                print("Image output downloaded to the specified directory.")
    finally:
        # Clean up the temporary kernel files
        shutil.rmtree(kernel_dir, ignore_errors=True)    


# Logout
def logout():
    """Logout from the current service (e.g., clear Kaggle credentials)."""
    # For Kaggle, we can delete the kaggle.json file or just inform the user to remove it.
    creds_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(creds_path):
        try:
            os.remove(creds_path)
            print("Kaggle credentials removed (logged out).")
        except Exception as e:
            print(f"Could not remove credentials file: {e}")
    else:
        print("No credentials found to remove.")
