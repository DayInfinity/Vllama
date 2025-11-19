import os
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
