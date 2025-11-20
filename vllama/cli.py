import argparse
import sys
import re
import os
from vllama import core, remote

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def validate_model_name(model_name: str) -> bool:
    """Validate model name to prevent path traversal and injection attacks."""
    # Model names should follow format: org/model-name or model-name
    # Allow alphanumeric, hyphens, underscores, dots, and single forward slash
    pattern = r'^[a-zA-Z0-9][a-zA-Z0-9._-]*(/[a-zA-Z0-9][a-zA-Z0-9._-]*)?$'
    if not re.match(pattern, model_name):
        return False
    # Prevent path traversal attempts
    if '..' in model_name or model_name.startswith('/') or model_name.endswith('/'):
        return False
    return True

def validate_output_dir(output_dir: str) -> bool:
    """Validate output directory path to prevent path traversal."""
    # Resolve to absolute path and check it doesn't escape intended boundaries
    try:
        abs_path = os.path.abspath(output_dir)
        # Check for suspicious patterns
        if '..' in output_dir:
            return False
        return True
    except (ValueError, OSError):
        return False

def sanitize_prompt(prompt: str) -> str:
    """Sanitize user prompt to prevent injection attacks."""
    # Remove any null bytes and control characters except newlines/tabs
    sanitized = ''.join(char for char in prompt if char.isprintable() or char in '\n\t')
    return sanitized.strip()

def main():
    parser = argparse.ArgumentParser(prog="vllama", description="vllama CLI - manage and run vision models locally or on the cloud GPUs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    login_parser = subparsers.add_parser("login", help="Login to a GPU service (e.g., Kaggle, Colab)")
    login_parser.add_argument("--service", choices=["kaggle", "colab"], required=True, help="Service to login (currently supports kaggle or colab)")
    login_parser.add_argument("--username", help="Kaggle username(if not using default credentials file)")
    login_parser.add_argument("--key", help="kaggle API key (if not using default credentials file)")


    init_parser = subparsers.add_parser("init", help="Initialize a GPU session on the specified service")
    init_parser.add_argument("gpu", choices=["gpu"], help="Keyword 'gpu' (to initialize a GPU runtime)")
    init_parser.add_argument("--service", choices=["kaggle", "colab"], required=True, help="Service to initialize the GPU on")

    show_parser = subparsers.add_parser("show", help="Show available models")
    show_parser.add_argument("models", nargs='?', const="models", help="(Usage: vllama show models)")

    install_parser = subparsers.add_parser("install", help="Install/download a model")
    install_parser.add_argument("model", help="Name of the model to install(eg.,stabilityai/sd-turbo)")

    run_parser = subparsers.add_parser("run", help="Run a model to generate outputs")
    run_parser.add_argument("model", help="Name of the model to run (must be installed or accessible)")
    run_parser.add_argument("--prompt", "-p", help="Text prompt for generation. If not provided, enters interactive mode.")
    run_parser.add_argument("--service", "-s", type=str, choices = ['kaggle'], help="Offload execution to a remote service (eg., 'kaggle' for kaggle notebooks)")
    run_parser.add_argument("--output_dir", "-o", help="Directory to save outputs (default: current directory)")

    post_parser = subparsers.add_parser("post", help="Send a prompt to a running model session")
    post_parser.add_argument("prompt", help="Prompt text to send to the model")
    post_parser.add_argument("--output_dir", "-o", help="Directory to save output (if applicable)")

    stop_parser = subparsers.add_parser("stop", help="Stop the running model session")

    logout_parser = subparsers.add_parser("logout", help="Logout from the current service")

    args = parser.parse_args()

    if args.command == "login":
        service = args.service
        username = args.username
        key = args.key
        remote.login(service, username, key)

    elif args.command == "init":
        service = args.service
        remote.init_gpu(service)

    elif args.command == "show":
        core.list_models()

    elif args.command == "install":
        model_name = args.model
        if not validate_model_name(model_name):
            print(f"Error: Invalid model name '{model_name}'. Model names should follow the format 'organization/model-name'.")
            sys.exit(1)
        core.install_model(model_name)

    elif args.command == "run":
        model_name = args.model
        if not validate_model_name(model_name):
            print(f"Error: Invalid model name '{model_name}'. Model names should follow the format 'organization/model-name'.")
            sys.exit(1)
        
        prompt = args.prompt
        output_dir = args.output_dir or "."
        
        if not validate_output_dir(output_dir):
            print(f"Error: Invalid output directory '{output_dir}'. Please use a valid path.")
            sys.exit(1)
        
        service = args.service
        if service and service.lower() == "kaggle":
            if not prompt:
                try:
                    prompt = input("Enter a prompt for image generation: ")
                except KeyboardInterrupt:
                    print("\nGeneration cancelled by user.")
                    sys.exit(0)
                if not prompt:
                    print("No prompt provided. Exiting.")
                    sys.exit(0)
            # Sanitize prompt before passing to remote service
            prompt = sanitize_prompt(prompt)
            if not prompt:
                print("Error: Prompt contains only invalid characters.")
                sys.exit(1)
            remote.run_kaggle(model_name, prompt, output_dir)
        else:
            # Sanitize prompt for local execution too
            if prompt:
                prompt = sanitize_prompt(prompt)
            core.run_model(model_name, prompt, output_dir)

    elif args.command == "post":
        prompt = args.prompt
        # Sanitize prompt
        prompt = sanitize_prompt(prompt)
        if not prompt:
            print("Error: Prompt contains only invalid characters.")
            sys.exit(1)
        
        output_dir = args.output_dir or "."
        if not validate_output_dir(output_dir):
            print(f"Error: Invalid output directory '{output_dir}'. Please use a valid path.")
            sys.exit(1)
        
        core.send_prompt(prompt, output_dir)

    elif args.command == "stop":
        core.stop_session()

    elif args.command == "logout":
        remote.logout()

    else:
        parser.print_help()
