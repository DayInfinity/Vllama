import argparse
import sys
from vllama import core, remote
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def main():
    parser = argparse.ArgumentParser(prog="vllama", description="vllama CLI - manage and run vision models locally or on the cloud GPUs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    login_parser = subparsers.add_parser("login", help="Login to a GPU service (e.g., Kaggle, Colab)")
    login_parser.add_argument("--service", choices=["kaggle", "colab"], required=True, help="Service to login (currently supports kaggle or colab)")
    login_parser.add_argument("--username", help="Kaggle username(if not using default credentials file)")
    login_parser.add_argument("--key", help="kaggle API key (if not using default credentials file)")


    init_parser = subparsers.add_parser("init", help="Initialize a GPU session on the specifies service")
    init_parser.add_argument("gpu", choices=["gpu"], help="Keyword 'gpu' (to initialize a GPU runtime)")
    init_parser.add_argument("--service", choices=["kaggle", "colab"], required=True, help="Service to initialize the GPU on")

    show_parser = subparsers.add_parser("show", help="Show availble models")
    show_parser.add_argument("models", nargs='?', const="models", help="(Usage: vllama show models)")

    install_parser = subparsers.add_parser("install", help="Install/downoad a model")
    install_parser.add_argument("model", help="Name of the model to install(eg.,stabilityai/sd-turbo)")

    run_parser = subparsers.add_parser("run", help="Run a model to genrate outputs")
    run_parser.add_argument("model", help="Name of the model to run (must be installed or accessible)")
    run_parser.add_argument("--prompt", "-p", help="Text prompt for generation. If not provided, entersinteractive mode.")
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
        core.install_model(model_name)

    elif args.command == "run":
        model_name = args.model
        prompt = args.prompt
        output_dir = args.output_dir or "."
        service = args.service
        if service and service.lower() == "kaggle":
            if not prompt:
                try:
                    prompt = input("Enter a prompt for image generation: ")
                except eyboardInterrupt:
                    print("\nGeneration cancelled by user.")
                    sys.exit(0)
                if not prompt:
                    print("No prompt provided. Exiting.")
                    sys.exit(0)
            remote.run_kaggle(model_name, prompt, output_dir)
        else:
            core.run_model(model_name, prompt, output_dir)

    elif args.command == "post":
        prompt = args.prompt
        output_dir = args.output_dir or "."
        core.send_prompt(prompt, output_dir)

    elif args.command == "stop":
        core.stop_session()

    elif args.command == "logout":
        remote.logout()

    else:
        parser.print_help()