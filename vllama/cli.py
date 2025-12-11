import argparse
import sys
from vllama import core, model_training, remote, preprocess
import os
from importlib.metadata import version as pkg_version

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def main():
    parser = argparse.ArgumentParser(prog="vllama", description="vllama CLI - manage and run vision models locally or on the cloud GPUs")
    
    parser.add_argument("--version", "-v", action = "version", version = f"%(prog)s {pkg_version('vllama')}")
    
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

    list_parser = subparsers.add_parser("list", help="List all installed models")
    list_parser.add_argument("models", nargs='?', const="models", help="(Usage: vllama list models)")

    install_parser = subparsers.add_parser("install", help="Install/download a model")
    install_parser.add_argument("model", help="Name of the model to install (eg., stabilityai/sd-turbo)")

    uninstall_parser = subparsers.add_parser("uninstall", help="Uninstall/remove a model")
    uninstall_parser.add_argument("model", help="Name of the model to uninstall (eg., stabilityai/sd-turbo)")

    run_parser = subparsers.add_parser("run", help="Run a model to generate outputs")
    run_parser.add_argument("model", help="Name of the model to run (must be installed or accessible)")
    run_parser.add_argument("--prompt", "-p", help="Text prompt for generation. If not provided, enters interactive mode.")
    run_parser.add_argument("--service", "-s", type=str, choices = ['kaggle'], help="Offload execution to a remote service (eg., 'kaggle' for kaggle notebooks)")
    run_parser.add_argument("--output_dir", "-o", help="Directory to save outputs (default: current directory)")

    run_video_parser = subparsers.add_parser("run_video", help="To generate video using prompt")
    run_video_parser.add_argument("model", help="Name of the model to run (must be installed or accessible)")
    run_video_parser.add_argument("--prompt", "-p", help="Text prompt for generation. If not provided, enters interactive mode.")
    run_video_parser.add_argument("--service", "-s", type=str, choices = ['kaggle'], help="Offload execution to a remote service (eg., 'kaggle' for kaggle notebooks)")
    run_video_parser.add_argument("--output_dir", "-o", help="Directory to save outputs (default: current directory)")


    run_llm_parser = subparsers.add_parser("run_llm", help="Run a local LLM model to generate text outputs")
    run_llm_parser.add_argument("model", help="Name of the LLM model to run locally (must be installed or accessible)")

    chat_llm_parser = subparsers.add_parser("chat_llm", help="Chat with a local LLM model interactively")
    
    tts_parser = subparsers.add_parser("tts", help="Convert text to speech using local TTS engine")
    tts_parser.add_argument("--text", help="Text to convert to speech")
    tts_parser.add_argument("--model", help="TTS model to use")
    tts_parser.add_argument("--output_dir", "-o", help="Directory to save output (if applicable)")

    stt_parser = subparsers.add_parser("stt", help="Convert speech to text using local STT engine")
    stt_parser.add_argument("--path", help="Path to the audio file for transcription")
    stt_parser.add_argument("--model", help="STT model to use")
    stt_parser.add_argument("--language", help="Language of the audio for better transcription")
    
    post_parser = subparsers.add_parser("post", help="Send a prompt to a running model session")
    post_parser.add_argument("prompt", help="Prompt text to send to the model")
    post_parser.add_argument("--output_dir", "-o", help="Directory to save output (if applicable)")

    stop_parser = subparsers.add_parser("stop", help="Stop the running model session")

    logout_parser = subparsers.add_parser("logout", help="Logout from the current service")

    data_parser = subparsers.add_parser("data", help="Dataset cleaning and processing")
    data_parser.add_argument("--path", help="Path to the dataset")
    data_parser.add_argument("--target", help="Target Column")
    data_parser.add_argument("--test_size", "-t", help="Test-train split")
    data_parser.add_argument("--output_dir", "-o", help="Directory to save output (if applicable)")

    train_parser = subparsers.add_parser("train", help="AutoML model training on processed data")
    train_parser.add_argument("--path", "-p", help="Path to the datasets folder, woth train_data.csv and test_data.csv")
    train_parser.add_argument("--target", "-t", help="Target Column")

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
        core.show_models()

    elif args.command == "list":
        core.list_downloads()

    elif args.command == "install":
        model_name = args.model
        core.install_model(model_name)
        
    elif args.command == "uninstall":
        model_name = args.model
        core.uninstall_model(model_name)

    elif args.command == "run":
        model_name = args.model
        prompt = args.prompt
        output_dir = args.output_dir or "."
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
            remote.run_kaggle(model_name, prompt, output_dir)
        else:
            core.run_model(model_name, prompt, output_dir)

    elif args.command == "run_video":
        model_name = args.model
        prompt = args.prompt
        output_dir = args.output_dir or "."
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
            remote.run_video_kaggle(model_name, prompt, output_dir)
        else:
            core.run_video_model(model_name, prompt, output_dir)

    elif args.command == "run_llm":
        model_name = args.model
        core.run_local_llm(model_name)

    elif args.command == "chat_llm":
        core.chat_with_local_llm()
    
    elif args.command == "tts":
        text = args.text
        model_id = args.model or "microsoft/speecht5_tts"
        output_dir = args.output_dir or "."
        core.interactive_text_to_speech(text = text, model_id = model_id, output_dir = output_dir)

    elif args.command == "stt":
        path = args.path
        model_id = args.model or "openai/whisper-small"
        language = args.language or "en"
        core.transcribe_from_path(path=path, model_id=model_id, language=language)

    elif args.command == "post":
        prompt = args.prompt
        output_dir = args.output_dir or "."
        core.send_prompt(prompt, output_dir)

    elif args.command == "stop":
        core.stop_session()

    elif args.command == "logout":
        remote.logout()

    elif args.command == "data":
        path = args.path
        target = args.target
        test_size = args.test_size or 0.2
        output_dir = args.output_dir or "."
        preprocess.autonomous_data_preprocessing(dataset_path = path, test_size=test_size,target_column = target, output_dir = output_dir)

    elif args.command == "train":
        path = args.path
        target_column = args.target
        model_training.run_automl_training(data_dir= path, target_column=target_column)

    else:
        parser.print_help()
