import setuptools

setuptools.setup(
    name="vllama",
    version="0.8.1",
    author="Gopu Manvith",
    description="CLI tool to download and run vision models (like Stable Diffusion) on local and cloud GPUs",
    packages=["vllama"],
    python_requires=">=3.8",
    install_requires=[
        "argparse",
        "torch>=2.0.0",
        "diffusers>=0.20.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "protobuf>=3.20.0",
        "kaggle>=1.5.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.2.0",
        "xgboost>=1.7.6",
        "lightgbm>=3.3.5",
        "catboost>=1.2.1",
        "joblib>=1.2.0",
        "imageio>=2.31.0",
        "build==1.3.0",
        "twine",
        "flask",
        "pyttsx3",
        "SpeechRecognition",
        "pyaudio",
    ],
    entry_points={
        "console_scripts": [
            "vllama = vllama.cli:main"
        ]
    }
)