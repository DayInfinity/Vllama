import setuptools

setuptools.setup(
    name="vllama",
    version="0.3.1",
    author="Gopu Manvith",
    description="CLI tool to download and run vision models (like Stable Diffusion) on local and cloud GPUs",
    packages=["vllama"],
    python_requires=">=3.8",
    install_requires=[
        "argparse",
        "diffusers>=0.18.0",
        "transformers>=4.0.0",
        "accelerate>=0.20.0",
        "torch",
        "kaggle",
        "protobuf==3.20.3"
    ],
    entry_points={
        "console_scripts": [
            "vllama = vllama.cli:main"
        ]
    }
)