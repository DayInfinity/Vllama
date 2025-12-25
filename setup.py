import setuptools
from setuptools import find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vllama",
    version="1.4.0",
    author="Gopu Manvith",
    author_email="manvithgopu1394@gmail.com",
    description="Comprehensive CLI tool and VS Code extension for vision models, AutoML, and local LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DayInfinity/Vllama",
    project_urls={
        "Bug Tracker": "https://github.com/DayInfinity/Vllama/issues",
        "Documentation": "https://github.com/DayInfinity/Vllama#readme",
        "Source Code": "https://github.com/DayInfinity/Vllama",
    },
    license="Apache-2.0",
    packages=find_packages(),
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
        "soundfile",
        "regex",
        "ultralytics",
        "opencv-python",
        "requests",
        "lap",
    ],
    entry_points={
        "console_scripts": [
            "vllama = vllama.cli:main"
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)