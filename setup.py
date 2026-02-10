import setuptools
from setuptools import find_packages
from setuptools.command.install import install
import platform
import subprocess
import sys

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

class CustomInstallCommand(install):
    """Custom install command that installs system dependencies on macOS."""
    
    def run(self):
        # Check if system is macOS
        if platform.system() == "Darwin":
            try:
                print("\n" + "="*60)
                print("macOS detected. Installing portaudio via Homebrew...")
                print("="*60)
                
                # Check if brew is installed
                subprocess.run(["brew", "--version"], check=True, capture_output=True)
                
                # Install portaudio
                subprocess.run(["brew", "install", "portaudio"], check=True)
                print("portaudio installed successfully!")
                print("="*60 + "\n")
            except FileNotFoundError:
                print("\n" + "!"*60)
                print("WARNING: Homebrew is not installed or not in PATH.")
                print("Please install Homebrew from https://brew.sh/")
                print("Then run: brew install portaudio")
                print("!"*60 + "\n")
            except subprocess.CalledProcessError as e:
                print(f"\nWARNING: Failed to install portaudio: {e}")
                print("You may need to install it manually with: brew install portaudio\n")
        
        # Run the standard install process
        super().run()

setuptools.setup(
    name="vllama",
    version="1.9.0",
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
    cmdclass={
        'install': CustomInstallCommand,
    },
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
        "imageio-ffmpeg",
        "open3d",
        "trimesh",
        "pyrender",
        "plyfile",
        "imageio_ffmpeg",
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