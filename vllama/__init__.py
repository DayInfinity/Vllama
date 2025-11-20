"""vllama - CLI tool for running vision models locally and on cloud GPUs."""

__version__ = "0.2.0"
__author__ = "Gopu Manvith"
__email__ = "manvithgopu1394@gmail.com"

from vllama import core, remote, cli

__all__ = ["core", "remote", "cli", "__version__"]
