# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.0] - 2025-12-05

### Added
- **VS Code Extension**: Direct integration with VS Code's native "Chat with AI" interface to chat with local LLMs.
- **Local LLM Server**: `vllama run_llm` command to run local LLMs as REST API servers.
- **CLI Chat**: `vllama chat_llm` command for interactive terminal chat with local LLMs.
- **Video Generation**: `vllama run_video` command to generate videos from text prompts (local and Kaggle).
- **Speech Capabilities**: `vllama tts` (Text-to-Speech) and `vllama stt` (Speech-to-Text) commands.
- **Model Management**: `vllama list` and `vllama uninstall` commands.
- **Open Source Files**: Added CODE_OF_CONDUCT.md, CONTRIBUTING.md updates, and GitHub issue templates.

### Changed
- **License**: Changed from GNU GPL v3.0 to Apache License 2.0.
- **Documentation**: Comprehensive README overhaul with new workflows and command references.
- **Project Structure**: Updated `pyproject.toml` and `setup.py` with improved metadata.

## [0.8.1] - 2025-12-03
- Implemented basic Speech to Text

## [0.8.0] - 2025-12-03
- Implemented basic Text-to-speech using pyttsx3

## [0.7.2] - 2025-12-02
- Added Docker Files

## [0.7.1] - 2025-12-02
- Displaying size of model while listing

## [0.7.0] - 2025-12-01
- Implemented local LLM and chat options

## [0.6.2] - 2025-11-29
- Fix GPU issue on Macs with MPS

## [0.6.1] - 2025-11-28
- Implemented listing the downloaded models

## [0.6.0] - 2025-11-28
- Implemented run video model in core

## [0.5.1] - 2025-11-28
- Implemented version and uninstall model

## [0.5.0] - 2025-11-25
- Implemented Run Video Kaggle

## [0.4.0] - 2025-11-24
- Implemented AutoML training

## [0.3.3] - 2025-11-24
- Implemented saving transformations

## [0.3.2] - 2025-11-21
- Documentations

## [0.3.1] - 2025-11-21
- Fixed typos and Keyboard interrupt

## [0.3.0] - 2025-11-21
- Implemented data processing

## [0.2.0] - 2025-11-20
- Implemented Kaggle CLI to run the model remotely on Kaggle GPU

## [0.1.3] - 2025-11-20
- Fixed GPU and low VRAM issue

## [0.1.2] - 2025-11-19
- Fixed run issue

## [0.1.1] - 2025-11-19
- Fixed errors

## [0.1.0] - 2025-11-19
- Basic implementation
