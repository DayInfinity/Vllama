# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.10.0] - 2026-02-10

### Added
- **Documentation**: Full documentation overhaul across README, CHANGELOG, and SECURITY.

### Changed
- Bumped version to 1.10.0.
- README: Updated version badge to 1.10.0; added command reference for `detect_image`, `detect_video`, `image3d`, `video3d`, `view3d`, and `translate`.
- README: Updated Key Features list to include object detection, image/video-to-3D, 3D viewer, and translation.
- README: Updated Recent Updates section to reflect 1.10.0 and 1.9.0; corrected roadmap (video generation supported; clarified local 3D as upcoming).
- SECURITY.md: Updated supported versions table to 1.10.x and 1.9.x.

### Fixed
- CLI: Typo in `train` command help text ("woth" → "with").

## [1.9.0] - 2026-02-10

### Changed
- Updated dependencies and PortAudio support on macOS.

## [1.8.0] - 2026-01-24

### Added
- MOV support for video-to-3D pipeline.

## [1.7.0] - 2026-01-24

### Changed
- Updated 3D viewer to display Pi3 models.

## [1.6.2] - 2026-01-22

### Fixed
- Added sleep time for dataset status polling in video-to-3D.

## [1.6.1] - 2026-01-22

### Added
- Basic local video-to-3D generation (in addition to Kaggle).

## [1.6.0] - 2026-01-20

### Added
- Video-to-3D generation (e.g. `vllama video3d` with Kaggle).

## [1.5.0] - 2025-12-26

### Added
- 3D model viewer (`vllama view3d`) for PLY, GLB, OBJ, STL, FBX.

## [1.4.0] - 2025-12-25

### Added
- Image-to-3D via Kaggle (`vllama image3d --service kaggle`).

## [1.3.0] - 2025-12-19

### Added
- Video object detection (`vllama detect_video`) using YOLO.

## [1.2.1] - 2025-12-19

### Fixed
- Class type declaration compatibility with older Python versions.

## [1.2.0] - 2025-12-13

### Added
- Image object detection (`vllama detect_image`) using YOLO.

## [1.1.0] - 2025-12-13

### Added
- Translation models (`vllama translate`) using NLLB.

## [1.0.3] - 2025-12-12

### Fixed
- List downloads issue.

## [1.0.2] - 2025-12-11

### Changed
- STT (speech-to-text) updated to use Whisper model.

## [1.0.1] - 2025-12-10

### Changed
- Text-to-speech updated to use Microsoft TTS model (SpeechT5).

## [1.0.0] - 2025-12-06

### Changed
- Updated project ownership to the organization `DayInfinity`.
- Updated repository links to reflect the new ownership.
- Bumped version to 1.0.0.

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
