# Contributing to Vllama

Thank you for your interest in contributing to Vllama! We appreciate your help in making this tool better.

## Code of Conduct

This project and everyone participating in it is governed by the [Vllama Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1.  **Fork the Repository**: Click the "Fork" button on the GitHub page.
2.  **Clone Locally**:
    ```bash
    git clone https://github.com/DayInfinity/Vllama.git
    cd Vllama
    ```
3.  **Set Up Environment**:
    It is recommended to use a virtual environment:
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```
4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

## Development Guidelines

*   **Code Style**: Keep code clean and readable. We use standard Python conventions (PEP 8).
*   **Documentation**: Add comments to explain complex logic. Update the README if you change how the tool is used.
*   **Testing**: If possible, add tests for your new features to ensure they work correctly.
*   **Commit Messages**: Write clear and descriptive commit messages.

## Submitting Changes

1.  Create a new branch for your feature: `git checkout -b feature/my-new-feature`
2.  Commit your changes: `git commit -m "Add some feature"`
3.  Push to your fork: `git push origin feature/my-new-feature`
4.  Open a Pull Request on the main Vllama repository.
    - Fill out the Pull Request template.
    - Link any relevant issues.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub using the provided templates.
