# Security Policy

## Supported Versions

Use this section to tell people about which versions of your project are currently being supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| 0.2.x   | :x:                |
| < 0.2   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability within Vllama, please follow these steps:

1.  **Do NOT open a public issue.**
2.  Email the details to the maintainer at [manvithgopu1394@gmail.com](mailto:manvithgopu1394@gmail.com).
3.  Include as much information as possible:
    *   Type of vulnerability (e.g., command injection, cross-site scripting).
    *   Steps to reproduce the issue.
    *   Affected versions.
    *   Potential impact.

We will acknowledge your report within 48 hours and provide an estimated timeline for a fix.

## Credential Management Best Practices

Vllama interacts with cloud services like Kaggle. To keep your account secure:

*   **Never commit your `kaggle.json` or API keys to version control (git).**
*   Use the `vllama login` command which stores credentials securely in your home directory or uses the default Kaggle configuration path.
*   If using environment variables, ensure they are not logged or exposed in shared environments.
*   Regularly rotate your API keys if you suspect they have been compromised.

## Remote Execution Security

When using the `remote` features (e.g., running on Kaggle):

*   Be aware that code is executed on a remote server.
*   Ensure your prompts do not contain sensitive information.
*   Vllama sanitizes inputs to prevent command injection, but always practice caution when interacting with remote execution environments.
