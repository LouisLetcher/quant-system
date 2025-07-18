# Security Policy

## Supported Versions

We currently support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via:

1. **Email**: Send details to the repository maintainers
2. **GitHub Security Advisories**: Use the "Security" tab to report privately

## What to Include

When reporting a vulnerability, please include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if known)

## Response Timeline

- **Initial response**: Within 48 hours
- **Status update**: Within 7 days
- **Fix timeline**: Depends on severity

## Security Best Practices

### For Contributors
- Never commit API keys or secrets
- Use environment variables for sensitive data
- Follow secure coding practices
- Run security scans before submitting PRs

### For Users
- Keep dependencies updated
- Use secure API key storage
- Monitor for security advisories
- Report suspicious behavior

## Dependencies

We use automated tools to monitor dependencies:

- **Dependabot**: Automatic dependency updates
- **Safety**: Python security vulnerability scanning
- **Bandit**: Static security analysis

## Disclosure Policy

- We follow responsible disclosure
- Security issues are prioritized
- Public disclosure after fix is available
- Credit given to security researchers
