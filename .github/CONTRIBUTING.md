# Contributing to Quant Trading System

Thank you for your interest in contributing! This project follows the KISS principle (Keep It Simple, Stupid).

## Quick Start

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `poetry run pytest`
5. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/quant-system.git
cd quant-system

# Install dependencies
poetry install --with dev

# Activate environment
poetry shell

# Run tests
pytest
```

## Code Style

- Follow PEP 8
- Use type hints
- Keep functions simple and focused
- Add docstrings for public functions
- Run `black .` and `ruff check .` before committing

## Testing

- Write tests for new features
- Maintain test coverage above 80%
- Use descriptive test names
- Mock external dependencies

## Submitting Changes

1. Ensure all tests pass
2. Update documentation if needed
3. Follow the pull request template
4. Keep commits focused and atomic
5. Write clear commit messages

## Questions?

- Check the documentation first
- Search existing issues
- Start a discussion
- Create an issue with the appropriate template

## Code of Conduct

- Be respectful and inclusive
- Focus on the code, not the person
- Help others learn and grow
- Keep discussions constructive
