# Contributing to Quant Trading System

## 📁 Repository Structure

This repository follows Python best practices for a production quantitative analysis system:

```
quant-system/
├── src/                    # Source code (importable package)
│   ├── ai/                # AI recommendation modules
│   ├── cli/               # Command-line interface
│   ├── core/              # Core trading logic
│   ├── database/          # Database models and queries
│   ├── portfolio/         # Portfolio management
│   ├── reporting/         # Report generation
│   └── utils/             # Utility functions
├── tests/                 # Test files (mirrors src/ structure)
├── config/                # Configuration files
│   └── collections/       # Asset collections (stocks, bonds, crypto)
├── docs/                  # Documentation
├── scripts/               # Utility scripts
│   ├── debug/            # Development debug scripts
│   └── fixes/            # One-time database fix scripts
├── exports/              # Generated outputs (organized by year/quarter)
├── cache/                # Data cache (Docker volume)
├── logs/                 # System logs (Docker volume)
└── quant-strategies/     # External strategies (Git submodule)
```

## 🧹 Code Quality

### Automated Tools
```bash
# Format code
poetry run ruff format .

# Lint and fix issues
poetry run ruff check --fix .

# Sort imports
poetry run isort . --skip quant-strategies

# Run tests
poetry run pytest

# Type checking
poetry run mypy .
```

### Pre-commit Hooks
```bash
# Install hooks
poetry run pre-commit install

# Run on all files
poetry run pre-commit run --all-files
```

## 🏗️ Development Workflow

### 1. Environment Setup
```bash
# Use Docker (recommended)
docker-compose up --build

# OR local development
poetry install
poetry shell
```

### 2. Making Changes
1. Create feature branch from main
2. Make changes following existing patterns
3. Add tests for new functionality
4. Run quality checks: `poetry run pre-commit run --all-files`
5. Test changes: `docker-compose run --rm quant pytest`

### 3. Database Changes
- Add migrations in `scripts/init-db.sql`
- Update models in `src/database/models.py`
- Test with: `docker-compose up postgres`

## 📊 Testing

### Running Tests
```bash
# All tests
docker-compose run --rm quant pytest

# Specific module
docker-compose run --rm quant pytest tests/core/

# With coverage
docker-compose run --rm quant pytest --cov=src
```

### Test Structure
- Tests mirror the `src/` structure
- Integration tests in `tests/integration/`
- Database tests require Docker PostgreSQL

## 🗂️ File Organization

### Source Code (`src/`)
- **Modules**: Each subdirectory is a distinct module
- **Imports**: Use absolute imports from src root
- **Dependencies**: Minimal coupling between modules

### Configuration (`config/`)
- **Collections**: Asset groupings by type
- **Settings**: JSON configuration files only

### Scripts (`scripts/`)
- **Production**: Root-level production scripts
- **Debug**: `debug/` for development debugging
- **Fixes**: `fixes/` for one-time database fixes

### Documentation (`docs/`)
- **Technical**: Implementation details
- **User**: CLI guides and feature documentation

## 🔒 Security & Best Practices

### Environment Variables
- Use `.env` for local secrets
- Never commit API keys or passwords
- Use Docker secrets in production

### Database Access
- Always use connection pooling
- Use transactions for multi-step operations
- Handle database errors gracefully

### Code Standards
- Type hints for all public functions
- Docstrings for complex logic
- Error handling with proper logging
- No hardcoded paths or URLs

## 📦 Dependencies

### Adding Dependencies
```bash
# Add to pyproject.toml
poetry add package-name

# Development dependencies
poetry add --group dev package-name

# Update Dockerfile if needed
docker-compose build quant
```

### Dependency Guidelines
- Prefer stable, well-maintained packages
- Pin major versions in pyproject.toml
- Test compatibility with existing stack
- Document new dependencies in README

## 🚀 Release Process

1. Update version in `pyproject.toml`
2. Run full test suite: `docker-compose run --rm quant pytest`
3. Validate CLI: `docker-compose run --rm quant python -m src.cli.unified_cli --help`
4. Generate changelog: `python scripts/generate_changelog.py`
5. Create release PR with updated documentation

## 🤝 Contributing Guidelines

- **Issues**: Use issue templates for bugs/features
- **PRs**: Include tests and documentation updates
- **Commits**: Use conventional commit format
- **Code Review**: All changes require review
- **CI/CD**: All checks must pass before merge

---

**Questions?** Check the [docs/](docs/) directory or open an issue.
