#!/bin/bash
# Code quality script

echo "Running code quality checks..."

# Format code with Black
echo "Formatting code with Black..."
poetry run black src/

# Sort imports with isort
echo "Sorting imports with isort..."
poetry run isort src/

# Run linter
echo "Running linter with Ruff..."
poetry run ruff check src/

echo "Code quality checks complete."