#!/usr/bin/env bash
# Run pre-commit hooks on all files and install hooks if needed

# Exit on error, print commands, expand variables
set -ex

# Check if pre-commit is installed
if ! poetry run pre-commit --version &>/dev/null; then
    echo "Installing pre-commit..."
    poetry add --group dev pre-commit
fi

# Install pre-commit hooks
poetry run pre-commit install

# Run pre-commit on all files
echo "Running pre-commit checks on all files..."
poetry run pre-commit run --all-files --show-diff-on-failure

echo "✅ All pre-commit checks passed!"
echo "You can now commit your changes."