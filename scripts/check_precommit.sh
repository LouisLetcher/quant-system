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

# Install custom hooks
HOOKS_DIR=.git/hooks
CUSTOM_HOOKS_DIR=scripts/hooks

# Install changelog hook
if [ -f "$CUSTOM_HOOKS_DIR/pre-commit-changelog" ]; then
    echo "Installing changelog pre-commit hook..."
    cp "$CUSTOM_HOOKS_DIR/pre-commit-changelog" "$HOOKS_DIR/pre-commit-changelog"
    chmod +x "$HOOKS_DIR/pre-commit-changelog"
    
    # Add to pre-commit if not already included
    if ! grep -q "pre-commit-changelog" "$HOOKS_DIR/pre-commit"; then
        echo -e "\n# Run changelog generator\n.git/hooks/pre-commit-changelog" >> "$HOOKS_DIR/pre-commit"
    fi
fi

# Run pre-commit on all files
echo "Running pre-commit checks on all files..."
poetry run pre-commit run --all-files --show-diff-on-failure

echo "✅ All pre-commit checks passed!"
