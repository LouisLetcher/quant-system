#!/bin/bash
# Pre-commit hook to update CHANGELOG.md

# Get the latest tag
LATEST_TAG=$(git describe --tags --abbrev=0 2>/dev/null)

if [ -z "$LATEST_TAG" ]; then
    # If no tags exist, use the initial commit
    LATEST_TAG=$(git rev-list --max-parents=0 HEAD)
fi

# Generate changelog since the latest tag
python scripts/generate_changelog.py --since "$LATEST_TAG"

# Add the updated CHANGELOG.md to the commit
git add CHANGELOG.md