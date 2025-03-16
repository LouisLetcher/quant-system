#!/bin/bash
# Setup script for the Quant Trading System

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
else
    echo "Poetry is already installed."
fi

# Install dependencies
echo "Installing dependencies..."
poetry install

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p reports_output
mkdir -p logs

# Make scripts executable
echo "Making scripts executable..."
chmod +x scripts/*.sh

echo "Setup complete! You can now run the scripts in the scripts directory."
