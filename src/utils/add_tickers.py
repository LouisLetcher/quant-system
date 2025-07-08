import argparse
import glob
import json
import os
import re
import sys

# Get the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
config_path = os.path.join(project_root, "config", "assets_config.json")
exports_dir = os.path.join(project_root, "exports")

# Create exports directory if it doesn't exist
if not os.path.exists(exports_dir):
    os.makedirs(exports_dir)
    print(f"Created exports directory at {exports_dir}")

# Set up argument parser
parser = argparse.ArgumentParser(
    description="Add tickers from export files to assets_config.json"
)
parser.add_argument(
    "export_file", nargs="?", help="Specific export file to process (optional)"
)
parser.add_argument(
    "--clear",
    action="store_true",
    help="Clear existing assets in the portfolio before adding new ones",
)
args = parser.parse_args()

# Load the existing assets_config.json
with open(config_path, "r") as f:
    config = json.load(f)

# Get available portfolios
available_portfolios = list(config["portfolios"].keys())

# Get export filename from command line argument, or process all files in exports directory
if args.export_file:
    # Process a specific file
    export_filename = args.export_file
    export_path = os.path.join(exports_dir, export_filename)
    export_files = [export_path]
else:
    # Process all *_export.txt files in the exports directory
    export_files = glob.glob(os.path.join(exports_dir, "*_export.txt"))
    if not export_files:
        print(f"No export files found in {exports_dir}")
        print(
            "Please place your export files in the 'exports' directory with the naming pattern: {portfolio_name}_export.txt"
        )
        print(f"Available portfolios: {', '.join(available_portfolios)}")
        exit(1)

# Process each export file
for export_path in export_files:
    # Extract portfolio name from filename
    filename = os.path.basename(export_path)
    match = re.match(r"([a-zA-Z0-9_]+)_export\.txt", filename)

    if not match:
        print(
            f"Warning: File {filename} does not follow the naming pattern: {{portfolio_name}}_export.txt"
        )
        print("Skipping this file.")
        continue

    portfolio_name = match.group(1)

    # Check if the portfolio exists
    if portfolio_name not in config["portfolios"]:
        print(f"Warning: Portfolio '{portfolio_name}' not found in assets_config.json")
        print(f"Available portfolios: {', '.join(available_portfolios)}")
        print(f"Skipping file {filename}")
        continue

    # Clear existing assets if --clear flag is used
    if args.clear:
        config["portfolios"][portfolio_name]["assets"] = []
        print(f"Cleared all existing assets from the {portfolio_name} portfolio.")

    # Extract existing tickers for this portfolio
    existing_tickers = set()
    for asset in config["portfolios"][portfolio_name]["assets"]:
        existing_tickers.add(asset["ticker"])

    # Read the tickers from the export file
    try:
        with open(export_path, "r") as f:
            input_string = f.read().strip()
    except FileNotFoundError:
        print(f"Error: Export file not found at {export_path}")
        continue

    # Parse the input string to extract tickers
    ticker_entries = input_string.split(",")
    new_tickers = []

    for entry in ticker_entries:
        # Skip header entries or empty entries
        if entry.startswith("###") or not entry.strip():
            continue

        # Extract ticker from exchange:ticker format and remove exchange prefix
        ticker = entry.strip()
        if ":" in ticker:
            # Remove exchange prefix (everything before and including ':')
            ticker = ticker.split(":", 1)[1].strip()

        # Add ticker if it's not already in the portfolio
        if ticker and ticker not in existing_tickers:
            new_tickers.append(ticker)
            existing_tickers.add(
                ticker
            )  # Add to set to avoid duplicates in new additions

    # Add new tickers to the portfolio
    for ticker in new_tickers:
        new_asset = {
            "ticker": ticker,
            "period": "max",
            "initial_capital": 1000,
            "commission": 0.002,
            "description": ticker,
        }
        config["portfolios"][portfolio_name]["assets"].append(new_asset)

    print(
        f"Added {len(new_tickers)} new tickers to the {portfolio_name} portfolio from {filename}."
    )
    if new_tickers:
        print("New tickers:", new_tickers)

# Save the updated configuration
with open(config_path, "w") as f:
    json.dump(config, f, indent=4)

print("Configuration saved successfully.")
print("All export files processed.")
print("Please check the assets_config.json file for the updated portfolios.")
