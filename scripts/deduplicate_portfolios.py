#!/usr/bin/env python3
"""Remove duplicate symbols from portfolio configuration files."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.file_utils import load_json_file, save_json_file


def deduplicate_portfolio(file_path: Path) -> None:
    """Remove duplicate symbols from a portfolio file."""
    data = load_json_file(file_path)

    for portfolio_data in data.values():
        if "symbols" in portfolio_data:
            original_count = len(portfolio_data["symbols"])
            unique_symbols = list(dict.fromkeys(portfolio_data["symbols"]))
            portfolio_data["symbols"] = unique_symbols

            if original_count != len(unique_symbols):
                print(
                    f"{file_path.name}: {original_count} -> {len(unique_symbols)} symbols"
                )

    save_json_file(file_path, data)


def main():
    """Deduplicate all portfolio files."""
    portfolio_dir = Path("config/portfolios")
    for json_file in portfolio_dir.glob("*.json"):
        deduplicate_portfolio(json_file)


if __name__ == "__main__":
    main()
