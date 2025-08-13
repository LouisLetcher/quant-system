"""
Raw Data CSV Export Utility

Exports portfolio performance data with best strategies and timeframes to CSV format.
Based on the features.md specification and crypto_best_strategies.csv format.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup


class RawDataCSVExporter:
    """
    Export raw portfolio data with best strategies and performance metrics to CSV.

    Features:
    - CSV export with symbol, best strategy, best timeframe, and performance metrics
    - Bulk export for all assets from quarterly reports
    - Customizable column selection (Sharpe, Sortino, profit, drawdown)
    - Integration with existing quarterly report structure
    """

    def __init__(self, output_dir: str = "exports/data_exports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = Path("exports/reports")
        self.logger = logging.getLogger(__name__)

    def export_from_quarterly_reports(
        self,
        quarter: str,
        year: str,
        output_filename: str | None = None,
        export_format: str = "full",
    ) -> list[str]:
        """
        Extract data from existing quarterly reports and export to CSV.
        Creates separate CSV files for each HTML report (e.g., Crypto_Portfolio_Q3_2025.csv).

        Args:
            quarter: Quarter (Q1, Q2, Q3, Q4)
            year: Year (YYYY)
            output_filename: Custom filename, auto-generated if None (used for single file export)
            export_format: 'full' or 'best-strategies'

        Returns:
            List of paths to exported CSV files
        """
        # Check if quarterly reports exist
        quarterly_reports_dir = self.reports_dir / year / quarter
        if not quarterly_reports_dir.exists():
            self.logger.warning("No quarterly reports found for %s %s", quarter, year)
            return []

        # Find HTML report files
        html_files = list(quarterly_reports_dir.glob("*.html"))
        if not html_files:
            self.logger.warning("No HTML reports found in %s", quarterly_reports_dir)
            return []

        self.logger.info(
            "Found %d HTML reports for %s %s", len(html_files), quarter, year
        )

        # Create quarterly directory structure
        quarterly_dir = self.output_dir / year / quarter
        quarterly_dir.mkdir(parents=True, exist_ok=True)

        exported_files = []

        # Process each HTML report separately
        for html_file in html_files:
            # Extract data from this specific report
            extracted_data = self._extract_data_from_html_report(html_file)

            if not extracted_data:
                self.logger.warning("No data extracted from %s", html_file.name)
                continue

            # Convert to DataFrame
            df = pd.DataFrame(extracted_data)

            # Generate CSV filename based on HTML filename
            csv_filename = html_file.stem + ".csv"  # Remove .html and add .csv

            # Override with custom filename if provided and only one file
            if output_filename and len(html_files) == 1:
                csv_filename = output_filename

            # Process based on format
            if export_format == "best-strategies":
                # Group by symbol and keep best performing strategy
                if "Symbol" in df.columns and "Sortino_Ratio" in df.columns:
                    df = (
                        df.sort_values("Sortino_Ratio", ascending=False)
                        .groupby("Symbol")
                        .first()
                        .reset_index()
                    )
                    df = df[["Symbol", "Strategy", "Timeframe"]].rename(
                        columns={
                            "Symbol": "Asset",
                            "Strategy": "Best Strategy",
                            "Timeframe": "Resolution",
                        }
                    )

            # Add quarterly metadata
            df["Quarter"] = quarter
            df["Year"] = year
            df["Export_Date"] = pd.Timestamp.now().strftime("%Y-%m-%d")

            # Sort by performance
            if "Sortino_Ratio" in df.columns:
                df = df.sort_values("Sortino_Ratio", ascending=False)
            elif "Total_Return_Pct" in df.columns:
                df = df.sort_values("Total_Return_Pct", ascending=False)

            # Export to quarterly directory
            output_path = quarterly_dir / csv_filename
            df.to_csv(output_path, index=False)

            exported_files.append(str(output_path))

            self.logger.info(
                "Exported %s data from %s to %s (%d rows)",
                export_format,
                html_file.name,
                output_path,
                len(df),
            )

        self.logger.info(
            "Exported %d CSV files from quarterly reports for %s %s",
            len(exported_files),
            quarter,
            year,
        )

        return exported_files

    def get_available_columns(self) -> list[str]:
        """Get list of all available columns for export."""
        return [
            "Symbol",
            "Strategy",
            "Timeframe",
            "Total_Return_Pct",
            "Sortino_Ratio",
            "Sharpe_Ratio",
            "Calmar_Ratio",
            "Max_Drawdown_Pct",
            "Win_Rate_Pct",
            "Profit_Factor",
            "Number_of_Trades",
            "Volatility_Pct",
            "Downside_Deviation",
            "Average_Win",
            "Average_Loss",
            "Longest_Win_Streak",
            "Longest_Loss_Streak",
            "Data_Points",
            "Backtest_Duration_Seconds",
        ]

    def _extract_data_from_html_report(self, html_file: Path) -> list[dict[str, Any]]:
        """Extract performance data from HTML report files."""
        try:
            with html_file.open("r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")

            extracted_data = []

            # Look for tables with performance data
            tables = soup.find_all("table")

            for table in tables:
                # Check if this is a performance metrics table
                headers = table.find("tr")
                if not headers:
                    continue

                header_cells = [
                    th.get_text(strip=True) for th in headers.find_all(["th", "td"])
                ]

                # Look for tables that contain symbol/strategy information
                if any(
                    keyword in " ".join(header_cells).lower()
                    for keyword in ["symbol", "strategy", "asset", "sortino", "sharpe"]
                ):
                    rows = table.find_all("tr")[1:]  # Skip header row

                    for row in rows:
                        cells = [
                            td.get_text(strip=True) for td in row.find_all(["td", "th"])
                        ]
                        if len(cells) < 2:
                            continue

                        # Try to extract data based on common patterns
                        row_data = self._parse_table_row(header_cells, cells)
                        if row_data:
                            extracted_data.append(row_data)

            # Also look for metric cards or divs with performance data
            metric_cards = soup.find_all(
                "div", class_=re.compile(r".*metric.*|.*card.*|.*performance.*", re.I)
            )
            for card in metric_cards:
                card_data = self._parse_metric_card(card)
                if card_data:
                    extracted_data.append(card_data)

            self.logger.info(
                "Extracted %d data points from %s", len(extracted_data), html_file.name
            )
            return extracted_data

        except Exception as e:
            self.logger.error("Failed to parse HTML file %s: %s", html_file, e)
            return []

    def _parse_table_row(
        self, headers: list[str], cells: list[str]
    ) -> dict[str, Any] | None:
        """Parse a table row and extract relevant metrics."""
        if len(headers) != len(cells):
            return None

        row_data = {}

        # Map common header patterns to our standard columns
        header_mapping = {
            "symbol": "Symbol",
            "asset": "Symbol",
            "strategy": "Strategy",
            "timeframe": "Timeframe",
            "resolution": "Timeframe",
            "total_return": "Total_Return_Pct",
            "return": "Total_Return_Pct",
            "sortino": "Sortino_Ratio",
            "sharpe": "Sharpe_Ratio",
            "calmar": "Calmar_Ratio",
            "drawdown": "Max_Drawdown_Pct",
            "win_rate": "Win_Rate_Pct",
            "profit_factor": "Profit_Factor",
            "trades": "Number_of_Trades",
            "volatility": "Volatility_Pct",
        }

        for i, header in enumerate(headers):
            if i >= len(cells):
                break

            header_lower = (
                header.lower()
                .replace(" ", "_")
                .replace("%", "")
                .replace("(", "")
                .replace(")", "")
            )

            # Find matching column name
            mapped_column = None
            for pattern, column in header_mapping.items():
                if pattern in header_lower:
                    mapped_column = column
                    break

            if mapped_column and cells[i]:
                try:
                    # Try to convert numeric values
                    if mapped_column in [
                        "Total_Return_Pct",
                        "Sortino_Ratio",
                        "Sharpe_Ratio",
                        "Calmar_Ratio",
                        "Max_Drawdown_Pct",
                        "Win_Rate_Pct",
                        "Profit_Factor",
                        "Volatility_Pct",
                    ]:
                        # Remove % signs and other formatting
                        clean_value = re.sub(r"[%$,\s]", "", cells[i])
                        if clean_value and clean_value != "-":
                            row_data[mapped_column] = float(clean_value)
                    elif mapped_column == "Number_of_Trades":
                        clean_value = re.sub(r"[,\s]", "", cells[i])
                        if clean_value and clean_value.isdigit():
                            row_data[mapped_column] = int(clean_value)
                    else:
                        row_data[mapped_column] = cells[i]
                except (ValueError, TypeError):
                    row_data[mapped_column] = cells[i]

        # Only return if we have at least symbol or strategy
        if "Symbol" in row_data or "Strategy" in row_data:
            # Set defaults
            if "Timeframe" not in row_data:
                row_data["Timeframe"] = "1d"
            return row_data

        return None

    def _parse_metric_card(self, card) -> dict[str, Any] | None:
        """Parse metric cards for performance data."""
        # This would need to be customized based on the actual HTML structure
        # of the reports generated by the system
        text = card.get_text(strip=True)

        # Look for patterns like "BTCUSDT: 45.2%" or "Strategy: BuyAndHold"
        symbol_match = re.search(r"([A-Z0-9]+USDT?):?\s*([-+]?\d+\.?\d*%?)", text)
        strategy_match = re.search(r"Strategy:?\s*([A-Za-z\s]+)", text)

        if symbol_match or strategy_match:
            card_data = {}
            if symbol_match:
                card_data["Symbol"] = symbol_match.group(1)
                if len(symbol_match.groups()) > 1:
                    try:
                        value = float(symbol_match.group(2).replace("%", ""))
                        card_data["Total_Return_Pct"] = value
                    except ValueError:
                        pass

            if strategy_match:
                card_data["Strategy"] = strategy_match.group(1).strip()

            card_data["Timeframe"] = "1d"  # Default
            return card_data

        return None
