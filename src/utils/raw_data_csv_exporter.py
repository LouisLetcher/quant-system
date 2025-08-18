"""
Raw Data CSV Export Utility

Exports portfolio performance data directly from PostgreSQL database to CSV format.
Replaces HTML parsing with direct database queries for better performance.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session


class RawDataCSVExporter:
    """
    Export portfolio data directly from PostgreSQL database to CSV format.

    Features:
    - Direct database queries (no HTML parsing)
    - CSV export with symbol, best strategy, timeframe, and performance metrics
    - Quarter/year filtering with proper date handling
    - Customizable column selection (Sharpe, Sortino, profit, drawdown)
    - Multiple export formats (full, best-strategies, quarterly summary)
    """

    def __init__(
        self, db_session: Optional[Session] = None, output_dir: str = "exports/csv"
    ):
        self.db_session = db_session
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = Path("exports/reports")  # Fallback for legacy HTML parsing
        self.logger = logging.getLogger(__name__)

    def export_from_database(
        self,
        quarter: Optional[str] = None,
        year: Optional[str] = None,
        output_filename: Optional[str] = None,
        export_format: str = "full",
        columns: Optional[list[str]] = None,
    ) -> str:
        """
        Export backtest results directly from PostgreSQL database to CSV.

        Args:
            quarter: Quarter filter (Q1, Q2, Q3, Q4)
            year: Year filter (YYYY)
            output_filename: Custom output filename
            export_format: Export format (full, best-strategies, quarterly)
            columns: Custom column selection

        Returns:
            Path to exported CSV file
        """
        if not self.db_session:
            self.logger.error("No database session - falling back to HTML parsing")
            return self.export_from_quarterly_reports(
                quarter, year, output_filename, export_format
            )

        # Use new query helper for better performance and consistency
        from ..database.query_helpers import DatabaseQueryHelper

        query_helper = DatabaseQueryHelper(self.db_session)

        # Get best strategies data
        strategies_data = query_helper.get_best_strategies(
            quarter=quarter, year=int(year) if year else None
        )

        if not strategies_data:
            self.logger.warning("No best strategies found in database")
            return ""

        self.logger.info(f"Found {len(strategies_data)} best strategies for CSV export")

        # For collection-based exports, create separate files per collection
        return self._export_collection_based_files(
            strategies_data, quarter, year, export_format
        )

    def _export_collection_based_files(
        self, strategies_data, quarter: str, year: str, export_format: str
    ) -> list[str]:
        """Create collection-based CSV files from database data."""

        # Define collections and their symbols
        collections = {
            "Commodities_Collection": [
                "GC=F",
                "SI=F",
                "CL=F",
                "NG=F",
                "ZC=F",
                "ZS=F",
                "ZW=F",
                "KC=F",
                "CC=F",
                "SB=F",
                "CT=F",
                "HE=F",
                "LE=F",
                "HG=F",
                "PA=F",
                "PL=F",
            ],
            "Bonds_Collection": [
                "HYG",
                "VMBS",
                "IEI",
                "GOVT",
                "VCIT",
                "EDV",
                "FLOT",
                "EMB",
                "JPST",
                "VCSH",
                "SHY",
                "TIP",
                "VGIT",
                "VGLT",
            ],
        }

        # Create output directory
        output_dir = self.output_dir / year / quarter
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths = []

        for collection_name, symbols in collections.items():
            # Filter strategies data for this collection
            collection_data = [s for s in strategies_data if s.symbol in symbols]

            if not collection_data:
                continue

            # Convert to DataFrame
            df = self._strategies_data_to_dataframe(
                collection_data, export_format, None
            )

            # Generate filename
            filename = f"{collection_name}_{quarter}_{year}.csv"
            output_path = output_dir / filename

            # Save to CSV
            df.to_csv(output_path, index=False)
            output_paths.append(str(output_path))

            self.logger.info(
                f"Exported {len(df)} records for {collection_name} to {output_path}"
            )

        return output_paths

    def export_from_quarterly_reports(
        self,
        quarter: str,
        year: str,
        output_filename: Optional[str] = None,
        export_format: str = "full",
    ) -> str:
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

            # Sort by performance (convert to numeric first to handle mixed types)
            if "Sortino_Ratio" in df.columns:
                df["Sortino_Ratio"] = pd.to_numeric(
                    df["Sortino_Ratio"], errors="coerce"
                )
                df = df.sort_values("Sortino_Ratio", ascending=False)
            elif "Total_Return_Pct" in df.columns:
                df["Total_Return_Pct"] = pd.to_numeric(
                    df["Total_Return_Pct"], errors="coerce"
                )
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

    def _strategies_data_to_dataframe(
        self,
        strategies_data: list[dict],
        export_format: str = "full",
        columns: list[str] = None,
    ) -> pd.DataFrame:
        """Convert strategies data to DataFrame."""

        data = []
        for strategy in strategies_data:
            row = {
                "Symbol": strategy["symbol"],
                "Strategy": strategy["strategy"],
                "Timeframe": strategy["timeframe"],
                "Sortino Ratio": strategy["sortino_ratio"],
                "Calmar Ratio": strategy["calmar_ratio"],
                "Sharpe Ratio": strategy["sharpe_ratio"],
                "Profit Factor": strategy["profit_factor"],
                "Total Return": strategy["total_return"],
                "Max Drawdown": strategy["max_drawdown"],
                "Volatility": strategy["volatility"],
                "Win Rate": strategy["win_rate"],
                "Trades Count": strategy["num_trades"],
                "Risk Score": strategy["risk_score"],
                "Risk Per Trade": strategy["risk_per_trade"],
                "Stop Loss %": strategy["stop_loss_pct"],
                "Take Profit %": strategy["take_profit_pct"],
                "Last Updated": strategy["last_updated"] or "",
            }
            data.append(row)

        df = pd.DataFrame(data)

        # Apply format-specific filtering
        if export_format == "best-strategies":
            essential_cols = [
                "Symbol",
                "Strategy",
                "Timeframe",
                "Sortino Ratio",
                "Total Return",
                "Max Drawdown",
                "Win Rate",
            ]
            df = df[essential_cols]

        elif export_format == "quarterly":
            quarterly_cols = [
                "Symbol",
                "Strategy",
                "Sortino Ratio",
                "Calmar Ratio",
                "Total Return",
                "Max Drawdown",
                "Trades Count",
            ]
            df = df[quarterly_cols]

        # Apply custom column selection
        if columns:
            available_cols = [col for col in columns if col in df.columns]
            if available_cols:
                df = df[available_cols]

        return df

    def _get_quarter_dates(self, quarter: str, year: str) -> tuple[datetime, datetime]:
        """Get start and end dates for a quarter."""
        quarter_num = int(quarter[1])  # Extract number from Q1, Q2, etc.
        start_month = (quarter_num - 1) * 3 + 1

        start_date = datetime(int(year), start_month, 1)

        if quarter_num == 4:
            end_date = datetime(int(year) + 1, 1, 1)
        else:
            end_month = quarter_num * 3
            end_date = datetime(int(year), end_month + 1, 1)

        return start_date, end_date
