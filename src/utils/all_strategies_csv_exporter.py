"""
All Strategies CSV Export Utility

Exports ALL strategy-asset combinations from PostgreSQL database to CSV format.
Shows comprehensive comparison of all strategies for each asset.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import or_
from sqlalchemy.orm import Session

from src.database.models import BacktestResult


class AllStrategiesCSVExporter:
    """
    Export all strategy-asset combinations directly from PostgreSQL database to CSV.

    This exporter shows ALL strategies tested for each asset, making it easy to
    compare performance differences and identify the best strategy per asset.
    """

    def __init__(self, db_session: Session, output_dir: str = "exports/csv"):
        self.db_session = db_session
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def export_all_strategies(
        self,
        quarter: Optional[str] = None,
        year: Optional[str] = None,
        output_filename: Optional[str] = None,
        symbols: Optional[list[str]] = None,
        min_sortino: Optional[float] = None,
    ) -> str:
        """
        Export ALL strategy-asset combinations from PostgreSQL database to CSV.

        Args:
            quarter: Quarter filter (Q1, Q2, Q3, Q4)
            year: Year filter (YYYY)
            output_filename: Custom output filename
            symbols: Specific symbols to export
            min_sortino: Minimum Sortino ratio filter

        Returns:
            Path to exported CSV file
        """
        query = self.db_session.query(BacktestResult)

        # Apply filters
        if quarter and year:
            start_date, end_date = self._get_quarter_dates(quarter, int(year))
            query = query.filter(
                BacktestResult.created_at >= start_date,
                BacktestResult.created_at < end_date,
            )

        if symbols:
            # Filter for results containing any of the specified symbols
            conditions = []
            for symbol in symbols:
                conditions.append(BacktestResult.symbols.contains([symbol]))
            query = query.filter(or_(*conditions))

        if min_sortino:
            query = query.filter(BacktestResult.sortino_ratio >= min_sortino)

        # Get all results
        results = query.order_by(BacktestResult.sortino_ratio.desc()).all()

        if not results:
            self.logger.warning("No backtest results found")
            return ""

        self.logger.info(f"Found {len(results)} backtest results for CSV export")

        # Convert to DataFrame
        data = []
        for result in results:
            # Each result may contain multiple symbols, so we create one row per symbol
            for symbol in result.symbols:
                row = {
                    "Symbol": symbol,
                    "Strategy": result.strategy,
                    "Timeframe": "1d",  # Default timeframe
                    "Sortino Ratio": float(result.sortino_ratio or 0),
                    "Calmar Ratio": float(result.calmar_ratio or 0),
                    "Sharpe Ratio": float(result.sharpe_ratio or 0),
                    "Profit Factor": float(result.profit_factor or 0),
                    "Total Return %": float(result.total_return or 0),
                    "Max Drawdown %": float(result.max_drawdown or 0),
                    "Volatility %": float(result.volatility or 0),
                    "Win Rate %": float(result.win_rate or 0),
                    "Number of Trades": int(result.num_trades or 0),
                    "Alpha": float(result.alpha or 0),
                    "Beta": float(result.beta or 1),
                    "Average Win %": float(result.average_win or 0),
                    "Average Loss %": float(result.average_loss or 0),
                    "Total Fees $": float(result.total_fees or 0),
                    "Portfolio Turnover": float(result.portfolio_turnover or 0),
                    "Strategy Capacity $": float(result.strategy_capacity or 1000000),
                    "Final Capital $": float(result.final_value or 0),
                    "Start Date": result.start_date.strftime("%Y-%m-%d")
                    if result.start_date
                    else "",
                    "End Date": result.end_date.strftime("%Y-%m-%d")
                    if result.end_date
                    else "",
                    "Last Updated": result.created_at.strftime("%Y-%m-%d %H:%M:%S")
                    if result.created_at
                    else "",
                }
                data.append(row)

        df = pd.DataFrame(data)

        # Sort by Symbol first, then by Sortino Ratio descending
        df = df.sort_values(["Symbol", "Sortino Ratio"], ascending=[True, False])

        # Generate filename
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if quarter and year:
                output_filename = f"all_strategies_{quarter}_{year}_{timestamp}.csv"
            elif year:
                output_filename = f"all_strategies_{year}_{timestamp}.csv"
            else:
                output_filename = f"all_strategies_{timestamp}.csv"

        # Ensure .csv extension
        if not output_filename.endswith(".csv"):
            output_filename += ".csv"

        # Save to CSV
        output_path = self.output_dir / output_filename
        df.to_csv(output_path, index=False)

        self.logger.info(
            f"Exported {len(df)} strategy-asset combinations to {output_path}"
        )
        return str(output_path)

    def _get_quarter_dates(self, quarter: str, year: int) -> tuple[datetime, datetime]:
        """Get start and end dates for a quarter."""
        quarter_num = int(quarter[1])  # Extract number from Q1, Q2, etc.
        start_month = (quarter_num - 1) * 3 + 1

        start_date = datetime(year, start_month, 1)

        if quarter_num == 4:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_month = quarter_num * 3
            end_date = datetime(year, end_month + 1, 1)

        return start_date, end_date
