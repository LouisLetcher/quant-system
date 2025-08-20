"""
Simple CSV Exporter - Direct Backtesting Library Data
Exports real performance data from database to CSV format.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.database import get_db_session
from src.database.models import BacktestResult, BestStrategy, Trade


class SimpleCSVExporter:
    """Export real backtesting data to CSV format."""

    def __init__(self, output_dir: str = "exports/csv"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_best_strategies(self, filename: Optional[str] = None) -> str:
        """Export best strategies with real backtesting library data to CSV."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"best_strategies_real_data_{timestamp}.csv"

        session = get_db_session()

        try:
            # Get all best strategies from database
            best_strategies = (
                session.query(BestStrategy)
                .order_by(BestStrategy.sortino_ratio.desc())
                .all()
            )

            self.logger.info(
                "Exporting %d best strategies to CSV", len(best_strategies)
            )

            # Convert to CSV format
            csv_data = []
            for bs in best_strategies:
                csv_data.append(
                    {
                        "Symbol": bs.symbol,
                        "Best_Strategy": bs.strategy,
                        "Timeframe": bs.timeframe,
                        "Sortino_Ratio": float(bs.sortino_ratio or 0),
                        "Sharpe_Ratio": float(bs.sharpe_ratio or 0),
                        "Total_Return_Percent": float(bs.total_return or 0),
                        "Max_Drawdown_Percent": float(bs.max_drawdown or 0),
                        "Calmar_Ratio": float(bs.calmar_ratio or 0),
                        "Data_Source": "backtesting_library_real_data",
                        "Last_Updated": bs.updated_at.strftime("%Y-%m-%d %H:%M:%S")
                        if bs.updated_at
                        else "",
                    }
                )

            # Create DataFrame and save
            df = pd.DataFrame(csv_data)
            output_path = self.output_dir / filename

            df.to_csv(output_path, index=False)

            self.logger.info("CSV export completed: %s", output_path)
            return str(output_path)

        except Exception as e:
            self.logger.error("CSV export failed: %s", e)
            raise e
        finally:
            session.close()

    def export_detailed_results(
        self, symbol: Optional[str] = None, filename: Optional[str] = None
    ) -> str:
        """Export detailed backtest results to CSV."""
        if not filename:
            symbol_suffix = f"_{symbol}" if symbol else "_all"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results{symbol_suffix}_{timestamp}.csv"

        session = get_db_session()

        try:
            query = session.query(BacktestResult)
            if symbol:
                query = query.filter(BacktestResult.symbols.any(symbol))

            results = query.order_by(BacktestResult.sortino_ratio.desc()).all()

            self.logger.info("Exporting %d detailed results to CSV", len(results))

            csv_data = []
            for result in results:
                symbols_str = ",".join(result.symbols) if result.symbols else ""
                csv_data.append(
                    {
                        "Symbols": symbols_str,
                        "Strategy": result.strategy,
                        "Timeframe": result.timeframe,
                        "Start_Date": result.start_date.strftime("%Y-%m-%d")
                        if result.start_date
                        else "",
                        "End_Date": result.end_date.strftime("%Y-%m-%d")
                        if result.end_date
                        else "",
                        "Initial_Capital": float(result.initial_capital or 0),
                        "Final_Value": float(result.final_value or 0),
                        "Total_Return_Percent": float(result.total_return or 0),
                        "Sortino_Ratio": float(result.sortino_ratio or 0),
                        "Sharpe_Ratio": float(result.sharpe_ratio or 0),
                        "Calmar_Ratio": float(result.calmar_ratio or 0),
                        "Max_Drawdown_Percent": float(result.max_drawdown or 0),
                        "Volatility_Percent": float(result.volatility or 0),
                        "Win_Rate_Percent": float(result.win_rate or 0),
                        "Number_of_Trades": int(result.trades_count or 0),
                        "Profit_Factor": float(result.profit_factor or 1),
                        "Data_Source": "backtesting_library_direct",
                        "Created_At": result.created_at.strftime("%Y-%m-%d %H:%M:%S")
                        if result.created_at
                        else "",
                    }
                )

            df = pd.DataFrame(csv_data)
            output_path = self.output_dir / filename

            df.to_csv(output_path, index=False)

            self.logger.info("Detailed CSV export completed: %s", output_path)
            return str(output_path)

        except Exception as e:
            self.logger.error("Detailed CSV export failed: %s", e)
            raise e
        finally:
            session.close()

    def export_trades(
        self, symbol: Optional[str] = None, filename: Optional[str] = None
    ) -> str:
        """Export real trade data to CSV."""
        if not filename:
            symbol_suffix = f"_{symbol}" if symbol else "_all"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"real_trades{symbol_suffix}_{timestamp}.csv"

        session = get_db_session()

        try:
            query = session.query(Trade)
            if symbol:
                query = query.filter_by(symbol=symbol)

            trades = query.order_by(Trade.trade_datetime.asc()).all()

            self.logger.info("Exporting %d real trades to CSV", len(trades))

            csv_data = []
            for trade in trades:
                csv_data.append(
                    {
                        "Symbol": trade.symbol,
                        "Strategy": trade.strategy,
                        "Timeframe": trade.timeframe,
                        "Trade_DateTime": trade.trade_datetime.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "Trade_Type": trade.side,
                        "Price": float(trade.price),
                        "Size": float(trade.size),
                        "Equity_Before": float(trade.equity_before or 0),
                        "Equity_After": float(trade.equity_after or 0),
                        "Data_Source": "backtesting_library_real_trades",
                    }
                )

            df = pd.DataFrame(csv_data)
            output_path = self.output_dir / filename

            df.to_csv(output_path, index=False)

            self.logger.info("Trades CSV export completed: %s", output_path)
            return str(output_path)

        except Exception as e:
            self.logger.error("Trades CSV export failed: %s", e)
            raise e
        finally:
            session.close()
