import logging
from typing import Dict, Any
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizerResultAnalyzer:
    """Analyzes the results of optimization backtests."""

    @staticmethod
    def analyze(results, ticker=None, initial_capital=10000):
        """Extracts key performance metrics from the backtest results."""
        if results is None:
            print("❌ No results returned from Backtest Engine.")
            return {
                "strategy": "N/A",
                "asset": "N/A" if ticker is None else ticker,
                "pnl": "$0.00",
                "sharpe_ratio": 0,
                "max_drawdown": "0.00%",
                "trades": 0,
                "initial_capital": initial_capital,
                "final_value": initial_capital
            }

        # Get strategy name directly from results
        strategy_name = results._strategy.__class__.__name__
        
        # Try multiple approaches to get the asset name
        asset_name = ticker if ticker else "N/A"
        if hasattr(results._strategy, '_data') and hasattr(results._strategy._data, 'name'):
            asset_name = results._strategy._data.name
        elif hasattr(results, '_data') and hasattr(results._data, 'name'):
            asset_name = results._data.name
        
        # Calculate PnL correctly
        final_value = results['Equity Final [$]']
        pnl = final_value - initial_capital
        
        # Access the stats via dictionary interface that Backtesting.py provides
        return {
            "strategy": strategy_name,
            "asset": asset_name,
            "pnl": f"${pnl:,.2f}",
            "sharpe_ratio": round(results['Sharpe Ratio'], 2),
            "max_drawdown": f"{results['Max. Drawdown [%]']:.2f}%",
            "trades": results['# Trades'],
            "win_rate": f"{results['Win Rate [%]']:.2f}%",
            "initial_capital": initial_capital,
            "final_value": final_value,
            "return_pct": f"{(pnl / initial_capital) * 100:.2f}%"
        }

    @staticmethod
    def calculate_max_drawdown(results) -> float:
        """Calculates the maximum drawdown from the backtest."""
        try:
            df = pd.DataFrame(results.analyzers.drawdown.get_analysis())
            if not df.empty:
                return df["drawdown"].max() / 100  # Convert percentage to float
        except Exception as e:
            logger.error(f"⚠️ Error calculating max drawdown: {e}")
        return 0.0  # Default to zero if analysis fails

    @staticmethod
    def calculate_sharpe_ratio(results) -> float:
        """Calculates the Sharpe Ratio from the backtest."""
        try:
            return results.analyzers.sharpe.get_analysis().get("sharperatio", 0)
        except Exception as e:
            logger.error(f"⚠️ Error calculating Sharpe Ratio: {e}")
            return 0.0  # Default to zero if analysis fails
