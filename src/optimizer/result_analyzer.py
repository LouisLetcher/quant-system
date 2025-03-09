import logging
from typing import Dict, Any
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizerResultAnalyzer:
    """Analyzes the results of optimization backtests."""
    @staticmethod
    def analyze(result, ticker=None, initial_capital=None):
        """Extracts key performance metrics from the backtest results."""
        if result is None:
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
        strategy_name = result._strategy.__class__.__name__
        
        # Try multiple approaches to get the asset name
        asset_name = ticker if ticker else "N/A"
        if hasattr(result._strategy, '_data') and hasattr(result._strategy._data, 'name'):
            asset_name = result._strategy._data.name
        elif hasattr(result, '_data') and hasattr(result._data, 'name'):
            asset_name = result._data.name
        
        # Calculate PnL correctly
        final_value = result['Equity Final [$]']
        pnl = final_value - initial_capital
        
        # Access the stats via dictionary interface that Backtesting.py provides
        analyzed_result = {
            "strategy": strategy_name,
            "asset": asset_name,
            "pnl": f"${pnl:,.2f}",
            "sharpe_ratio": round(result['Sharpe Ratio'], 2),
            "max_drawdown": f"{result['Max. Drawdown [%]']:.2f}%",
            "trades": result['# Trades'],
            "initial_capital": initial_capital,
            "final_value": final_value,
            "return_pct": f"{(pnl / initial_capital) * 100:.2f}%"
        }

        # Calculate win rate (percentage of profitable trades)
        if 'trades' in result and hasattr(result, 'trades') and result.trades:
            winning_trades = sum(1 for trade in result.trades if trade.pl > 0)
            total_trades = len(result.trades)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            analyzed_result['win_rate'] = win_rate
        else:
            analyzed_result['win_rate'] = 0
        
        # Calculate profit factor (gross profit / gross loss)
        if 'trades' in result and hasattr(result, 'trades') and result.trades:
            gross_profit = sum(trade.pl for trade in result.trades if trade.pl > 0)
            gross_loss = abs(sum(trade.pl for trade in result.trades if trade.pl < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0)
            analyzed_result['profit_factor'] = profit_factor
        else:
            analyzed_result['profit_factor'] = 0
        
        # Total P&L is already calculated as final_value - initial_capital
        
        return analyzed_result
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
