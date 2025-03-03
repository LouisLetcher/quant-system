import numpy as np

class ResultAnalyzer:
    """Analyzes backtest results and extracts performance metrics."""

    @staticmethod
    def analyze(results, ticker=None):
        """Extracts key performance metrics from the backtest results."""
        if results is None:
            print("❌ No results returned from Backtest Engine.")
            return {
                "strategy": "N/A",
                "asset": "N/A",
                "pnl": "$0.00",
                "sharpe_ratio": 0,
                "max_drawdown": "0.00%",
            }
        
        # Use the provided ticker if available
        asset_name = ticker if ticker else "N/A"
        
        # Backtesting.py provides these metrics directly
        return {
            "strategy": results._strategy.__class__.__name__,
            "asset": asset_name,
            "pnl": f"${results['Return [%]'] * results['Equity Final [$]'] / 100:,.2f}",
            "sharpe_ratio": round(results['Sharpe Ratio'], 2),
            "max_drawdown": f"{results['Max. Drawdown [%]']:.2f}%",
        }

    @staticmethod
    def calculate_max_drawdown(strategy_instance):
        """Calculates maximum drawdown from the account value."""
        account_values = np.array([strategy_instance.broker.getvalue() for _ in range(len(strategy_instance))])
        peak = np.maximum.accumulate(account_values)
