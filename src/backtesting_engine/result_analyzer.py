import numpy as np
import pandas as pd
from datetime import datetime

class BacktestResultAnalyzer:
    """Analyzes backtest results and extracts performance metrics."""

    @staticmethod
    def analyze(backtest_result, ticker=None, initial_capital=10000):
        """Extracts key performance metrics from the backtest results."""
        if backtest_result is None:
            print("❌ No results returned from Backtest Engine.")
            return {
                "strategy": "N/A",
                "asset": "N/A" if ticker is None else ticker,
                "pnl": "$0.00",
                "sharpe_ratio": 0,
                "max_drawdown": "0.00%",
                "trades": 0,
                "initial_capital": initial_capital,
                "final_value": initial_capital,
                "suspicious_result": False,
                "win_rate": 0,
                "profit_factor": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "equity_curve": [],
                "drawdown_curve": [],
                "trades_list": []
            }

        # Extract metrics directly from Backtesting.py results
        # The results object from Backtesting.py contains various metrics

        # For single asset results
        asset_name = ticker if ticker else "N/A"
        final_value = backtest_result['Equity Final [$]']
        pnl = final_value - initial_capital

        # Add validation for suspicious results
        return_pct = (pnl / initial_capital) * 100
        trade_count = backtest_result.get('# Trades', 0)

        # Flag unrealistic results
        suspicious = trade_count == 0 and return_pct > 1000

        # Extract additional metrics
        win_rate = backtest_result.get('Win Rate [%]', 0)
        profit_factor = backtest_result.get('Profit Factor', 0)
        avg_win = backtest_result.get('Best Trade [%]', 0)
        avg_loss = backtest_result.get('Worst Trade [%]', 0)

        # Extract equity curve and trades data
        equity_curve = BacktestResultAnalyzer._extract_equity_curve(backtest_result)
        drawdown_curve = BacktestResultAnalyzer._extract_drawdown_curve(backtest_result)
        trades_list = BacktestResultAnalyzer._extract_trades_list(backtest_result)
        tv_profit_factor = BacktestResultAnalyzer.calculate_tradingview_profit_factor(trades_list)

        results = {
            "strategy": backtest_result._strategy.__class__.__name__,
            "asset": asset_name,
            "pnl": f"${pnl:,.2f}",
            "sharpe_ratio": round(backtest_result['Sharpe Ratio'], 2),
            "profit_factor": profit_factor,
            "tv_profit_factor": round(tv_profit_factor, 2),
            "max_drawdown": f"{backtest_result['Max. Drawdown [%]']:.2f}%",
            "trades": trade_count,
            "initial_capital": initial_capital,
            "final_value": final_value,
            "return_pct": f"{return_pct:.2f}%",
            "is_portfolio": False,
            "suspicious_result": suspicious,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "equity_curve": equity_curve,
            "drawdown_curve": drawdown_curve,
            "trades_list": trades_list
        }

        # Ensure profit factor is correctly added to results
        if 'profit_factor' not in results and hasattr(backtest_result, '_trades') and len(backtest_result._trades) > 0:
            trades_list = [
                {
                    'pnl': trade['PnL'],
                    'entry_date': trade['EntryTime'],
                    'exit_date': trade['ExitTime'],
                    'entry_price': trade['EntryPrice'],
                    'exit_price': trade['ExitPrice'],
                    'size': trade['Size'],
                    'type': 'LONG'  # Default to LONG if not specified
                }
                for trade in backtest_result._trades.to_dict('records')
            ]
            
            # Add trades list to results
            results['trades_list'] = trades_list
            
            # Calculate and add profit factor
            gross_profit = sum(trade['pnl'] for trade in trades_list if trade['pnl'] > 0)
            gross_loss = sum(abs(trade['pnl']) for trade in trades_list if trade['pnl'] < 0)
            
            if gross_loss > 0:
                results['profit_factor'] = gross_profit / gross_loss
            else:
                results['profit_factor'] = float('inf') if gross_profit > 0 else 0
        
        # Make sure trade count is properly set
        if 'trades' not in results and hasattr(backtest_result, '_trades'):
            results['trades'] = len(backtest_result._trades)

        return results
    # Helper methods for extracting data from results
    @staticmethod
    def _extract_equity_curve(results):
        """Extract equity curve data from results."""
        # Implementation details...
        pass

    @staticmethod
    def _extract_drawdown_curve(results):
        """Extract drawdown curve data from results."""
        # Implementation details...
        pass

    @staticmethod
    def _extract_trades_list(results):
        """Extract list of trades from results."""
        # Implementation details...
        pass

    @staticmethod
    def calculate_profit_factor(trades_list):
        """Calculate profit factor safely, handling edge cases."""
        gross_profit = sum(trade.get('pnl', 0) for trade in trades_list if trade.get('pnl', 0) > 0)
        gross_loss = sum(abs(trade.get('pnl', 0)) for trade in trades_list if trade.get('pnl', 0) < 0)

        # Prevent division by zero
        if gross_loss == 0:
            return 0.0 if gross_profit == 0 else float('inf')

        return gross_profit / gross_loss