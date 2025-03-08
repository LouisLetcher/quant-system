import numpy as np

class BacktestResultAnalyzer:
    """Analyzes backtest results and extracts performance metrics."""

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
        
        # Check if this is a portfolio result
        if results.get('_portfolio', False):
            # Process portfolio results
            portfolio_name = ticker if ticker else "Portfolio"
            final_value = results['Equity Final [$]']
            pnl = final_value - initial_capital
            
            # Create main portfolio summary
            portfolio_summary = {
                "strategy": results['_strategy'].__name__,
                "asset": portfolio_name,
                "pnl": f"${pnl:,.2f}",
                "sharpe_ratio": round(results['Sharpe Ratio'], 2),
                "max_drawdown": f"{results['Max. Drawdown [%]']:.2f}%",
                "trades": results['# Trades'],
                "initial_capital": initial_capital,
                "final_value": final_value,
                "return_pct": f"{results['Return [%]']:.2f}%",
                "is_portfolio": True,
                "assets": results['_assets'],
                "asset_details": {}
            }
            
            # Add individual asset details
            for asset_ticker, asset_result in results['asset_results'].items():
                asset_final = asset_result['Equity Final [$]']
                asset_initial = initial_capital / len(results['_assets'])
                asset_pnl = asset_final - asset_initial
                
                portfolio_summary["asset_details"][asset_ticker] = {
                    "pnl": f"${asset_pnl:,.2f}",
                    "sharpe_ratio": round(asset_result['Sharpe Ratio'], 2),
                    "max_drawdown": f"{asset_result['Max. Drawdown [%]']:.2f}%",
                    "trades": asset_result['# Trades'],
                    "initial_capital": asset_initial,
                    "final_value": asset_final,
                    "return_pct": f"{asset_result['Return [%]']:.2f}%",
                    "weight": f"{(asset_final / final_value) * 100:.2f}%"
                }
            
            return portfolio_summary
        else:
            # Handle single asset results (existing code)
            asset_name = ticker if ticker else "N/A"
            final_value = results['Equity Final [$]']
            pnl = final_value - initial_capital
            
            return {
                "strategy": results._strategy.__class__.__name__,
                "asset": asset_name,
                "pnl": f"${pnl:,.2f}",
                "sharpe_ratio": round(results['Sharpe Ratio'], 2),
                "max_drawdown": f"{results['Max. Drawdown [%]']:.2f}%",
                "trades": results.get('# Trades', 0),
                "initial_capital": initial_capital,
                "final_value": final_value,
                "return_pct": f"{(pnl / initial_capital) * 100:.2f}%",
                "is_portfolio": False
            }

    @staticmethod
    def calculate_max_drawdown(strategy_instance):
        """Calculates maximum drawdown from the account value."""
        account_values = np.array([strategy_instance.broker.getvalue() for _ in range(len(strategy_instance))])
        peak = np.maximum.accumulate(account_values)
        drawdown = (peak - account_values) / peak * 100
        return drawdown.max()
