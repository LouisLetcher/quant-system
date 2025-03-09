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
                "final_value": initial_capital,
                "suspicious_result": False
            }
        
        # Check if this is a portfolio result
        if results.get('_portfolio', False):
            # Process portfolio results
            portfolio_name = ticker if ticker else "Portfolio"
            final_value = results['Equity Final [$]']
            pnl = final_value - initial_capital
            
            # Add validation for suspicious results
            return_pct = (pnl / initial_capital) * 100
            trade_count = results['# Trades']
            
            # Flag unrealistic results
            if trade_count == 0 and return_pct > 1000:  # Over 1000% with no trades
                print(f"⚠️ WARNING: Strategy shows {return_pct:.2f}% return with 0 trades. Results may be unreliable.")
                suspicious = True
            else:
                suspicious = False
            
            # Create main portfolio summary
            portfolio_summary = {
                "strategy": results['_strategy'].__name__,
                "asset": portfolio_name,
                "pnl": f"${pnl:,.2f}",
                "sharpe_ratio": round(results['Sharpe Ratio'], 2),
                "max_drawdown": f"{results['Max. Drawdown [%]']:.2f}%",
                "trades": trade_count,
                "initial_capital": initial_capital,
                "final_value": final_value,
                "return_pct": f"{results['Return [%]']:.2f}%",
                "is_portfolio": True,
                "assets": results['_assets'],
                "asset_details": {},
                "suspicious_result": suspicious
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
            
            # Add validation for suspicious results
            return_pct = (pnl / initial_capital) * 100
            trade_count = results.get('# Trades', 0)
            
            # Flag unrealistic results
            if trade_count == 0 and return_pct > 1000:  # Over 1000% with no trades
                print(f"⚠️ WARNING: Strategy shows {return_pct:.2f}% return with 0 trades. Results may be unreliable.")
                suspicious = True
            else:
                suspicious = False
            
            return {
                "strategy": results._strategy.__class__.__name__,
                "asset": asset_name,
                "pnl": f"${pnl:,.2f}",
                "sharpe_ratio": round(results['Sharpe Ratio'], 2),
                "max_drawdown": f"{results['Max. Drawdown [%]']:.2f}%",
                "trades": trade_count,
                "initial_capital": initial_capital,
                "final_value": final_value,
                "return_pct": f"{return_pct:.2f}%",
                "is_portfolio": False,
                "suspicious_result": suspicious
            }

    @staticmethod
    def calculate_max_drawdown(strategy_instance):
        """Calculates maximum drawdown from the account value."""
        account_values = np.array([strategy_instance.broker.getvalue() for _ in range(len(strategy_instance))])
        peak = np.maximum.accumulate(account_values)
        drawdown = (peak - account_values) / peak * 100
        return drawdown.max()

    @staticmethod
    def compare_strategies(results_dict, metric='sharpe_ratio'):
        """
        Compares results from multiple strategies.
        
        Args:
            results_dict: Dictionary mapping strategy names to results
            metric: Metric to compare ('sharpe_ratio', 'pnl', etc.)
        
        Returns:
            Dictionary with best strategy and comparison data
        """
        if not results_dict:
            return {'best_strategy': None, 'best_score': 0, 'comparison': {}}
        
        comparison = {}
        best_score = -float('inf')
        best_strategy = None
        
        for strategy_name, results in results_dict.items():
            # Extract metric value (handle both numeric and string formats)
            if metric == 'sharpe_ratio':
                score = results.get('sharpe_ratio', 0)
                if isinstance(score, str):
                    try:
                        score = float(score)
                    except ValueError:
                        score = 0
            elif metric == 'pnl':
                pnl_str = results.get('pnl', '$0.00')
                try:
                    score = float(pnl_str.replace('$', '').replace(',', ''))
                except ValueError:
                    score = 0
            else:
                score = results.get(metric, 0)
                
            comparison[strategy_name] = score
            
            if score > best_score:
                best_score = score
                best_strategy = strategy_name
        
        return {
            'best_strategy': best_strategy,
            'best_score': best_score,
            'comparison': comparison
        }
