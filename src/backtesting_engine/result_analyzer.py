import numpy as np
import pandas as pd
from datetime import datetime

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
                "suspicious_result": False,
                "win_rate": 0,
                "profit_factor": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "equity_curve": [],
                "drawdown_curve": [],
                "trades_list": []
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
            
            # Extract additional metrics
            win_rate = results.get('Win Rate [%]', 0)
            profit_factor = results.get('Profit Factor', 0)
            avg_win = results.get('Best Trade [%]', 0)
            avg_loss = results.get('Worst Trade [%]', 0)
            
            # Extract equity curve data if available
            equity_curve = BacktestResultAnalyzer._extract_equity_curve(results)
            drawdown_curve = BacktestResultAnalyzer._extract_drawdown_curve(results)
            trades_list = BacktestResultAnalyzer._extract_trades_list(results)
            
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
                "suspicious_result": suspicious,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "equity_curve": equity_curve,
                "drawdown_curve": drawdown_curve,
                "trades_list": trades_list
            }
            
            # Add individual asset details
            for asset_ticker, asset_result in results['asset_results'].items():
                asset_final = asset_result['Equity Final [$]']
                asset_initial = initial_capital / len(results['_assets'])
                asset_pnl = asset_final - asset_initial

                # Extract additional metrics for this asset
                asset_win_rate = asset_result.get('Win Rate [%]', 0)
                asset_profit_factor = asset_result.get('Profit Factor', 0)
                asset_avg_win = asset_result.get('Best Trade [%]', 0)
                asset_avg_loss = asset_result.get('Worst Trade [%]', 0)

                # Extract asset-specific curves and trades
                asset_equity_curve = BacktestResultAnalyzer._extract_equity_curve(asset_result)
                asset_drawdown_curve = BacktestResultAnalyzer._extract_drawdown_curve(asset_result)
                asset_trades_list = BacktestResultAnalyzer._extract_trades_list(asset_result)
                tv_profit_factor = BacktestResultAnalyzer.calculate_tradingview_profit_factor(trades_list)

                portfolio_summary["asset_details"][asset_ticker] = {
                    "pnl": f"${asset_pnl:,.2f}",
                    "sharpe_ratio": round(asset_result['Sharpe Ratio'], 2),
                    "max_drawdown": f"{asset_result['Max. Drawdown [%]']:.2f}%",
                    "trades": asset_result['# Trades'],
                    "initial_capital": asset_initial,
                    "final_value": asset_final,
                    "return_pct": f"{asset_result['Return [%]']:.2f}%",
                    "weight": f"{(asset_final / final_value) * 100:.2f}%",
                    "win_rate": asset_win_rate,
                    "profit_factor": asset_profit_factor,
                    "tv_profit_factor": round(tv_profit_factor, 2),
                    "avg_win": asset_avg_win,
                    "avg_loss": asset_avg_loss,
                    "equity_curve": asset_equity_curve,
                    "drawdown_curve": asset_drawdown_curve,
                    "trades_list": asset_trades_list
                }

            return portfolio_summary
        else:
            # Handle single asset results
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
            
            # Extract additional metrics
            win_rate = results.get('Win Rate [%]', 0)
            profit_factor = results.get('Profit Factor', 0)
            avg_win = results.get('Best Trade [%]', 0)
            avg_loss = results.get('Worst Trade [%]', 0)
            
            # Extract equity curve and trades data
            equity_curve = BacktestResultAnalyzer._extract_equity_curve(results)
            drawdown_curve = BacktestResultAnalyzer._extract_drawdown_curve(results)
            trades_list = BacktestResultAnalyzer._extract_trades_list(results)
            tv_profit_factor = BacktestResultAnalyzer.calculate_tradingview_profit_factor(trades_list)
            
            return {
                "strategy": results._strategy.__class__.__name__,
                "asset": asset_name,
                "pnl": f"${pnl:,.2f}",
                "sharpe_ratio": round(results['Sharpe Ratio'], 2),
                "profit_factor": profit_factor,
                "tv_profit_factor": round(tv_profit_factor, 2),
                "max_drawdown": f"{results['Max. Drawdown [%]']:.2f}%",
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

    @staticmethod
    def _extract_equity_curve(results):
        """Extract equity curve data from results if available."""
        try:
            # For Backtesting.py _Stats object
            if hasattr(results, '_equity_curve') and hasattr(results, '_strategy'):
                # In backtesting.py, the equity curve is often stored in the strategy object
                equity_data = []
                
                # Extract the equity curve from the _equity_curve attribute if it exists
                if hasattr(results, '_equity_curve') and isinstance(results._equity_curve, pd.DataFrame):
                    for date, equity in zip(results._equity_curve.index, results._equity_curve['Equity']):
                        date_str = date.strftime('%Y-%m-%d') if isinstance(date, pd.Timestamp) else str(date)
                        equity_data.append({
                            'date': date_str,
                            'value': float(equity)
                        })
                    return equity_data
                    
            # Backtesting.py may also store equity curve directly in the results object
            if hasattr(results, 'equity_curve') and isinstance(results.equity_curve, pd.DataFrame):
                equity_data = []
                
                if 'Equity' in results.equity_curve.columns:
                    for date, row in results.equity_curve.iterrows():
                        date_str = date.strftime('%Y-%m-%d') if isinstance(date, pd.Timestamp) else str(date)
                        equity_data.append({
                            'date': date_str,
                            'value': float(row['Equity'])
                        })
                    return equity_data
                    
            # If we're dealing with backtesting._stats._Stats
            # Try to access the internal _df attribute which often contains equity data
            if hasattr(results, '_df') and isinstance(results._df, pd.DataFrame):
                equity_data = []
                
                # Check if there's an 'Equity' column
                if 'Equity' in results._df.columns:
                    for date, row in results._df.iterrows():
                        date_str = date.strftime('%Y-%m-%d') if isinstance(date, pd.Timestamp) else str(date)
                        equity_data.append({
                            'date': date_str,
                            'value': float(row['Equity'])
                        })
                    return equity_data
                
            # Last resort: Generate synthetic equity curve from trade data
            if hasattr(results, '_trades') and isinstance(results._trades, (list, pd.DataFrame)):
                # If we have trade data but no equity curve, we can reconstruct a rough equity curve
                trades_list = []
                initial_equity = results.get('initial_capital', 10000)
                
                if isinstance(results._trades, pd.DataFrame):
                    trades_df = results._trades
                    
                    # Sort trades by entry date
                    trades_df = trades_df.sort_values('EntryTime')
                    
                    equity = initial_equity
                    equity_data = [{'date': trades_df.iloc[0]['EntryTime'].strftime('%Y-%m-%d'), 'value': equity}]
                    
                    for _, trade in trades_df.iterrows():
                        equity += trade['PnL']
                        exit_date = trade['ExitTime'].strftime('%Y-%m-%d') if hasattr(trade['ExitTime'], 'strftime') else str(trade['ExitTime'])
                        equity_data.append({
                            'date': exit_date,
                            'value': float(equity)
                        })
                    
                    return equity_data
                
            # If we can't find equity curve data in any format
            return []
            
        except Exception as e:
            print(f"⚠️ Error extracting equity curve: {e}")
            return []


    @staticmethod
    def _extract_drawdown_curve(results):
        """Extract drawdown curve data from results if available."""
        try:
            # For Backtesting.py _Stats object
            if hasattr(results, '_equity_curve') and isinstance(results._equity_curve, pd.DataFrame):
                drawdown_data = []
                
                # Check for DrawdownPct column (standard in backtesting.py)
                if 'DrawdownPct' in results._equity_curve.columns:
                    for date, dd in zip(results._equity_curve.index, results._equity_curve['DrawdownPct']):
                        date_str = date.strftime('%Y-%m-%d') if isinstance(date, pd.Timestamp) else str(date)
                        drawdown_data.append({
                            'date': date_str,
                            'value': float(dd * 100)  # Convert to percentage
                        })
                    return drawdown_data
            
            # Try accessing the internal _df attribute
            if hasattr(results, '_df') and isinstance(results._df, pd.DataFrame):
                drawdown_data = []
                
                # Check for DrawdownPct column
                if 'DrawdownPct' in results._df.columns:
                    for date, row in results._df.iterrows():
                        date_str = date.strftime('%Y-%m-%d') if isinstance(date, pd.Timestamp) else str(date)
                        drawdown_data.append({
                            'date': date_str,
                            'value': float(row['DrawdownPct'] * 100)  # Convert to percentage
                        })
                    return drawdown_data
            
            # If we can't find drawdown data
            return []
            
        except Exception as e:
            print(f"⚠️ Error extracting drawdown curve: {e}")
            return []

    @staticmethod
    def calculate_tradingview_profit_factor(trades_list):
        """Calculate profit factor using TradingView's method."""
        gross_profit = sum(trade['pnl'] for trade in trades_list if trade['pnl'] > 0)
        gross_loss = sum(abs(trade['pnl']) for trade in trades_list if trade['pnl'] < 0)
        
        if gross_loss == 0:
            return float('inf')  # Avoid division by zero
        
        return gross_profit / gross_loss

    @staticmethod
    def _extract_trades_list(results):
        """Extract list of trades from the results if available."""
        try:
            # For Backtesting.py _stats object
            if hasattr(results, '_trades'):
                # Check if _trades is a DataFrame (common in backtesting.py)
                if isinstance(results._trades, pd.DataFrame):
                    trades_list = []
                    
                    # Iterate through trades DataFrame safely
                    for i, trade in results._trades.iterrows():
                        trade_data = {
                            'id': i + 1,
                            'type': 'LONG' if trade.get('Size', 0) > 0 else 'SHORT',
                            'entry_date': str(trade.get('EntryTime', 'N/A')),
                            'exit_date': str(trade.get('ExitTime', 'N/A')),
                            'entry_price': float(trade.get('EntryPrice', 0)),
                            'exit_price': float(trade.get('ExitPrice', 0)),
                            'size': abs(float(trade.get('Size', 0))),
                            'pnl': float(trade.get('PnL', 0)),
                            'return_pct': float(trade.get('ReturnPct', 0)),
                            'duration': str(trade.get('Duration', 'N/A'))
                        }
                        trades_list.append(trade_data)
                    
                    return trades_list
                
                # If _trades is a list
                elif isinstance(results._trades, list):
                    trades_list = []
                    
                    for i, trade in enumerate(results._trades):
                        trade_data = {
                            'id': i + 1,
                            'type': trade.get('type', 'N/A'),
                            'entry_date': str(trade.get('entry_time', 'N/A')),
                            'exit_date': str(trade.get('exit_time', 'N/A')),
                            'entry_price': float(trade.get('entry_price', 0)),
                            'exit_price': float(trade.get('exit_price', 0)),
                            'size': float(trade.get('size', 0)),
                            'pnl': float(trade.get('pnl', 0)),
                            'return_pct': float(trade.get('return_pct', 0)),
                            'duration': trade.get('duration', 'N/A')
                        }
                        trades_list.append(trade_data)
                    
                    return trades_list
            
            # Check for _trades attribute from Backtesting.py
            elif hasattr(results, '_trades') and results._trades:
                trades_list = []
                
                for i, trade in enumerate(results._trades):
                    # Format depends on the backtesting library
                    trade_data = {
                        'id': i + 1,
                        'type': trade.get('type', 'N/A'),
                        'entry_date': trade.get('entry_time', 'N/A'),
                        'exit_date': trade.get('exit_time', 'N/A'),
                        'entry_price': float(trade.get('entry_price', 0)),
                        'exit_price': float(trade.get('exit_price', 0)),
                        'size': float(trade.get('size', 0)),
                        'pnl': float(trade.get('pnl', 0)),
                        'return_pct': float(trade.get('return_pct', 0)),
                        'duration': trade.get('duration', 'N/A')
                    }
                    trades_list.append(trade_data)
                
                return trades_list
            
            # If trades are stored in a different format, try to extract them
            elif hasattr(results, 'orders') and results.orders:
                # Convert orders to trades (simplified)
                trades_list = []
                for i, order in enumerate(results.orders):
                    # Basic info extraction - adapt to your specific format
                    trades_list.append({
                        'id': i + 1,
                        'type': order.get('side', 'N/A'),
                        'entry_date': order.get('entry_date', 'N/A'),
                        'exit_date': order.get('exit_date', 'N/A'),
                        'entry_price': float(order.get('entry_price', 0)),
                        'exit_price': float(order.get('exit_price', 0)),
                        'size': float(order.get('size', 0)),
                        'pnl': float(order.get('pnl', 0)),
                        'return_pct': float(order.get('return_pct', 0)),
                        'duration': order.get('duration', 'N/A')
                    })
                return trades_list
            
            # If we can't find trade data in any expected format
            return []
            
        except Exception as e:
            print(f"⚠️ Error extracting trades list: {e}")
            return []

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
            metric: Metric to compare ('sharpe_ratio', 'profit_factor', 'pnl', etc.)
        
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
            elif metric == 'profit_factor':
                # Properly extract profit factor from results
                score = results.get('profit_factor', 0)
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
