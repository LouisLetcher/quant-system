import pandas as pd
import numpy as np
from datetime import datetime, timezone

class ReportFormatter:
    """Formats data for reports before generating them."""

    @staticmethod
    def format_backtest_results(results):
        """Formats backtest results into a readable structure."""
        # Extract the pnl value and ensure proper formatting
        pnl = results.get('pnl', 0)
        if isinstance(pnl, str):
            # If pnl is already a string (like "$0.00"), use it directly
            formatted_pnl = pnl
        else:
            # If pnl is a number, format it
            formatted_pnl = f"${pnl:,.2f}"

        return {
            "strategy": results.get("strategy", "N/A"),
            "asset": results.get("asset", "N/A"),
            "pnl": formatted_pnl,
            "sharpe_ratio": round(results.get("sharpe_ratio", 0), 2),
            "max_drawdown": f"{results.get('max_drawdown', 0) * 100:.2f}%" if isinstance(results.get('max_drawdown', 0), float) else results.get('max_drawdown', "0.00%"),
        }

    @staticmethod
    def format_optimization_results(results):
        """Formats optimization results into a structured list."""
        return [
            {
                "parameters": res.get("best_params", {}),
                "score": round(res.get("best_score", 0), 2)
            }
            for res in results
        ]

    @staticmethod
    def format_multiasset_results(results_dict, metric='sharpe_ratio'):
        """
        Formats results from multiple assets or strategies for reporting.
        
        Args:
            results_dict: Dictionary of results by asset/strategy
            metric: Performance metric used for comparison
        
        Returns:
            Formatted data structure for report template
        """
        formatted_data = {
            'strategies': {},
            'assets': {},
            'metric': metric,
            'comparison': []
        }
        
        # Determine if this is a multi-strategy or multi-asset result
        is_multi_strategy = 'is_multi_strategy' in next(iter(results_dict.values())) \
                            if results_dict else False
        
        for name, result in results_dict.items():
            # Extract the score based on the metric
            if metric == 'sharpe_ratio':
                score = result.get('sharpe_ratio', 0)
            elif metric == 'return':
                return_str = result.get('return_pct', '0%')
                # Extract numeric value from percentage
                score = float(return_str.replace('%', '')) if isinstance(return_str, str) else result.get('return_pct', 0)
            else:
                score = result.get(metric, 0)
            
            comparison_entry = {
                'name': name,
                'score': score,
                'pnl': result.get('pnl', '$0.00'),
                'trades': result.get('trades', 0)
            }
            
            formatted_data['comparison'].append(comparison_entry)
            
            if is_multi_strategy:
                formatted_data['strategies'][name] = ReportFormatter.format_backtest_results(result)
            else:
                formatted_data['assets'][name] = ReportFormatter.format_backtest_results(result)
        
        # Sort comparison by score
        formatted_data['comparison'] = sorted(
            formatted_data['comparison'], 
            key=lambda x: x['score'], 
            reverse=True
        )
        
        # Add metadata
        formatted_data['is_multi_strategy'] = is_multi_strategy
        formatted_data['is_multi_asset'] = not is_multi_strategy
        formatted_data['best_name'] = formatted_data['comparison'][0]['name'] if formatted_data['comparison'] else None
        formatted_data['best_score'] = formatted_data['comparison'][0]['score'] if formatted_data['comparison'] else 0
        
        return formatted_data
class ReportFormatter:
    """Formats data for reports before generating them."""

    @staticmethod
    def _convert_to_float(value):
        """Convert potential string percentages or values to float."""
        if isinstance(value, str):
            # Remove any '%' character and convert to float
            return float(value.replace('%', '').strip())
        return float(value)

    @staticmethod
    def prepare_portfolio_report_data(portfolio_results):
        """Prepare portfolio data for comprehensive HTML reporting.
        
        Args:
            portfolio_results: Dictionary of portfolio backtest results
            
        Returns:
            Formatted data structure for detailed report template
        """
        report_data = {
            'portfolio_name': portfolio_results.get('portfolio', 'Unknown Portfolio'),
            'description': portfolio_results.get('description', ''),
            'date_generated': datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'assets': [],
            'summary': {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0,
                'total_trades': 0,
                'win_rate': 0,
                'best_asset': '',
                'worst_asset': ''
            }
        }

        # Process each asset's results
        best_score = -float('inf')
        worst_score = float('inf')
        total_trades = 0
        winning_trades = 0
        total_return_pct = 0

        # Extract asset data from best_combinations or from assets
        asset_data = portfolio_results.get('best_combinations', portfolio_results.get('assets', {}))

        for ticker, data in asset_data.items():
            # Get the strategy results - structure varies between different result types
            if 'strategies' in data:
                # This is from _backtest_all_strategies output
                strategy_name = data.get('best_strategy', 'Unknown')
                strategy_data = data['strategies'].get(strategy_name, {})
                results = strategy_data.get('results', {})
                score = strategy_data.get('score', 0)
            else:
                # This is from direct strategy output or portfolio_optimal
                strategy_name = data.get('strategy', 'Unknown')
                results = data
                score = data.get('score', 0)

            # Format trades list
            trades_list = []
            if 'trades_list' in data:
                trades_list = data['trades_list']
            elif '_trades' in results:
                trades_df = results['_trades']
                for _, trade in trades_df.iterrows():
                    trade_dict = {
                        'entry_date': str(trade.get('EntryTime', '')),
                        'exit_date': str(trade.get('ExitTime', '')),
                        'type': 'LONG',  # Default to LONG
                        'entry_price': float(trade.get('EntryPrice', 0)),
                        'exit_price': float(trade.get('ExitPrice', 0)),
                        'size': int(trade.get('Size', 0)),
                        'pnl': float(trade.get('PnL', 0)),
                        'return_pct': float(trade.get('ReturnPct', 0)) * 100,
                        'duration': trade.get('Duration', '')
                    }
                    trades_list.append(trade_dict)

            # Process equity curve
            equity_curve = []
            if 'equity_curve' in data:
                equity_curve = data['equity_curve']
            elif '_equity_curve' in results:
                equity_data = results['_equity_curve']
                if isinstance(equity_data, pd.DataFrame):
                    for date, row in equity_data.iterrows():
                        equity_curve.append({
                            'date': str(date),
                            'value': float(row.iloc[0] if isinstance(row, pd.Series) else row)
                        })

            # Get metrics with fallbacks to different naming conventions
            win_rate = results.get('Win Rate [%]', data.get('win_rate', 0))
            total_trades += results.get('# Trades', data.get('trades', 0))
            return_pct = results.get('Return [%]', 0)
            if isinstance(return_pct, str):
                try:
                    return_pct = float(return_pct.replace('%', ''))
                except ValueError:
                    return_pct = 0
            total_return_pct += return_pct

            # Track best and worst assets
            if score > best_score:
                best_score = score
                report_data['summary']['best_asset'] = ticker
            if score < worst_score and score != -float('inf'):
                worst_score = score
                report_data['summary']['worst_asset'] = ticker

            # Create asset entry
            asset_entry = {
                'ticker': ticker,
                'strategy': strategy_name,
                'interval': data.get('interval', '1d'),
                'profit_factor': results.get('Profit Factor', data.get('profit_factor', 0)),
                'return_pct': return_pct,
                'sharpe_ratio': results.get('Sharpe Ratio', data.get('sharpe_ratio', 0)),
                'max_drawdown': results.get('Max. Drawdown [%]', data.get('max_drawdown', 0)),
                'win_rate': win_rate,
                'trades_count': results.get('# Trades', data.get('trades', 0)),
                'trades': trades_list,
                'equity_curve': equity_curve,
                'score': score
            }

            # Add to assets list
            report_data['assets'].append(asset_entry)

            # Calculate winning trades if win rate is available
            if win_rate and asset_entry['trades_count']:
                winning_trades += (win_rate / 100) * asset_entry['trades_count']

        # Calculate summary statistics
        if report_data['assets']:
            report_data['summary']['total_return'] = total_return_pct / len(report_data['assets'])
            report_data['summary']['total_trades'] = total_trades
            report_data['summary']['win_rate'] = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            # Calculate average metrics from all assets
            avg_metrics = {
                'sharpe_ratio': sum(asset.get('sharpe_ratio', 0) for asset in report_data['assets']) / len(report_data['assets']),
                'max_drawdown': sum(ReportFormatter._convert_to_float(asset.get('max_drawdown', 0)) for asset in report_data['assets']) / len(report_data['assets']),
                'profit_factor': sum(ReportFormatter._convert_to_float(asset.get('profit_factor', 0)) for asset in report_data['assets']) / len(report_data['assets'])
            }

            report_data['summary'].update(avg_metrics)

        return report_data