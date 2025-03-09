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
