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
