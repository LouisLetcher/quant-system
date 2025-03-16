from __future__ import annotations

import copy
import json
import math
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape


class ReportGenerator:
    """Generates HTML reports using Jinja templates."""

    TEMPLATE_DIR = "src/reports/templates"
    TEMPLATES = {
        "single_strategy": "backtest_report.html",
        "multi_strategy": "multi_strategy_report.html",
        "portfolio": "multi_asset_report.html",
        "portfolio_detailed": "portfolio_detailed_report.html",
        "optimizer": "optimizer_report.html",
    }

    def __init__(self):
        self.env = Environment(
            loader=FileSystemLoader(self.TEMPLATE_DIR),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Add custom filters for formatting
        self.env.filters["number_format"] = lambda value, precision=2: (
            f"{float(value):,.{precision}f}"
            if isinstance(value, (int, float))
            or (isinstance(value, str) and value.replace(".", "").isdigit())
            else value
        )
        self.env.filters["currency"] = (
            lambda value: f"${float(value if pd.notna(value) else 0):,.2f}"
        )
        # More comprehensive filter handling
        self.env.filters["percent"] = lambda value: (
            f"{float(value if pd.notna(value) and not isinstance(value, str) else 0):.2f}%"
            if value != "N/A"
            else "N/A"
        )
        # Add global functions for templates
        self.env.globals["now"] = datetime.now
        self.env.globals["float"] = float  # Make float() available in templates
        self.env.globals["str"] = str  # Also add str() for good measure

    def generate_report(
        self, results: Dict[str, Any], template_name: str, output_path: str
    ) -> str:
        try:
            # Deep copy to avoid modifying original data
            safe_results = copy.deepcopy(results)

            # Pre-process best_combinations to fix common issues
            if "best_combinations" in safe_results:
                for ticker, combo in safe_results["best_combinations"].items():
                    # Fix trade count discrepancy
                    if "trades_list" in combo and (
                        combo.get("trades", 0) == 0 or combo.get("trades") is None
                    ):
                        combo["trades"] = len(combo["trades_list"])

            # Continue with existing code...
            template_vars = self._prepare_template_variables(
                safe_results, template_name
            )
            template = self.env.get_template(template_name)
            rendered_html = template.render(**template_vars)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Write output file
            with open(output_path, "w") as f:
                f.write(rendered_html)

            print(f"✅ Report successfully generated: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Error generating report: {e!s}")
            print(f"Template: {template_name}")
            import traceback

            print(traceback.format_exc())
            self._generate_error_report(results, e, output_path)
            return output_path

    def _prepare_template_variables(
        self, data: Dict[str, Any], template_name: str
    ) -> Dict[str, Any]:
        # Deep clean any NaN values in the data
        def clean_special_values(obj):
            if isinstance(obj, dict):
                for k, v in list(obj.items()):
                    if isinstance(v, dict) or isinstance(v, list):
                        clean_special_values(v)
                    elif v is None or (
                        isinstance(v, float) and (math.isnan(v) or math.isinf(v))
                    ):
                        obj[k] = 0.0
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, dict) or isinstance(item, list):
                        clean_special_values(item)
                    elif item is None or (
                        isinstance(item, float)
                        and (math.isnan(item) or math.isinf(item))
                    ):
                        obj[i] = 0.0

        # Apply cleaning to the data
        clean_special_values(data)

        # For detailed portfolio reports (new template type)
        if template_name == self.TEMPLATES["portfolio_detailed"]:
            # Direct passthrough of all variables from prepare_portfolio_report_data
            return data

        # For portfolio reports (multi-asset)
        if template_name == self.TEMPLATES["portfolio"]:
            return {
                "data": data,
                "strategy": data.get("strategy", "Unknown Strategy"),
                "is_portfolio": data.get("is_portfolio", True),
                "assets": data.get("assets", {}),
                "asset_details": data.get("asset_details", {}),
                "asset_list": data.get("asset_list", []),
            }

        # For multi-strategy reports
        if template_name == self.TEMPLATES["multi_strategy"]:
            return {
                "data": data,
                "strategy": data.get("strategy", "Unknown Strategy"),
                "asset": data.get("asset", "Unknown Asset"),
                "strategies": data.get("strategies", {}),
                "best_strategy": data.get("best_strategy"),
                "best_score": data.get("best_score", 0),
                "metric": data.get("metric", "sharpe"),
            }

        # For optimizer reports
        if template_name == self.TEMPLATES["optimizer"]:
            return {
                "strategy": data.get("strategy", "Unknown Strategy"),
                "ticker": data.get("ticker"),
                "results": data.get("results", []),
                "metric": data.get("metric", "sharpe"),
            }

        # When preparing portfolio optimal report data
        if template_name == "portfolio_optimal_report.html":
            for ticker, combinations in data.get("best_combinations", {}).items():
                # Create mappings from backtesting.py's naming to template's expected naming
                if (
                    "Profit Factor" in combinations
                    and "profit_factor" not in combinations
                ):
                    combinations["profit_factor"] = combinations["Profit Factor"]

                if "Return [%]" in combinations and "return" not in combinations:
                    combinations["return"] = f"{combinations['Return [%]']:.2f}%"

                if "Win Rate [%]" in combinations and "win_rate" not in combinations:
                    combinations["win_rate"] = combinations["Win Rate [%]"]

                if (
                    "Max. Drawdown [%]" in combinations
                    and "max_drawdown" not in combinations
                ):
                    combinations["max_drawdown"] = (
                        f"{combinations['Max. Drawdown [%]']:.2f}%"
                    )

                if "# Trades" in combinations and "trades" not in combinations:
                    combinations["trades"] = combinations["# Trades"]
                elif "trades_list" in combinations and (
                    "trades" not in combinations or combinations["trades"] == 0
                ):
                    combinations["trades"] = len(combinations["trades_list"])

        # For standard single-strategy reports
        result = {"data": data}

        # Add default values if needed
        if isinstance(data, dict):
            if "trades" not in data and "# Trades" in data:
                data["trades"] = data["# Trades"]

            if "trades_list" not in data and "trades" in data:
                data["trades_list"] = []

        return result

    def _generate_error_report(
        self, data: Dict[str, Any], error: Exception, output_path: str
    ) -> None:
        """
        Generates a simple HTML error report when template rendering fails.
        """  # Create a copy of data to avoid modifying the original
        safe_data = copy.deepcopy(data) if data is not None else {}

        # Recursively clean NaN values in the data
        def clean_nan_recursive(d):
            if isinstance(d, dict):
                for k, v in list(d.items()):
                    if isinstance(v, (np.ndarray, pd.Series)):
                        if pd.isna(v).any():
                            d[k] = 0.0
                    elif isinstance(v, (dict, list)):
                        clean_nan_recursive(v)
                    elif pd.isna(v):
                        d[k] = 0.0
            elif isinstance(d, list):
                for i, item in enumerate(d):
                    if isinstance(item, (np.ndarray, pd.Series)):
                        if pd.isna(item).any():
                            d[i] = 0.0
                    elif isinstance(item, (dict, list)):
                        clean_nan_recursive(item)
                    elif pd.isna(item):
                        d[i] = 0.0

        # Clean NaN values
        try:
            clean_nan_recursive(safe_data)
        except Exception as e:
            # If recursive cleaning fails, use a simpler approach
            print(f"⚠️ Error cleaning data: {e}, using empty data")
            safe_data = {"error": "Data could not be processed due to format issues"}

        # Safely convert data to string for display
        try:
            # Custom JSON encoder for handling undefined and other special types
            def default_handler(obj):
                try:
                    return str(obj)
                except:
                    return "UNSERIALIZABLE_OBJECT"

            data_str = json.dumps(safe_data, indent=2, default=default_handler)[
                :1000
            ]  # Limit to 1000 chars
        except Exception as json_error:
            data_str = f"Could not serialize data: {json_error!s}\nData type: {type(safe_data)}"

        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Report Generation Error</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 30px; line-height: 1.6; }}
                h1 {{ color: #d9534f; }}
                pre {{ background-color: #f5f5f5; padding: 15px; overflow: auto; }}
            </style>
        </head>
        <body>
            <h1>Error Generating Report</h1>
            <p><strong>Error message:</strong> {error!s}</p>
            <h2>Data Received:</h2>
            <pre>{data_str}</pre>
            <p>Check your template variables and data structure to resolve this issue.</p>
        </body>
        </html>
        """

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(error_html)

    def generate_backtest_report(
        self, backtest_results: Dict[str, Any], output_path: Optional[str] = None
    ) -> str:
        """
        Generates a single-strategy backtest HTML report.

        Args:
            backtest_results: Dictionary containing backtest results
            output_path: Path where the report will be saved (optional)

        Returns:
            Path to the generated report file
        """
        # Generate default output path if not provided
        if output_path is None:
            ticker = backtest_results.get("asset", "unknown")
            strategy = backtest_results.get("strategy", "unknown")
            output_path = f"reports_output/backtest_{strategy}_{ticker}.html"

        # Generate the report
        return self.generate_report(
            backtest_results, self.TEMPLATES["single_strategy"], output_path
        )

    def generate_multi_strategy_report(
        self, comparison_data: Dict[str, Any], output_path: Optional[str] = None
    ) -> str:
        """
        Generates a multi-strategy comparison report.

        Args:
            comparison_data: Dictionary with comparison results
            output_path: Path where the report will be saved (optional)

        Returns:
            Path to the generated report
        """
        # Generate default output path if not provided
        if output_path is None:
            ticker = comparison_data.get("asset", "unknown")
            output_path = f"reports_output/all_strategies_{ticker}.html"

        # Generate the report
        return self.generate_report(
            comparison_data, self.TEMPLATES["multi_strategy"], output_path
        )

    def format_report_data(self, backtest_results):
        """Pre-process data before passing to template to ensure proper formatting"""
        # Deep copy to avoid modifying original
        formatted_data = copy.deepcopy(backtest_results)

        # Ensure profit factor is properly formatted
        for ticker, combinations in formatted_data.get("best_combinations", {}).items():
            # Set trades count based on trades_list if necessary
            if "trades_list" in combinations and (
                combinations.get("trades", 0) == 0 or combinations.get("trades") is None
            ):
                combinations["trades"] = len(combinations["trades_list"])

            # Process profit factor
            if "profit_factor" in combinations:

                # Format depending on value size
                if combinations["profit_factor"] > 100:
                    combinations["profit_factor"] = round(
                        combinations["profit_factor"], 1
                    )
                elif combinations["profit_factor"] > 10:
                    combinations["profit_factor"] = round(
                        combinations["profit_factor"], 2
                    )
                else:
                    combinations["profit_factor"] = round(
                        combinations["profit_factor"], 4
                    )

            # Calculate total P&L from trades if not present
            if "total_pnl" not in combinations and "trades_list" in combinations:
                combinations["total_pnl"] = sum(
                    trade.get("pnl", 0) for trade in combinations["trades_list"]
                )

            # Ensure other key metrics exist
            if 'Return [%]' in combinations:
                combinations['return'] = f"{combinations['Return [%]']:.2f}%"
            elif 'return_pct' in combinations and isinstance(combinations['return_pct'], (int, float)):
                combinations['return'] = f"{combinations['return_pct']:.2f}%"
            elif 'return' not in combinations:
                combinations['return'] = '0.00%'
            if "win_rate" not in combinations and "trades_list" in combinations:
                trades = combinations["trades_list"]
                win_count = sum(1 for trade in trades if trade.get("pnl", 0) > 0)
                combinations["win_rate"] = (
                    round((win_count / len(trades) * 100), 2) if trades else 0
                )

            if "max_drawdown" not in combinations:
                combinations["max_drawdown"] = "0.00%"

            # Format trade data
            if "trades_list" in combinations:
                for trade in combinations["trades_list"]:
                    # Ensure date formatting
                    if "entry_date" in trade and not isinstance(
                        trade["entry_date"], str
                    ):
                        trade["entry_date"] = str(trade["entry_date"])
                    if "exit_date" in trade and not isinstance(trade["exit_date"], str):
                        trade["exit_date"] = str(trade["exit_date"])

        return formatted_data

    def generate_portfolio_report(
        self, portfolio_results: Dict[str, Any], output_path: Optional[str] = None
    ) -> str:
        """
        Generates a portfolio HTML report from backtest results.

        Args:
            portfolio_results: Dictionary containing portfolio backtest results
            output_path: Path where the report will be saved (optional)

        Returns:
            Path to the generated report file
        """
        # Generate default output path if not provided
        if output_path is None:
            portfolio_name = portfolio_results.get("portfolio", "unknown")
            output_path = f"reports_output/portfolio_{portfolio_name}.html"

        # Create asset_list for template if not present
        if (
            "best_combinations" in portfolio_results
            and "asset_list" not in portfolio_results
        ):
            asset_list = []
            for ticker, data in portfolio_results["best_combinations"].items():
                # Use direct mappings from backtesting.py metrics when available
                asset_data = {
                    "name": ticker,
                    "strategy": data.get("strategy", "Unknown"),
                    "interval": data.get("interval", "1d"),
                    "score": data.get("score", 0),
                    "profit_factor": data.get(
                        "Profit Factor", data.get("profit_factor", 0)
                    ),
                    "return": data.get("Return [%]", data.get("return", "0.00%")),
                    "total_pnl": data.get("Equity Final [$]", 0)
                    - data.get("initial_capital", 0),
                    "win_rate": data.get("Win Rate [%]", data.get("win_rate", 0)),
                    "max_drawdown": data.get(
                        "Max. Drawdown [%]", data.get("max_drawdown", "0.00%")
                    ),
                    "trades": data.get("# Trades", data.get("trades", 0)),
                }

                # Format numeric values appropriately
                if isinstance(asset_data["return"], (int, float)):
                    asset_data["return"] = f"{asset_data['return']:.2f}%"

                if isinstance(asset_data["max_drawdown"], (int, float)):
                    asset_data["max_drawdown"] = f"{asset_data['max_drawdown']:.2f}%"

                asset_list.append(asset_data)

            portfolio_results["asset_list"] = asset_list

        # Generate the report
        return self.generate_report(
            portfolio_results, self.TEMPLATES["portfolio"], output_path
        )

    def generate_optimizer_report(
        self, optimizer_results: dict[str, Any], output_path: Optional[str] = None
    ) -> str:
        """
        Generate an optimizer HTML report.

        Args:
            optimizer_results: Dictionary containing optimizer results
            output_path: Path where the report will be saved (optional)

        Returns:
            Path to the generated report file
        """
        # Generate default output path if not provided
        if output_path is None:
            strategy = optimizer_results.get("strategy", "unknown")
            ticker = optimizer_results.get("ticker", "unknown")
            output_path = f"reports_output/optimizer_{strategy}_{ticker}.html"

        # Generate the report
        return self.generate_report(
            optimizer_results, self.TEMPLATES["optimizer"], output_path
        )

    def generate_detailed_portfolio_report(self, portfolio_results, output_path=None):
        """
        Generates a comprehensive HTML report for portfolio backtest results
        with equity curves, drawdown charts, and detailed trade tables.

        Args:
            portfolio_results: Dictionary containing portfolio backtest results
            output_path: Path where the report will be saved (optional)

        Returns:
            Path to the generated report file
        """
        # Generate default output path if not provided
        if output_path is None:
            portfolio_name = portfolio_results.get("portfolio", "unknown")
            output_path = f"reports_output/portfolio_detailed_{portfolio_name}.html"

        # Prepare data for the report
        from datetime import datetime
        
        # Format the data for reporting
        report_data = self._prepare_portfolio_detailed_report_data(portfolio_results)
        
        # Generate the report
        return self.generate_report(
            report_data, self.TEMPLATES["portfolio_detailed"], output_path
        )

    def _prepare_portfolio_detailed_report_data(self, portfolio_results):
        """
        Prepare data for the detailed portfolio report.
        
        Args:
            portfolio_results: Raw portfolio results
            
        Returns:
            Formatted data for the template
        """
        from datetime import datetime
        
        # Basic portfolio information
        report_data = {
            "portfolio_name": portfolio_results.get("portfolio", "Unknown Portfolio"),
            "description": portfolio_results.get("description", ""),
            "date_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metric": portfolio_results.get("metric", "sharpe"),
            "assets": [],
            "summary": {
                "total_return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "profit_factor": 0,
                "total_trades": 0,
                "win_rate": 0
            }
        }
        
        # Process each asset's results
        best_combinations = portfolio_results.get("best_combinations", {})
        all_results = portfolio_results.get("all_results", {})
        
        total_return = 0
        total_trades = 0
        weighted_sharpe = 0
        max_drawdown = 0
        total_profit_factor = 0
        total_win_rate = 0
        asset_count = 0
        
        # Process each asset
        for ticker, best_combo in best_combinations.items():
            if best_combo is None or best_combo.get("strategy") is None:
                continue
                
            asset_count += 1
            
            # Get the asset's detailed results
            asset_results = all_results.get(ticker, {})
            
            # Create asset entry for the report
            asset_entry = {
                "ticker": ticker,
                "strategy": best_combo.get("strategy", "Unknown"),
                "interval": best_combo.get("interval", "1d"),
                "return_pct": best_combo.get("return_pct", 0),
                "sharpe_ratio": best_combo.get("sharpe_ratio", 0),
                "max_drawdown": best_combo.get("max_drawdown_pct", 0),
                "win_rate": best_combo.get("win_rate", 0),
                "trades_count": best_combo.get("trades_count", 0),
                "profit_factor": best_combo.get("profit_factor", 0),
                "equity_curve": best_combo.get("equity_curve", []),
                "equity_chart": None,  # Will be populated later if visualization is added
                "strategies": []
            }
            
            # Accumulate summary statistics
            total_return += best_combo.get("return_pct", 0)
            total_trades += best_combo.get("trades_count", 0)
            weighted_sharpe += best_combo.get("sharpe_ratio", 0)
            max_drawdown = max(max_drawdown, best_combo.get("max_drawdown_pct", 0))
            total_profit_factor += best_combo.get("profit_factor", 0)
            total_win_rate += best_combo.get("win_rate", 0)
            
            # Process strategies for this asset
            if "strategies" in asset_results:
                for strategy_data in asset_results["strategies"]:
                    strategy_entry = {
                        "name": strategy_data.get("name", "Unknown"),
                        "best_timeframe": strategy_data.get("best_timeframe", ""),
                        "best_score": strategy_data.get("best_score", 0),
                        "timeframes": []
                    }
                    
                    # Process timeframes for this strategy
                    for timeframe_data in strategy_data.get("timeframes", []):
                        # Skip entries with errors
                        if "error" in timeframe_data:
                            continue
                            
                        timeframe_entry = {
                            "interval": timeframe_data.get("interval", ""),
                            "return_pct": timeframe_data.get("return_pct", 0),
                            "sharpe_ratio": timeframe_data.get("sharpe_ratio", 0),
                            "max_drawdown": timeframe_data.get("max_drawdown_pct", 0),
                            "win_rate": timeframe_data.get("win_rate", 0),
                            "trades_count": timeframe_data.get("trades_count", 0),
                            "profit_factor": timeframe_data.get("profit_factor", 0),
                            "equity_curve": timeframe_data.get("equity_curve", []),
                            "equity_chart": None,  # Will be populated if visualization is added
                            "trades": timeframe_data.get("trades", [])
                        }
                        
                        strategy_entry["timeframes"].append(timeframe_entry)
                    
                    asset_entry["strategies"].append(strategy_entry)
            
            report_data["assets"].append(asset_entry)
        
        # Calculate portfolio summary
        if asset_count > 0:
            report_data["summary"] = {
                "total_return": total_return / asset_count,
                "sharpe_ratio": weighted_sharpe / asset_count,
                "max_drawdown": max_drawdown,
                "profit_factor": total_profit_factor / asset_count,
                "total_trades": total_trades,
                "win_rate": total_win_rate / asset_count
            }
        
        # Add visualization capabilities if needed
        try:
            import matplotlib.pyplot as plt
            import io
            import base64
            from matplotlib.dates import DateFormatter
            import pandas as pd
            import numpy as np
            from datetime import datetime
            
            # Function to generate equity curve and drawdown visualizations
            def generate_equity_chart(equity_curve_data):
                if not equity_curve_data or len(equity_curve_data) < 2:
                    return None
                    
                # Convert equity curve data to DataFrame
                dates = []
                values = []
                
                for point in equity_curve_data:
                    try:
                        date = datetime.strptime(point["date"], "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        try:
                            date = datetime.strptime(point["date"], "%Y-%m-%d")
                        except ValueError:
                            continue
                            
                    dates.append(date)
                    values.append(point["value"])
                    
                if not dates or not values:
                    return None
                    
                equity_df = pd.DataFrame({"equity": values}, index=dates)
                
                # Calculate drawdown
                equity_df["peak"] = equity_df["equity"].cummax()
                equity_df["drawdown"] = (equity_df["equity"] / equity_df["peak"] - 1) * 100
                
                # Create the plot
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]})
                
                # Plot equity curve
                ax1.plot(equity_df.index, equity_df["equity"], label="Equity", color="#2980b9")
                ax1.set_title("Equity Curve")
                ax1.set_ylabel("Equity ($)")
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Plot drawdown
                ax2.fill_between(equity_df.index, 0, equity_df["drawdown"], color="#e74c3c", alpha=0.3)
                ax2.plot(equity_df.index, equity_df["drawdown"], color="#e74c3c")
                ax2.set_title("Drawdown")
                ax2.set_ylabel("Drawdown (%)")
                ax2.set_ylim(bottom=min(equity_df["drawdown"].min() * 1.1, -1), top=1)
                ax2.grid(True, alpha=0.3)
                
                # Format x-axis dates
                date_format = DateFormatter("%Y-%m-%d")
                ax1.xaxis.set_major_formatter(date_format)
                ax2.xaxis.set_major_formatter(date_format)
                
                plt.tight_layout()
                
                # Convert plot to base64 string
                buffer = io.BytesIO()
                plt.savefig(buffer, format="png")
                buffer.seek(0)
                image_png = buffer.getvalue()
                buffer.close()
                plt.close(fig)
                
                return base64.b64encode(image_png).decode("utf-8")
            
            # Generate charts for each asset and strategy timeframe
            for asset in report_data["assets"]:
                # Generate chart for the best combination
                if asset["equity_curve"]:
                    asset["equity_chart"] = generate_equity_chart(asset["equity_curve"])
                    
                # Generate charts for each strategy timeframe
                for strategy in asset["strategies"]:
                    for timeframe in strategy["timeframes"]:
                        if timeframe["equity_curve"]:
                            timeframe["equity_chart"] = generate_equity_chart(timeframe["equity_curve"])
        
        except ImportError:
            print("⚠️ Matplotlib not available. Equity charts will not be generated.")
        except Exception as e:
            print(f"⚠️ Error generating equity charts: {e}")
        
        return report_data