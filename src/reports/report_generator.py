import os
import json
import math
import copy
from typing import Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape
import pandas as pd
import numpy as np
from datetime import datetime

class ReportGenerator:
    """Generates HTML reports using Jinja templates."""

    TEMPLATE_DIR = "src/reports/templates"
    TEMPLATES = {
        'single_strategy': "backtest_report.html",
        'multi_strategy': "multi_strategy_report.html", 
        'portfolio': "multi_asset_report.html",
        'optimizer': "optimizer_report.html"
    }

    def __init__(self):
        self.env = Environment(
            loader=FileSystemLoader(self.TEMPLATE_DIR),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters for formatting
        self.env.filters['number_format'] = lambda value, precision=2: f"{float(value if pd.notna(value) else 0):,.{precision}f}"
        self.env.filters['currency'] = lambda value: f"${float(value if pd.notna(value) else 0):,.2f}"
        self.env.filters['percent'] = lambda value: f"{float(value if pd.notna(value) else 0):.2f}%"
        
        # Add global functions for templates
        self.env.globals['now'] = datetime.now
        self.env.globals['float'] = float  # Make float() available in templates
        self.env.globals['str'] = str      # Also add str() for good measure

    def generate_report(self, results: Dict[str, Any], template_name: str, output_path: str) -> str:
        try:
            # Deep copy to avoid modifying original data
            safe_results = copy.deepcopy(results)
        
            # Handle data differently based on report type
            template_vars = self._prepare_template_variables(safe_results, template_name)

            # Render template with prepared variables
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
            print(f"❌ Error generating report: {str(e)}")
            print(f"Template: {template_name}")
            import traceback
            print(traceback.format_exc())
            self._generate_error_report(results, e, output_path)
            return output_path
    def _prepare_template_variables(self, data: Dict[str, Any], template_name: str) -> Dict[str, Any]:
        # Deep clean any NaN values in the data
        def clean_special_values(obj):
            if isinstance(obj, dict):
                for k, v in list(obj.items()):
                    if isinstance(v, dict) or isinstance(v, list):
                        clean_special_values(v)
                    elif v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                        obj[k] = 0.0
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, dict) or isinstance(item, list):
                        clean_special_values(item)
                    elif item is None or (isinstance(item, float) and (math.isnan(item) or math.isinf(item))):
                        obj[i] = 0.0
        
        # Apply cleaning to the data
        clean_special_values(data)
        
        # Continue with your existing code...
        """
        Prepares variables for template rendering based on report type.
    
        Args:
            data: Original data dictionary
            template_name: Name of the template to prepare variables for
        
        Returns:
            Dictionary of variables ready for template rendering
        """
        # For portfolio reports (multi-asset)
        if template_name == self.TEMPLATES['portfolio']:
            return {
                "data": data,
                "strategy": data.get("strategy", "Unknown Strategy"),
                "is_portfolio": data.get("is_portfolio", True),
                "assets": data.get("assets", {}),
                "asset_details": data.get("asset_details", {}),
                "asset_list": data.get("asset_list", [])
            }
    
        # For multi-strategy reports
        elif template_name == self.TEMPLATES['multi_strategy']:
            return {
                "data": data,
                "strategy": data.get("strategy", "Unknown Strategy"),
                "asset": data.get("asset", "Unknown Asset"),
                "strategies": data.get("strategies", {}),
                "best_strategy": data.get("best_strategy", None),
                "best_score": data.get("best_score", 0),
                "metric": data.get("metric", "sharpe")
            }
        
        # For optimizer reports
        elif template_name == self.TEMPLATES['optimizer']:
            return {
                "strategy": data.get("strategy", "Unknown Strategy"),
                "ticker": data.get("ticker", None),
                "results": data.get("results", []),
                "metric": data.get("metric", "sharpe")
            }
    
        # For standard single-strategy reports
        result = {"data": data}
    
        # Add default values if needed
        if isinstance(data, dict):
            if 'trades' not in data:
                data['trades'] = data.get('trades', 0)
            
            if 'trades_list' not in data and 'trades' in data:
                data['trades_list'] = []
    
        return result    
    def _generate_error_report(self, data: Dict[str, Any], error: Exception, output_path: str) -> None:
        """
        Generates a simple HTML error report when template rendering fails.
        """
        # Create a copy of data to avoid modifying the original
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
                
            data_str = json.dumps(safe_data, indent=2, default=default_handler)[:1000]  # Limit to 1000 chars
        except Exception as json_error:
            data_str = f"Could not serialize data: {str(json_error)}\nData type: {type(safe_data)}"
        
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
            <p><strong>Error message:</strong> {str(error)}</p>
            <h2>Data Received:</h2>
            <pre>{data_str}</pre>
            <p>Check your template variables and data structure to resolve this issue.</p>
        </body>
        </html>
        """
    
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(error_html)
            
    def generate_backtest_report(self, backtest_results: Dict[str, Any], output_path: Optional[str] = None) -> str:
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
            ticker = backtest_results.get('asset', 'unknown')
            strategy = backtest_results.get('strategy', 'unknown')
            output_path = f"reports_output/backtest_{strategy}_{ticker}.html"
        
        # Generate the report
        return self.generate_report(backtest_results, self.TEMPLATES['single_strategy'], output_path)
    
    def generate_multi_strategy_report(self, comparison_data: Dict[str, Any], output_path: Optional[str] = None) -> str:
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
            ticker = comparison_data.get('asset', 'unknown')
            output_path = f"reports_output/all_strategies_{ticker}.html"
        
        # Generate the report
        return self.generate_report(comparison_data, self.TEMPLATES['multi_strategy'], output_path)
    
    def generate_portfolio_report(self, portfolio_results: Dict[str, Any], output_path: Optional[str] = None) -> str:
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
            portfolio_name = portfolio_results.get('portfolio', 'unknown')
            output_path = f"reports_output/portfolio_{portfolio_name}.html"
        
        # Generate the portfolio report
        return self.generate_report(portfolio_results, self.TEMPLATES['portfolio'], output_path)
        
    def generate_optimizer_report(self, optimizer_results: dict[str, Any], output_path: Optional[str] = None) -> str:
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
            strategy = optimizer_results.get('strategy', 'unknown')
            ticker = optimizer_results.get('ticker', 'unknown')
            output_path = f"reports_output/optimizer_{strategy}_{ticker}.html"
        
        # Generate the report
        return self.generate_report(optimizer_results, self.TEMPLATES['optimizer'], output_path)