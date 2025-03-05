import os
import json
from typing import Dict, Any, Union
from jinja2 import Environment, FileSystemLoader, select_autoescape
import pandas as pd
from datetime import datetime  # Add this import

class ReportGenerator:
    """Generates HTML reports using Jinja templates."""

    TEMPLATE_DIR = "src/reports/templates"

    def __init__(self):
        self.env = Environment(
            loader=FileSystemLoader(self.TEMPLATE_DIR),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters for formatting
        self.env.filters['number_format'] = lambda value, precision=2: f"{float(value):,.{precision}f}"
        self.env.filters['currency'] = lambda value: f"${float(value):,.2f}"
        self.env.filters['percent'] = lambda value: f"{float(value):.2f}%"
        
        # Add global functions for templates
        self.env.globals['now'] = datetime.now  # Add the now() function
        
    def generate_report(self, data: Dict[str, Any], template_name: str, output_path: str) -> str:
        """
        Generates an HTML report from a template and data.
        
        Args:
            data: Dictionary containing report data
            template_name: Name of the template file
            output_path: Path where the report will be saved
            
        Returns:
            Path to the generated report file
        """
        try:
            # Handle data differently based on report type
            template_vars = self._prepare_template_variables(data, template_name)
            
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
            print(f"❌ Error generating report: {e}")
            self._generate_error_report(data, e, output_path)
            return output_path
    
    def _prepare_template_variables(self, data: Dict[str, Any], template_name: str) -> Dict[str, Any]:
        """
        Prepares variables for template rendering based on report type.
        
        Args:
            data: Original data dictionary
            template_name: Name of the template to prepare variables for
            
        Returns:
            Dictionary of variables ready for template rendering
        """
        # For multi-asset reports, we need to unpack the structure
        if template_name == "multi_asset_report.html":
            # Direct access to strategy and assets in template
            return {
                "strategy": data.get("strategy", "Unknown Strategy"),
                "assets": data.get("assets", {}),
                # Also include the original data for backward compatibility
                "data": data
            }
        
        # For standard reports, ensure required fields exist
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
        
        Args:
            data: The data that caused the error
            error: The exception that was raised
            output_path: Path where to save the error report
        """
        # Safely convert data to string for display
        try:
            data_str = json.dumps(data, indent=2, default=str)[:1000]  # Limit to 1000 chars
        except:
            data_str = str(data)[:1000]
            
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
