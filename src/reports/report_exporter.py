import json
import pandas as pd
import os

class ReportExporter:
    """Handles exporting reports in various formats."""

    @staticmethod
    def export_to_json(data: dict, file_path: str):
        """Exports data to JSON format."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Ensure trade counts are included in the report data
        if isinstance(data, dict) and 'trades' not in data and 'results' in data:
            data['trades'] = data['results'].get('# Trades', 0)
            
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        return file_path

    @staticmethod
    def export_to_csv(data: list, file_path: str):
        """Exports data to CSV format."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Convert to DataFrame for CSV export
        df = pd.DataFrame(data)
        
        # Ensure trade count is included in the output
        if 'trades' not in df.columns and len(data) > 0 and isinstance(data[0], dict):
            if 'results' in data[0]:
                df['trades'] = [item['results'].get('# Trades', 0) for item in data]
            elif 'trades' not in data[0] and isinstance(data[0], dict):
                df['trades'] = [0] * len(data)  # Default value if not found
        
        df.to_csv(file_path, index=False)
        return file_path

    @staticmethod
    def export_to_html(data: list, file_path: str):
        """Exports data to HTML format."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Convert to DataFrame for HTML export
        df = pd.DataFrame(data)
        
        # Ensure trade counts are visible in the report
        if 'trades' not in df.columns and len(data) > 0:
            if isinstance(data[0], dict) and 'results' in data[0]:
                df['trades'] = [item['results'].get('# Trades', 0) for item in data]
            elif isinstance(data[0], dict) and 'trades' not in data[0]:
                df['trades'] = [0] * len(data)  # Default value if not found
        
        # Format the HTML with better styling
        html = df.to_html(index=False)
        
        with open(file_path, "w") as f:
            f.write(f"""
            <html>
            <head>
                <style>
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    th {{ background-color: #4CAF50; color: white; }}
                </style>
            </head>
            <body>
                <h1>Backtest Results</h1>
                {html}
            </body>
            </html>
            """)
        
        return file_path