import json
import pandas as pd
import os
from typing import Dict, Any, List, Optional, Union

class ReportExporter:
    """Handles exporting reports in various formats."""

    @staticmethod
    def export_to_json(data: Dict[str, Any], file_path: str) -> str:
        """Exports data to JSON format."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Ensure trade counts are included in the report data
        if isinstance(data, dict) and 'trades' not in data and 'results' in data:
            data['trades'] = data['results'].get('# Trades', 0)
            
        with open(file_path, "w") as f:
            # Custom serialization function to handle non-serializable objects
            def json_serializer(obj):
                try:
                    return str(obj)
                except:
                    return "UNSERIALIZABLE_OBJECT"
            
            json.dump(data, f, indent=4, default=json_serializer)
        return file_path

    @staticmethod
    def export_to_csv(data: Union[List, Dict], file_path: str) -> str:
        """Exports data to CSV format."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Convert dict to list for DataFrame if needed
        if isinstance(data, dict):
            if 'asset_list' in data:
                # Use asset_list for portfolio reports
                csv_data = data['asset_list']
            elif 'results' in data and isinstance(data['results'], list):
                # Use results for optimizer reports
                csv_data = data['results']
            else:
                # Convert dict to list of dicts for other reports
                csv_data = [{'key': k, 'value': v} for k, v in data.items()]
        else:
            csv_data = data
        
        # Convert to DataFrame for CSV export
        df = pd.DataFrame(csv_data)
        
        # Ensure trade count is included in the output
        if isinstance(csv_data, list) and len(csv_data) > 0:
            if 'trades' not in df.columns and isinstance(csv_data[0], dict):
                if 'results' in csv_data[0]:
                    df['trades'] = [item['results'].get('# Trades', 0) for item in csv_data]
                elif 'trades' not in csv_data[0]:
                    df['trades'] = [0] * len(csv_data)  # Default value if not found
        
        df.to_csv(file_path, index=False)
        return file_path

    @staticmethod
    def export_to_html(data: Union[List, Dict], file_path: str) -> str:
        """Exports data to HTML format."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Convert dict to list for DataFrame if needed
        if isinstance(data, dict):
            if 'asset_list' in data:
                # Use asset_list for portfolio reports
                df_data = data['asset_list']
            elif 'results' in data and isinstance(data['results'], list):
                # Use results for optimizer reports
                df_data = data['results']
            else:
                # For single strategy reports, create a summary table
                df_data = [{
                    'Metric': k,
                    'Value': v
                } for k, v in data.items() if not isinstance(v, (dict, list))]
        else:
            df_data = data
        
        # Convert to DataFrame for HTML export
        df = pd.DataFrame(df_data)
        
        # Format the HTML with better styling
        html = df.to_html(index=False)
        
        report_title = "Backtest Results"
        if isinstance(data, dict):
            if 'strategy' in data and 'asset' in data:
                report_title = f"Backtest Results: {data['strategy']} on {data['asset']}"
            elif 'portfolio' in data:
                report_title = f"Portfolio Results: {data['portfolio']}"
            elif 'is_multi_strategy' in data:
                report_title = f"Strategy Comparison: {data.get('asset', 'Multiple Assets')}"
        
        with open(file_path, "w") as f:
            f.write(f"""
            <html>
            <head>
                <title>{report_title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 30px; line-height: 1.6; }}
                    h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                                        tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    th {{ background-color: #3498db; color: white; }}
                    .positive {{ color: #27ae60; font-weight: bold; }}
                    .negative {{ color: #e74c3c; font-weight: bold; }}
                </style>
            </head>
            <body>
                <h1>{report_title}</h1>
                {html}
                <footer style="margin-top: 50px; border-top: 1px solid #eee; padding-top: 20px; text-align: center; color: #7f8c8d; font-size: 12px;">
                    <p>Generated by Quant Trading System on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </footer>
            </body>
            </html>
            """)
        
        return file_path
    
    @staticmethod
    def export_report(data: Dict[str, Any], file_prefix: str, formats: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Exports report data in multiple formats.
        
        Args:
            data: Report data
            file_prefix: Prefix for output filenames
            formats: List of formats to export ('json', 'csv', 'html')
        
        Returns:
            Dictionary mapping formats to file paths
        """
        if formats is None:
            formats = ['json', 'html']
        
        output_files = {}
        os.makedirs('reports_output', exist_ok=True)
        
        for fmt in formats:
            if fmt == 'json':
                path = f"reports_output/{file_prefix}.json"
                ReportExporter.export_to_json(data, path)
                output_files['json'] = path
            elif fmt == 'csv':
                path = f"reports_output/{file_prefix}.csv"
                ReportExporter.export_to_csv(data, path)
                output_files['csv'] = path
            elif fmt == 'html':
                path = f"reports_output/{file_prefix}.html"
                ReportExporter.export_to_html(data, path)
                output_files['html'] = path
        
        return output_files
    
    @staticmethod
    def export_optimization_results(results: Dict[str, Any], file_prefix: str, formats: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Exports optimization results in multiple formats.
        
        Args:
            results: Optimization results
            file_prefix: Prefix for output filenames
            formats: List of formats to export ('json', 'csv', 'html')
        
        Returns:
            Dictionary mapping formats to file paths
        """
        if formats is None:
            formats = ['json']
        
        # Prepare data for CSV/HTML export by flattening nested structures
        if 'results' in results and isinstance(results['results'], list):
            export_data = results
        else:
            # Convert results to a list format suitable for CSV/HTML
            if isinstance(results, dict) and 'best_params' in results:
                export_data = {
                    'strategy': results.get('strategy', 'Unknown'),
                    'results': [
                        {
                            'parameter': k, 
                            'value': v
                        } 
                        for k, v in results.get('best_params', {}).items()
                    ]
                }
                export_data['results'].append({
                    'parameter': 'best_score', 
                    'value': results.get('best_score', 0)
                })
            else:
                export_data = results
        
        return ReportExporter.export_report(export_data, file_prefix, formats)

