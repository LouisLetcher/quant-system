import json
import pandas as pd
import os

class ReportExporter:
    """Handles exporting reports in various formats."""

    @staticmethod
    def export_to_json(data: dict, file_path: str):
        """Exports data to JSON format."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        return file_path

    @staticmethod
    def export_to_csv(data: list, file_path: str):
        """Exports data to CSV format."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        return file_path

    @staticmethod
    def export_to_html(data: list, file_path: str):
        """Exports data to HTML format."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df = pd.DataFrame(data)
        df.to_html(file_path, index=False)
        return file_path