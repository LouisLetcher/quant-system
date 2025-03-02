import pandas as pd

class Transformer:
    @staticmethod
    def to_json(data: pd.DataFrame):
        """Converts a DataFrame to JSON format."""
        return data.to_json(orient="records")

    @staticmethod
    def to_csv(data: pd.DataFrame, filename: str):
        """Saves a DataFrame to a CSV file."""
        data.to_csv(filename)