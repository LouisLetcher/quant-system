import pandas as pd

class DataCleaner:
    @staticmethod
    def remove_missing_values(data: pd.DataFrame):
        """Removes missing values from the dataset."""
        cleaned_data = data.dropna()
        print(f"🧹 Cleaned data: {len(cleaned_data)} rows remaining after removing missing values.")
        return cleaned_data

    @staticmethod
    def normalize_prices(data: pd.DataFrame):
        """Normalizes prices to start at 100 for easier comparison."""
        return data / data.iloc[0] * 100