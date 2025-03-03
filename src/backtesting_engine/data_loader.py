import pandas as pd
from src.data_scraper.data_manager import DataManager

class DataLoader:
    @staticmethod
    def load_data(ticker, start, end):
        """Fetches stock data and formats it for Backtrader."""
        df = DataManager.get_stock_data(ticker, start, end)

        if df is None or df.empty:
            raise ValueError(f"❌ No data available for {ticker} from {start} to {end}.")

        print(f"✅ Data Loaded: {df.shape[0]} rows, {df.columns.tolist()}")

        # 🔍 **Fix multi-index issue** if it occurs
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)  # Drop extra index level

        # Ensure correct column names
        column_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        df.rename(columns=column_mapping, inplace=True)

        # Ensure the correct order for Backtrader
        df = df[["open", "high", "low", "close", "volume"]]
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        return df