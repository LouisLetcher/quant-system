from __future__ import annotations

import os

import pandas as pd


class Cache:
    CACHE_DIR = "cache/"
    
    @staticmethod
    def save_to_cache(ticker: str, data: pd.DataFrame, interval="1d"):
        """Saves stock data to a local CSV cache."""
        os.makedirs(Cache.CACHE_DIR, exist_ok=True)
        file_path = Cache._get_cache_file_path(ticker, interval)
    
        # Convert MultiIndex columns to simple columns before saving
        if isinstance(data.columns, pd.MultiIndex):
            # Create a new DataFrame with simple column names
            simplified_data = pd.DataFrame()
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                # Find the column in the MultiIndex
                for full_col in data.columns:
                    col_name = full_col[0] if isinstance(full_col, tuple) else full_col
                    if str(col_name).lower() == col.lower():
                        simplified_data[col] = data[full_col]
                        break
            # Only save if we found all required columns
            if len(simplified_data.columns) == 5:
                simplified_data.to_csv(file_path, index=True)
                return
    
        # Original logic if not MultiIndex or conversion failed
        data.to_csv(file_path, index=True)
        
    @staticmethod
    def load_from_cache(ticker: str, interval="1d") -> pd.DataFrame:
        """Load data from cache if it exists"""
        file_path = Cache._get_cache_file_path(ticker, interval)
        if os.path.exists(file_path):
            try:
                # Read the cached data with more flexible date parsing
                df = pd.read_csv(
                    file_path,
                    index_col=0,
                    header=0,
                    parse_dates=True,
                    # Remove strict format requirement
                )

                # Ensure proper data types after loading from cache
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                # Convert index to datetime more flexibly
                df.index = pd.to_datetime(df.index, errors="coerce")
    
                # Only filter out rows where ALL values are NaN
                df = df.dropna(how='all')
    
                # Add check to ensure DataFrame has data
                if df.empty:
                    print(f"⚠️ Cache file for {ticker} exists but contains no data")
                    return None

                print(f"✅ Successfully loaded {len(df)} rows from cache for {ticker}")
                return df
            except Exception as e:
                print(f"⚠️ Cache loading error: {e}")
                return None
        return None

    @staticmethod
    def _get_cache_file_path(ticker: str, interval: str = "1d") -> str:
        """Returns the full file path for a ticker's cache file."""
        os.makedirs(Cache.CACHE_DIR, exist_ok=True)
        return os.path.join(Cache.CACHE_DIR, f"{ticker}_{interval}.csv")
