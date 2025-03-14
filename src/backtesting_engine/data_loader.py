import pandas as pd
from src.data_scraper.data_manager import DataManager

class DataLoader:
    """Loads historical price data for backtesting, using cache when available."""

    @staticmethod
    def load_data(ticker, period="max", interval="1d", start=None, end=None):
        """
        Loads price data for a ticker using DataManager.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period for data (e.g., "1y", "max")
            interval: Bar interval ("1d", "1wk", etc.)
            start: Start date (overrides period if provided)
            end: End date
            
        Returns:
            DataFrame with OHLCV data or None if data not available
        """
        print(f"🔍 Loading {ticker} data with interval {interval}...")

        # Load data with the specific requested interval
        data = DataManager.get_stock_data(ticker, start, end, interval)

        if data is None or data.empty:
            print(f"⚠️ No data available for {ticker} with {interval} interval. Trying daily data...")

            # Fall back to daily data if the specific interval isn't available
            data = DataManager.get_stock_data(ticker, start, end, "1d")

            if data is None or data.empty:
                print(f"❌ No data available for {ticker}")
                return None

            # If daily data is loaded but a different interval was requested, resample
            if interval != "1d":
                try:
                    # Map common intervals to pandas resample rule
                    interval_map = {
                        "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
                        "1h": "1H", "4h": "4H", "1d": "1D", "1wk": "1W", "1mo": "1ME", "3mo": "3ME"
                    }
                    resample_rule = interval_map.get(interval, "1D")

                    # Resample OHLCV data
                    data = data.resample(resample_rule).agg({
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last',
                        'Volume': 'sum'
                    }).dropna()

                    print(f"✅ Resampled daily data to {interval} - got {len(data)} bars")
                except Exception as e:
                    print(f"❌ Error resampling data: {str(e)}")
                    return None

        print(f"✅ Got {len(data)} bars for {ticker}")
        return data
