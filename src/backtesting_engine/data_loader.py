from __future__ import annotations

from src.data_scraper.data_manager import DataManager


class DataLoader:
    """Loads historical price data for backtesting, using cache when available."""

    @staticmethod
    def load_data(ticker, period="max", interval="1d", start=None, end=None):
        """
        Loads price data for a ticker using DataManager.
        """
        print(f"🔍 Loading {ticker} data with interval {interval}...")

        # Load data with the specific requested interval
        data = DataManager.get_stock_data(ticker, start, end, interval)

        if data is None or data.empty:
            print(
                f"⚠️ No data available for {ticker} with {interval} interval. Trying daily data..."
            )

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
                        "1m": "1min",
                        "5m": "5min",
                        "15m": "15min",
                        "30m": "30min",
                        "1h": "1h",
                        "4h": "4h",
                        "1d": "1D",
                        "1wk": "1W",
                        "1mo": "1M",
                        "3mo": "3M",
                    }
                    resample_rule = interval_map.get(interval, "1D")

                    # Save original data length
                    original_length = len(data)

                    # Resample OHLCV data
                    data = (
                        data.resample(resample_rule)
                        .agg(
                            {
                                "Open": "first",
                                "High": "max",
                                "Low": "min",
                                "Close": "last",
                                "Volume": "sum",
                            }
                        )
                        .dropna()
                    )

                    # Check if we have enough data after resampling
                    if len(data) > 0:
                        print(
                            f"✅ Resampled daily data to {interval} - got {len(data)} bars from {original_length} original bars"
                        )
                        return data
                    print(f"⚠️ No data available after resampling to {interval}")
                    # Return the original daily data instead of None
                    print(
                        f"⚠️ Falling back to daily data with {original_length} bars"
                    )
                    return DataManager.get_stock_data(ticker, start, end, "1d")
                except Exception as e:
                    print(f"❌ Error resampling data: {e!s}")
                    # Return the original daily data instead of None
                    print("⚠️ Falling back to daily data due to resampling error")
                    return DataManager.get_stock_data(ticker, start, end, "1d")

        print(f"✅ Got {len(data)} bars for {ticker}")
        return data
