"""Memory profiling script for performance analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from memory_profiler import profile


@profile
def create_large_dataset():
    """Create a large dataset to test memory usage."""
    print("Creating large dataset...")
    data = pd.DataFrame(
        {
            "Open": np.random.randn(100000).cumsum() + 100,
            "High": np.random.randn(100000).cumsum() + 102,
            "Low": np.random.randn(100000).cumsum() + 98,
            "Close": np.random.randn(100000).cumsum() + 101,
            "Volume": np.random.randint(1000, 10000, 100000),
        }
    )
    print(f"Dataset shape: {data.shape}")
    return data


@profile
def process_data(data):
    """Process the dataset with various calculations."""
    print("Processing data...")
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["RSI"] = calculate_rsi(data["Close"])
    data["Volatility"] = data["Close"].rolling(window=20).std()
    print("Data processing complete")
    return data


def calculate_rsi(prices, period=14):
    """Calculate RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


if __name__ == "__main__":
    print("Starting memory profiling...")
    data = create_large_dataset()
    processed_data = process_data(data)
    print(f"Final dataset shape: {processed_data.shape}")
    print("Memory profiling complete")
