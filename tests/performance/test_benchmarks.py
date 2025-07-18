"""Performance benchmark tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.strategy import BuyAndHoldStrategy


class TestPerformanceBenchmarks:
    """Performance benchmark tests for critical components."""

    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range("2023-01-01", periods=1000, freq="D")
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "Open": rng.standard_normal(1000).cumsum() + 100,
                "High": rng.standard_normal(1000).cumsum() + 102,
                "Low": rng.standard_normal(1000).cumsum() + 98,
                "Close": rng.standard_normal(1000).cumsum() + 101,
                "Volume": rng.integers(1000, 10000, 1000),
            },
            index=dates,
        )

    @pytest.mark.benchmark
    def test_data_processing_performance(self, benchmark, sample_data):
        """Benchmark data processing performance."""

        def process_data():
            # Simulate data processing operations
            data = sample_data.copy()
            data["SMA_20"] = data["Close"].rolling(window=20).mean()
            data["SMA_50"] = data["Close"].rolling(window=50).mean()
            data["RSI"] = self._calculate_rsi(data["Close"])
            return data

        result = benchmark(process_data)
        assert len(result) == 1000

    @pytest.mark.benchmark
    def test_strategy_signal_generation(self, benchmark, sample_data):
        """Benchmark strategy signal generation."""
        strategy = BuyAndHoldStrategy()

        def generate_signals():
            return strategy.generate_signals(sample_data)

        signals = benchmark(generate_signals)
        assert len(signals) == len(sample_data)

    @pytest.mark.benchmark
    def test_large_dataset_processing(self, benchmark):
        """Benchmark processing of large datasets."""
        # Create a large dataset
        rng = np.random.default_rng(42)
        large_data = pd.DataFrame({"Close": rng.standard_normal(10000).cumsum() + 100})

        def process_large_data():
            # Simulate heavy computation
            result = large_data.copy()
            result["MA_50"] = result["Close"].rolling(50).mean()
            result["MA_200"] = result["Close"].rolling(200).mean()
            result["Volatility"] = result["Close"].rolling(20).std()
            return result

        result = benchmark(process_large_data)
        assert len(result) == 10000

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI for testing."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
