"""Unit tests for strategy module."""

import os
import sys
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.strategy import (
    BaseStrategy,
    BollingerBandsStrategy,
    BuyAndHoldStrategy,
    MACDStrategy,
    MeanReversionStrategy,
    MovingAverageCrossoverStrategy,
    RSIStrategy,
)


class TestBaseStrategy:
    """Test suite for BaseStrategy abstract class."""

    def test_cannot_instantiate_base_strategy(self):
        """Test that BaseStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseStrategy()


class TestBuyAndHoldStrategy:
    """Test suite for BuyAndHoldStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create a BuyAndHoldStrategy instance."""
        return BuyAndHoldStrategy()

    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        return pd.DataFrame(
            {
                "Open": range(100, 110),
                "High": range(102, 112),
                "Low": range(98, 108),
                "Close": range(101, 111),
                "Volume": [1000] * 10,
            },
            index=dates,
        )

    def test_generate_signals(self, strategy, sample_data):
        """Test signal generation for buy and hold."""
        signals = strategy.generate_signals(sample_data)

        # First signal should be buy (1), rest should be hold (0)
        assert signals.iloc[0] == 1
        assert all(signals.iloc[1:] == 0)
        assert len(signals) == len(sample_data)

    def test_optimize_parameters(self, strategy, sample_data):
        """Test parameter optimization (should return empty dict)."""
        params = strategy.optimize_parameters(sample_data)
        assert params == {}

    def test_empty_data(self, strategy):
        """Test strategy with empty data."""
        empty_data = pd.DataFrame()
        signals = strategy.generate_signals(empty_data)
        assert len(signals) == 0


class TestMovingAverageCrossoverStrategy:
    """Test suite for MovingAverageCrossoverStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create a MovingAverageCrossoverStrategy instance."""
        return MovingAverageCrossoverStrategy(short_window=3, long_window=5)

    @pytest.fixture
    def trend_data(self):
        """Create trending market data."""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        # Create upward trending data
        close_prices = [100 + i + np.random.normal(0, 0.1) for i in range(20)]
        return pd.DataFrame(
            {
                "Open": close_prices,
                "High": [p + 1 for p in close_prices],
                "Low": [p - 1 for p in close_prices],
                "Close": close_prices,
                "Volume": [1000] * 20,
            },
            index=dates,
        )

    def test_initialization(self):
        """Test proper initialization of parameters."""
        strategy = MovingAverageCrossoverStrategy(short_window=5, long_window=10)
        assert strategy.short_window == 5
        assert strategy.long_window == 10

    def test_generate_signals(self, strategy, trend_data):
        """Test signal generation for moving average crossover."""
        signals = strategy.generate_signals(trend_data)

        # Should have same length as input data
        assert len(signals) == len(trend_data)
        # Should only contain valid signal values (-1, 0, 1)
        assert all(signal in [-1, 0, 1] for signal in signals)

    def test_calculate_moving_averages(self, strategy, trend_data):
        """Test moving average calculation."""
        ma_short, ma_long = strategy.calculate_moving_averages(trend_data)

        assert len(ma_short) == len(trend_data)
        assert len(ma_long) == len(trend_data)
        # First few values should be NaN due to window size
        assert pd.isna(ma_short.iloc[0:2]).all()
        assert pd.isna(ma_long.iloc[0:4]).all()

    def test_insufficient_data(self, strategy):
        """Test strategy with insufficient data."""
        short_data = pd.DataFrame(
            {"Close": [100, 101]}, index=pd.date_range("2024-01-01", periods=2)
        )

        signals = strategy.generate_signals(short_data)
        # Should handle gracefully and return appropriate signals
        assert len(signals) == 2


class TestRSIStrategy:
    """Test suite for RSIStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create an RSIStrategy instance."""
        return RSIStrategy(
            rsi_period=14, oversold_threshold=30, overbought_threshold=70
        )

    @pytest.fixture
    def oscillating_data(self):
        """Create oscillating market data for RSI testing."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        # Create data that oscillates to trigger RSI signals
        close_prices = [
            100 + 10 * np.sin(i * 0.3) + np.random.normal(0, 0.5) for i in range(30)
        ]
        return pd.DataFrame(
            {
                "Open": close_prices,
                "High": [p + 1 for p in close_prices],
                "Low": [p - 1 for p in close_prices],
                "Close": close_prices,
                "Volume": [1000] * 30,
            },
            index=dates,
        )

    def test_initialization(self):
        """Test proper initialization of RSI parameters."""
        strategy = RSIStrategy(
            rsi_period=10, oversold_threshold=25, overbought_threshold=75
        )
        assert strategy.rsi_period == 10
        assert strategy.oversold_threshold == 25
        assert strategy.overbought_threshold == 75

    def test_calculate_rsi(self, strategy, oscillating_data):
        """Test RSI calculation."""
        rsi = strategy.calculate_rsi(oscillating_data["Close"])

        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert all(0 <= value <= 100 for value in valid_rsi)
        # Should have NaN values for initial period
        assert sum(pd.isna(rsi)) >= strategy.rsi_period

    def test_generate_signals(self, strategy, oscillating_data):
        """Test signal generation based on RSI."""
        signals = strategy.generate_signals(oscillating_data)

        assert len(signals) == len(oscillating_data)
        assert all(signal in [-1, 0, 1] for signal in signals)

    def test_extreme_rsi_values(self, strategy):
        """Test RSI with extreme price movements."""
        # Create data with extreme movements
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        close_prices = [100] * 10 + [150] * 10  # Sharp increase
        data = pd.DataFrame({"Close": close_prices}, index=dates)

        rsi = strategy.calculate_rsi(data["Close"])
        # Should handle extreme movements gracefully
        valid_rsi = rsi.dropna()
        assert all(0 <= value <= 100 for value in valid_rsi)


class TestMACDStrategy:
    """Test suite for MACDStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create a MACDStrategy instance."""
        return MACDStrategy(fast_period=12, slow_period=26, signal_period=9)

    @pytest.fixture
    def trending_data(self):
        """Create trending data for MACD testing."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        # Create data with trend changes
        close_prices = [100 + i * 0.5 + 5 * np.sin(i * 0.1) for i in range(50)]
        return pd.DataFrame({"Close": close_prices}, index=dates)

    def test_calculate_macd(self, strategy, trending_data):
        """Test MACD calculation."""
        macd_line, signal_line, histogram = strategy.calculate_macd(
            trending_data["Close"]
        )

        assert len(macd_line) == len(trending_data)
        assert len(signal_line) == len(trending_data)
        assert len(histogram) == len(trending_data)

        # Check that histogram = macd_line - signal_line (where both are not NaN)
        valid_mask = ~(pd.isna(macd_line) | pd.isna(signal_line))
        expected_histogram = macd_line[valid_mask] - signal_line[valid_mask]
        actual_histogram = histogram[valid_mask]
        assert np.allclose(expected_histogram, actual_histogram, rtol=1e-10)

    def test_generate_signals(self, strategy, trending_data):
        """Test MACD signal generation."""
        signals = strategy.generate_signals(trending_data)

        assert len(signals) == len(trending_data)
        assert all(signal in [-1, 0, 1] for signal in signals)


class TestBollingerBandsStrategy:
    """Test suite for BollingerBandsStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create a BollingerBandsStrategy instance."""
        return BollingerBandsStrategy(period=20, std_dev=2)

    @pytest.fixture
    def volatile_data(self):
        """Create volatile data for Bollinger Bands testing."""
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        # Create volatile data
        close_prices = [100 + 20 * np.random.normal(0, 1) for _ in range(40)]
        return pd.DataFrame({"Close": close_prices}, index=dates)

    def test_calculate_bollinger_bands(self, strategy, volatile_data):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = strategy.calculate_bollinger_bands(
            volatile_data["Close"]
        )

        assert len(upper) == len(volatile_data)
        assert len(middle) == len(volatile_data)
        assert len(lower) == len(volatile_data)

        # Upper band should be above middle, middle above lower
        valid_mask = ~(pd.isna(upper) | pd.isna(middle) | pd.isna(lower))
        assert all(upper[valid_mask] >= middle[valid_mask])
        assert all(middle[valid_mask] >= lower[valid_mask])

    def test_generate_signals(self, strategy, volatile_data):
        """Test Bollinger Bands signal generation."""
        signals = strategy.generate_signals(volatile_data)

        assert len(signals) == len(volatile_data)
        assert all(signal in [-1, 0, 1] for signal in signals)


class TestMeanReversionStrategy:
    """Test suite for MeanReversionStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create a MeanReversionStrategy instance."""
        return MeanReversionStrategy(lookback_period=20, z_threshold=2.0)

    @pytest.fixture
    def mean_reverting_data(self):
        """Create mean-reverting data."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        # Create mean-reverting data around 100
        close_prices = [100 + 10 * np.random.normal(0, 1) for _ in range(50)]
        return pd.DataFrame({"Close": close_prices}, index=dates)

    def test_calculate_z_score(self, strategy, mean_reverting_data):
        """Test Z-score calculation."""
        z_scores = strategy.calculate_z_score(mean_reverting_data["Close"])

        assert len(z_scores) == len(mean_reverting_data)
        # Z-scores should be centered around 0 for the lookback period
        valid_z_scores = z_scores.dropna()
        if len(valid_z_scores) > 0:
            # Mean should be close to 0
            assert abs(valid_z_scores.mean()) < 0.5

    def test_generate_signals(self, strategy, mean_reverting_data):
        """Test mean reversion signal generation."""
        signals = strategy.generate_signals(mean_reverting_data)

        assert len(signals) == len(mean_reverting_data)
        assert all(signal in [-1, 0, 1] for signal in signals)

    def test_extreme_z_scores(self, strategy):
        """Test strategy with extreme Z-scores."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        # Create data with extreme outliers
        close_prices = [100] * 25 + [150, 50, 100, 100, 100]
        data = pd.DataFrame({"Close": close_prices}, index=dates)

        z_scores = strategy.calculate_z_score(data["Close"])
        signals = strategy.generate_signals(data)

        # Should handle extreme values gracefully
        assert len(signals) == len(data)
        assert all(signal in [-1, 0, 1] for signal in signals)


if __name__ == "__main__":
    pytest.main([__file__])
