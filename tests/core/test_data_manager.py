"""Unit tests for UnifiedDataManager."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.data_manager import UnifiedDataManager


class TestUnifiedDataManager:
    """Test cases for UnifiedDataManager."""

    @pytest.fixture
    def mock_cache_manager(self):
        """Mock cache manager."""
        mock_cache = Mock()
        mock_cache.get_cache_stats.return_value = {
            "total_size_gb": 0.1,
            "max_size_gb": 10.0,
            "utilization": 0.01,
        }
        return mock_cache

    @pytest.fixture
    def data_manager(self, mock_cache_manager):
        """Create UnifiedDataManager instance."""
        return UnifiedDataManager(cache_manager=mock_cache_manager)

    @pytest.fixture
    def sample_data(self):
        """Sample market data."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "Open": rng.uniform(100, 200, 100),
                "High": rng.uniform(100, 200, 100),
                "Low": rng.uniform(100, 200, 100),
                "Close": rng.uniform(100, 200, 100),
                "Volume": rng.integers(1000000, 10000000, 100),
            },
            index=dates,
        )

    def test_init(self, data_manager):
        """Test initialization."""
        assert isinstance(data_manager, UnifiedDataManager)
        # Default sources are initialized automatically
        assert len(data_manager.sources) >= 1  # At least yahoo_finance and bybit
        assert data_manager.cache_manager is not None

    def test_get_data(self, data_manager, sample_data):
        """Test the main data fetching method."""
        # Mock the cache manager to return None (cache miss)
        data_manager.cache_manager.get_data.return_value = None

        # Mock Yahoo Finance source
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = sample_data
            mock_ticker.return_value = mock_ticker_instance

            result = data_manager.get_data(
                symbol="AAPL",
                start_date="2023-01-01",
                end_date="2023-12-31",
                interval="1d",
            )

            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            assert all(
                col in result.columns for col in ["open", "high", "low", "close"]
            )

    def test_add_source_with_actual_source(self, data_manager):
        """Test adding actual DataSource objects."""
        from src.core.data_manager import DataSource, DataSourceConfig

        # Create a mock DataSource for testing
        class MockDataSource(DataSource):
            def __init__(self):
                config = DataSourceConfig(
                    name="test_source",
                    priority=5,
                    rate_limit=1.0,
                    max_retries=3,
                    timeout=30.0,
                    supports_batch=False,
                    asset_types=["stocks"],
                )
                super().__init__(config)

            def fetch_data(self, symbol, start_date, end_date, interval="1d", **kwargs):
                return None

            def fetch_batch_data(
                self, symbols, start_date, end_date, interval="1d", **kwargs
            ):
                return {}

            def get_available_symbols(self, asset_type=None):
                return []

        # Test adding a DataSource object
        test_source = MockDataSource()
        initial_count = len(data_manager.sources)
        data_manager.add_source(test_source)

        assert len(data_manager.sources) == initial_count + 1
        assert "test_source" in data_manager.sources
        assert data_manager.sources["test_source"] == test_source

    def test_default_sources_initialization(self):
        """Test that default sources are properly initialized."""
        from src.core.data_manager import UnifiedDataManager

        # Create a fresh instance
        manager = UnifiedDataManager()

        # Should have at least yahoo_finance and bybit
        assert len(manager.sources) >= 2
        assert "yahoo_finance" in manager.sources
        assert "bybit" in manager.sources

        # Check that sources are properly configured
        yahoo_source = manager.sources["yahoo_finance"]
        assert yahoo_source.config.name == "yahoo_finance"
        assert yahoo_source.config.supports_batch is True

        bybit_source = manager.sources["bybit"]
        assert bybit_source.config.name == "bybit"
        assert bybit_source.config.supports_futures is True

    def test_cache_integration(self, data_manager, sample_data):
        """Test that caching works properly."""
        # Test cache hit
        data_manager.cache_manager.get_data.return_value = sample_data

        result = data_manager.get_data(
            symbol="AAPL",
            start_date="2023-01-01",
            end_date="2023-12-31",
            use_cache=True,
        )

        assert isinstance(result, pd.DataFrame)
        data_manager.cache_manager.get_data.assert_called_once()

        # Test cache miss and cache storage
        data_manager.cache_manager.get_data.return_value = None

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = sample_data
            mock_ticker.return_value = mock_ticker_instance

            result = data_manager.get_data(
                symbol="TSLA",
                start_date="2023-01-01",
                end_date="2023-12-31",
                use_cache=True,
            )

            assert isinstance(result, pd.DataFrame)
            # Verify cache_data was called to store the result
            data_manager.cache_manager.cache_data.assert_called_once()

    def test_asset_type_detection(self, data_manager):
        """Test asset type detection from symbols."""
        # Test crypto detection
        assert data_manager._detect_asset_type("BTCUSDT") == "crypto"
        assert data_manager._detect_asset_type("ETH-USD") == "crypto"

        # Test forex detection
        assert data_manager._detect_asset_type("EURUSD=X") == "forex"
        assert data_manager._detect_asset_type("GBPUSD") == "forex"

        # Test stocks detection (default)
        assert data_manager._detect_asset_type("AAPL") == "stocks"
        assert data_manager._detect_asset_type("MSFT") == "stocks"

    def test_source_status(self, data_manager):
        """Test getting source status information."""
        status = data_manager.get_source_status()

        assert isinstance(status, dict)
        assert "yahoo_finance" in status
        assert "bybit" in status

        # Check status structure
        yahoo_status = status["yahoo_finance"]
        assert "priority" in yahoo_status
        assert "supports_batch" in yahoo_status
        assert "asset_types" in yahoo_status
