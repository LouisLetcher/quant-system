"""Unit tests for UnifiedDataManager."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.data_manager import DataSource, UnifiedDataManager


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
        data = pd.DataFrame(
            {
                "Open": np.random.uniform(100, 200, 100),
                "High": np.random.uniform(100, 200, 100),
                "Low": np.random.uniform(100, 200, 100),
                "Close": np.random.uniform(100, 200, 100),
                "Volume": np.random.randint(1000000, 10000000, 100),
            },
            index=dates,
        )
        return data

    def test_init(self, data_manager):
        """Test initialization."""
        assert isinstance(data_manager, UnifiedDataManager)
        assert len(data_manager.sources) == 0
        assert data_manager.cache_manager is not None

    def test_add_data_source(self, data_manager):
        """Test adding data sources."""
        # Test adding yahoo finance source
        data_manager.add_source("yahoo_finance")
        assert "yahoo_finance" in data_manager.sources

        # Test adding bybit source
        data_manager.add_source("bybit")
        assert "bybit" in data_manager.sources

        # Test invalid source
        with pytest.raises(ValueError):
            data_manager.add_source("invalid_source")

    def test_remove_data_source(self, data_manager):
        """Test removing data sources."""
        data_manager.add_source("yahoo_finance")
        data_manager.remove_source("yahoo_finance")
        assert "yahoo_finance" not in data_manager.sources

    @patch("src.core.data_manager.yf.download")
    def test_fetch_yahoo_finance_data(
        self, mock_yf_download, data_manager, sample_data
    ):
        """Test fetching data from Yahoo Finance."""
        mock_yf_download.return_value = sample_data

        data_manager.add_source("yahoo_finance")
        result = data_manager.fetch_data(
            symbol="AAPL", start_date="2023-01-01", end_date="2023-12-31"
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100
        assert all(
            col in result.columns for col in ["Open", "High", "Low", "Close", "Volume"]
        )

    @patch("src.core.data_manager.requests.get")
    def test_fetch_bybit_data(self, mock_get, data_manager):
        """Test fetching data from Bybit."""
        # Mock Bybit API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "list": [
                    [
                        "1640995200000",
                        "47000",
                        "47500",
                        "46500",
                        "47200",
                        "1000",
                        "47000000",
                    ],
                    [
                        "1641081600000",
                        "47200",
                        "47800",
                        "46800",
                        "47400",
                        "1200",
                        "56880000",
                    ],
                ]
            }
        }
        mock_get.return_value = mock_response

        data_manager.add_source("bybit")
        result = data_manager.fetch_data(
            symbol="BTCUSDT",
            start_date="2022-01-01",
            end_date="2022-01-02",
            asset_type="crypto",
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_batch_fetch_data(self, data_manager, sample_data, mock_cache_manager):
        """Test batch data fetching."""
        with patch("src.core.data_manager.yf.download") as mock_yf_download:
            mock_yf_download.return_value = sample_data

            data_manager.add_source("yahoo_finance")
            results = data_manager.batch_fetch_data(
                symbols=["AAPL", "MSFT"], start_date="2023-01-01", end_date="2023-12-31"
            )

            assert isinstance(results, dict)
            assert len(results) == 2
            assert "AAPL" in results
            assert "MSFT" in results

    def test_validate_symbol(self, data_manager):
        """Test symbol validation."""
        # Valid symbols
        assert data_manager._validate_symbol("AAPL", "stocks") == True
        assert data_manager._validate_symbol("BTCUSDT", "crypto") == True
        assert data_manager._validate_symbol("EURUSD=X", "forex") == True

        # Invalid symbols
        assert data_manager._validate_symbol("", "stocks") == False
        assert data_manager._validate_symbol("INVALID123", "stocks") == False

    def test_get_available_symbols(self, data_manager):
        """Test getting available symbols."""
        symbols = data_manager.get_available_symbols("stocks")
        assert isinstance(symbols, list)
        assert len(symbols) > 0

    def test_get_source_info(self, data_manager):
        """Test getting source information."""
        data_manager.add_source("yahoo_finance")
        info = data_manager.get_source_info()

        assert isinstance(info, dict)
        assert "yahoo_finance" in info
        assert "priority" in info["yahoo_finance"]
        assert "supports_batch" in info["yahoo_finance"]

    def test_error_handling(self, data_manager):
        """Test error handling."""
        # Test with no sources
        with pytest.raises(ValueError):
            data_manager.fetch_data("AAPL", "2023-01-01", "2023-12-31")

        # Test with invalid date format
        data_manager.add_source("yahoo_finance")
        with pytest.raises(ValueError):
            data_manager.fetch_data("AAPL", "invalid-date", "2023-12-31")

    def test_cache_integration(self, data_manager, sample_data):
        """Test cache integration."""
        data_manager.cache_manager.get_data.return_value = sample_data

        # Test cache hit
        result = data_manager.fetch_data(
            symbol="AAPL", start_date="2023-01-01", end_date="2023-12-31"
        )

        assert isinstance(result, pd.DataFrame)
        data_manager.cache_manager.get_data.assert_called_once()

    @pytest.mark.parametrize(
        "asset_type,expected_interval",
        [("stocks", "1d"), ("crypto", "1h"), ("forex", "1d")],
    )
    def test_get_default_interval(self, data_manager, asset_type, expected_interval):
        """Test getting default intervals for different asset types."""
        interval = data_manager._get_default_interval(asset_type)
        assert interval == expected_interval

    def test_data_quality_checks(self, data_manager, sample_data):
        """Test data quality validation."""
        # Test with good data
        is_valid = data_manager._validate_data_quality(sample_data)
        assert is_valid == True

        # Test with data containing NaN
        bad_data = sample_data.copy()
        bad_data.iloc[0, 0] = np.nan
        is_valid = data_manager._validate_data_quality(bad_data)
        assert is_valid == False

        # Test with empty data
        empty_data = pd.DataFrame()
        is_valid = data_manager._validate_data_quality(empty_data)
        assert is_valid == False

    def test_data_normalization(self, data_manager):
        """Test data normalization across different sources."""
        # Create test data with different formats
        test_data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [99, 100, 101],
                "close": [104, 105, 106],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        normalized = data_manager._normalize_data(test_data)

        # Check that columns are standardized
        expected_columns = ["Open", "High", "Low", "Close", "Volume"]
        assert list(normalized.columns) == expected_columns

    def test_concurrent_requests(self, data_manager, sample_data):
        """Test handling of concurrent data requests."""
        with patch("src.core.data_manager.yf.download") as mock_yf_download:
            mock_yf_download.return_value = sample_data

            data_manager.add_source("yahoo_finance")

            # Simulate concurrent requests
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
            results = data_manager.batch_fetch_data(
                symbols=symbols, start_date="2023-01-01", end_date="2023-12-31"
            )

            assert len(results) == len(symbols)
            for symbol in symbols:
                assert symbol in results
                assert isinstance(results[symbol], pd.DataFrame)
