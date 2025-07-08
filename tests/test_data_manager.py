"""Unit tests for data_manager module."""

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.data_manager import DataSourceConfig, UnifiedDataManager


class TestUnifiedDataManager:
    """Test suite for UnifiedDataManager."""

    @pytest.fixture
    def data_manager(self):
        """Create a UnifiedDataManager instance for testing."""
        return UnifiedDataManager()

    @pytest.fixture
    def sample_config(self):
        """Sample data source configuration."""
        return DataSourceConfig(
            primary_source="yahoo",
            fallback_sources=["alpha_vantage"],
            symbols=["AAPL", "GOOGL"],
            symbol_map={"AAPL": {"yahoo": "AAPL", "alpha_vantage": "AAPL"}},
        )

    def test_initialization(self, data_manager):
        """Test proper initialization of UnifiedDataManager."""
        assert data_manager.cache_dir.exists()
        assert hasattr(data_manager, "sources")
        assert "yahoo" in data_manager.sources

    def test_transform_symbol_yahoo_to_yahoo(self, data_manager):
        """Test symbol transformation for Yahoo Finance."""
        result = data_manager.transform_symbol("AAPL", "yahoo", "yahoo")
        assert result == "AAPL"

    def test_transform_symbol_crypto_binance(self, data_manager):
        """Test crypto symbol transformation for Binance format."""
        result = data_manager.transform_symbol("BTC-USD", "yahoo", "bybit")
        assert result == "BTCUSDT"

    def test_transform_symbol_forex_oanda(self, data_manager):
        """Test forex symbol transformation."""
        result = data_manager.transform_symbol("EURUSD=X", "yahoo", "alpha_vantage")
        assert result == "EUR/USD"

    @patch("yfinance.download")
    def test_fetch_data_yahoo_success(self, mock_download, data_manager):
        """Test successful data fetch from Yahoo Finance."""
        # Mock successful response
        mock_data = pd.DataFrame(
            {
                "Open": [100, 101],
                "High": [102, 103],
                "Low": [99, 100],
                "Close": [101, 102],
                "Volume": [1000, 1100],
            },
            index=pd.date_range("2024-01-01", periods=2),
        )
        mock_download.return_value = mock_data

        result = data_manager.fetch_data(
            "AAPL", "yahoo", datetime(2024, 1, 1), datetime(2024, 1, 2)
        )

        assert not result.empty
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]

    @patch("yfinance.download")
    def test_fetch_data_fallback(self, mock_download, data_manager):
        """Test fallback mechanism when primary source fails."""
        # Mock primary source failure
        mock_download.side_effect = Exception("Network error")

        with patch.object(data_manager, "_fetch_alpha_vantage") as mock_av:
            mock_av.return_value = pd.DataFrame(
                {
                    "Open": [100],
                    "High": [102],
                    "Low": [99],
                    "Close": [101],
                    "Volume": [1000],
                },
                index=[datetime(2024, 1, 1)],
            )

            result = data_manager.fetch_data_with_fallback(
                "AAPL",
                primary_source="yahoo",
                fallback_sources=["alpha_vantage"],
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 2),
            )

            assert not result.empty
            mock_av.assert_called_once()

    def test_validate_data_complete(self, data_manager):
        """Test data validation for complete dataset."""
        valid_data = pd.DataFrame(
            {
                "Open": [100, 101],
                "High": [102, 103],
                "Low": [99, 100],
                "Close": [101, 102],
                "Volume": [1000, 1100],
            },
            index=pd.date_range("2024-01-01", periods=2),
        )

        assert data_manager.validate_data(valid_data)

    def test_validate_data_missing_columns(self, data_manager):
        """Test data validation fails for missing columns."""
        invalid_data = pd.DataFrame(
            {"Open": [100, 101], "Close": [101, 102]},
            index=pd.date_range("2024-01-01", periods=2),
        )

        assert not data_manager.validate_data(invalid_data)

    def test_validate_data_empty(self, data_manager):
        """Test data validation fails for empty dataset."""
        empty_data = pd.DataFrame()
        assert not data_manager.validate_data(empty_data)

    @patch("pandas.DataFrame.to_parquet")
    def test_cache_data(self, mock_to_parquet, data_manager):
        """Test data caching functionality."""
        test_data = pd.DataFrame({"Close": [100, 101]})
        data_manager.cache_data("AAPL", test_data, "1d")
        mock_to_parquet.assert_called_once()

    def test_get_cache_key(self, data_manager):
        """Test cache key generation."""
        key = data_manager.get_cache_key(
            "AAPL", datetime(2024, 1, 1), datetime(2024, 1, 31), "1d"
        )
        expected = "AAPL_2024-01-01_2024-01-31_1d"
        assert key == expected

    def test_is_cache_valid_recent(self, data_manager):
        """Test cache validity for recent data."""
        # Mock recent file
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_path.stat.return_value.st_mtime = (
            datetime.now().timestamp() - 1800
        )  # 30 min ago

        with patch.object(data_manager, "cache_dir") as mock_cache_dir:
            mock_cache_dir.__truediv__.return_value = mock_path
            assert data_manager.is_cache_valid("test_key", max_age_hours=1)

    def test_is_cache_valid_expired(self, data_manager):
        """Test cache validity for expired data."""
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_path.stat.return_value.st_mtime = (
            datetime.now().timestamp() - 7200
        )  # 2 hours ago

        with patch.object(data_manager, "cache_dir") as mock_cache_dir:
            mock_cache_dir.__truediv__.return_value = mock_path
            assert not data_manager.is_cache_valid("test_key", max_age_hours=1)


class TestDataSourceConfig:
    """Test suite for DataSourceConfig."""

    def test_initialization(self):
        """Test proper initialization of DataSourceConfig."""
        config = DataSourceConfig(
            primary_source="yahoo",
            fallback_sources=["alpha_vantage"],
            symbols=["AAPL"],
            symbol_map={"AAPL": {"yahoo": "AAPL"}},
        )

        assert config.primary_source == "yahoo"
        assert config.fallback_sources == ["alpha_vantage"]
        assert "AAPL" in config.symbols
        assert config.symbol_map["AAPL"]["yahoo"] == "AAPL"

    def test_get_symbol_for_source(self):
        """Test getting symbol for specific source."""
        config = DataSourceConfig(
            primary_source="yahoo",
            fallback_sources=[],
            symbols=["BTC-USD"],
            symbol_map={"BTC-USD": {"yahoo": "BTC-USD", "bybit": "BTCUSDT"}},
        )

        assert config.get_symbol_for_source("BTC-USD", "yahoo") == "BTC-USD"
        assert config.get_symbol_for_source("BTC-USD", "bybit") == "BTCUSDT"
        assert (
            config.get_symbol_for_source("BTC-USD", "unknown") == "BTC-USD"
        )  # fallback


if __name__ == "__main__":
    pytest.main([__file__])
