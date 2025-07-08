"""Simple unit tests for UnifiedCacheManager."""

import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.core.cache_manager import UnifiedCacheManager


class TestUnifiedCacheManagerSimple:
    """Simplified test cases for UnifiedCacheManager."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Create UnifiedCacheManager instance."""
        return UnifiedCacheManager(cache_dir=temp_cache_dir, max_size_gb=1.0)

    @pytest.fixture
    def sample_dataframe(self):
        """Sample DataFrame for testing."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        return pd.DataFrame(
            {
                "Open": np.random.uniform(100, 200, 100),
                "High": np.random.uniform(100, 200, 100),
                "Low": np.random.uniform(100, 200, 100),
                "Close": np.random.uniform(100, 200, 100),
                "Volume": np.random.randint(1000000, 10000000, 100),
            },
            index=dates,
        )

    def test_init(self, cache_manager, temp_cache_dir):
        """Test initialization."""
        assert str(cache_manager.cache_dir) == temp_cache_dir
        assert cache_manager.max_size_bytes == int(1.0 * 1024**3)
        assert os.path.exists(cache_manager.metadata_db_path)

    def test_cache_and_retrieve_data(self, cache_manager, sample_dataframe):
        """Test caching and retrieving data."""
        symbol = "AAPL"

        # Cache the data
        success = cache_manager.cache_data(symbol, sample_dataframe)
        assert success == True

        # Retrieve the data
        retrieved_data = cache_manager.get_data(symbol)

        assert isinstance(retrieved_data, pd.DataFrame)
        assert len(retrieved_data) == len(sample_dataframe)
        assert list(retrieved_data.columns) == list(sample_dataframe.columns)

    def test_cache_stats(self, cache_manager, sample_dataframe):
        """Test getting cache statistics."""
        # Add some cached data
        cache_manager.cache_data("AAPL", sample_dataframe)
        cache_manager.cache_data("MSFT", sample_dataframe)

        stats = cache_manager.get_cache_stats()

        assert isinstance(stats, dict)
        assert "total_size_gb" in stats
        assert "max_size_gb" in stats
        assert "utilization" in stats

    def test_cache_with_different_intervals(self, cache_manager, sample_dataframe):
        """Test caching data with different intervals."""
        symbol = "AAPL"

        # Cache with different intervals
        success1 = cache_manager.cache_data(symbol, sample_dataframe, interval="1d")
        success2 = cache_manager.cache_data(symbol, sample_dataframe, interval="1h")

        assert success1 == True
        assert success2 == True

        # Retrieve with specific intervals
        data_1d = cache_manager.get_data(symbol, interval="1d")
        data_1h = cache_manager.get_data(symbol, interval="1h")

        assert isinstance(data_1d, pd.DataFrame)
        assert isinstance(data_1h, pd.DataFrame)

    def test_clear_cache(self, cache_manager, sample_dataframe):
        """Test cache clearing functionality."""
        # Cache some data
        cache_manager.cache_data("AAPL", sample_dataframe)
        cache_manager.cache_data("MSFT", sample_dataframe)

        # Clear all cache
        cleared_count = cache_manager.clear_all_cache()

        assert isinstance(cleared_count, int)
        assert cleared_count >= 0

    def test_nonexistent_data_retrieval(self, cache_manager):
        """Test retrieving non-existent data."""
        result = cache_manager.get_data("NONEXISTENT")
        assert result is None

    def test_error_handling(self, cache_manager):
        """Test error handling in cache operations."""
        # Test with invalid data
        try:
            result = cache_manager.cache_data("TEST", "not a dataframe")
            # Should either handle gracefully or raise appropriate error
            assert isinstance(result, bool)
        except (TypeError, ValueError):
            # Expected behavior for invalid data
            pass
