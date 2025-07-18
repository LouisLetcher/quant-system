"""Unit tests for UnifiedCacheManager."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.cache_manager import UnifiedCacheManager


class TestUnifiedCacheManager:
    """Test cases for UnifiedCacheManager."""

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

    @pytest.fixture
    def sample_backtest_result(self):
        """Sample backtest result for testing."""
        return {
            "symbol": "AAPL",
            "strategy": "rsi",
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.08,
            "trades": 25,
            "win_rate": 0.64,
        }

    def test_init(self, cache_manager, temp_cache_dir):
        """Test initialization."""
        assert str(cache_manager.cache_dir) == temp_cache_dir
        assert cache_manager.max_size_bytes == int(1.0 * 1024**3)
        assert Path(cache_manager.metadata_db).exists()

    def test_generate_cache_key(self, cache_manager):
        """Test cache key generation."""
        key1 = cache_manager._generate_key("data", symbol="AAPL", interval="1d")
        key2 = cache_manager._generate_key("data", symbol="AAPL", interval="1d")

        # Same parameters should generate same key
        assert key1 == key2

        # Different parameters should generate different keys
        key3 = cache_manager._generate_key("data", symbol="MSFT", interval="1d")
        assert key1 != key3

    def test_cache_data(self, cache_manager, sample_dataframe):
        """Test caching DataFrame data."""
        # Cache the data
        key = cache_manager.cache_data("AAPL", sample_dataframe, ttl_hours=1)
        assert key is not None

        # Verify file was created
        expected_path = cache_manager.data_dir / f"{key}.gz"
        assert expected_path.exists()

    def test_get_data(self, cache_manager, sample_dataframe):
        """Test retrieving cached data."""
        # Cache the data first
        cache_manager.cache_data("AAPL", sample_dataframe)

        # Retrieve the data
        retrieved_data = cache_manager.get_data("AAPL")

        assert isinstance(retrieved_data, pd.DataFrame)
        assert len(retrieved_data) == len(sample_dataframe)
        assert list(retrieved_data.columns) == list(sample_dataframe.columns)

    def test_cache_backtest_result(self, cache_manager, sample_backtest_result):
        """Test caching backtest results."""
        parameters = {"rsi_period": 14, "rsi_overbought": 70}

        # Cache the result
        key = cache_manager.cache_backtest_result(
            "AAPL", "rsi", parameters, sample_backtest_result
        )
        assert key is not None

        # Verify file was created
        expected_path = cache_manager.backtest_dir / f"{key}.gz"
        assert expected_path.exists()

    def test_get_backtest_result(self, cache_manager, sample_backtest_result):
        """Test retrieving cached backtest results."""
        parameters = {"rsi_period": 14, "rsi_overbought": 70}

        # Cache the result first
        cache_manager.cache_backtest_result(
            "AAPL", "rsi", parameters, sample_backtest_result
        )

        # Retrieve the result
        retrieved_result = cache_manager.get_backtest_result("AAPL", "rsi", parameters)

        assert isinstance(retrieved_result, dict)
        assert retrieved_result["symbol"] == sample_backtest_result["symbol"]
        assert (
            abs(
                retrieved_result["total_return"]
                - sample_backtest_result["total_return"]
            )
            < 0.001
        )

    def test_cache_optimization_result(self, cache_manager):
        """Test caching optimization results."""
        optimization_config = {"param_ranges": {"rsi_period": [10, 20]}}
        optimization_result = {
            "best_params": {"rsi_period": 14, "rsi_overbought": 70},
            "best_score": 1.5,
            "all_results": [
                {"params": {"rsi_period": 10}, "score": 1.2},
                {"params": {"rsi_period": 14}, "score": 1.5},
            ],
        }

        # Cache the result
        key = cache_manager.cache_optimization_result(
            "AAPL", "rsi", optimization_config, optimization_result
        )
        assert key is not None

        # Retrieve the result
        retrieved_result = cache_manager.get_optimization_result(
            "AAPL", "rsi", optimization_config
        )

        assert isinstance(retrieved_result, dict)
        assert abs(retrieved_result["best_score"] - 1.5) < 0.001

    def test_cache_expiration(self, cache_manager, sample_dataframe):
        """Test cache expiration."""
        # Cache data with short TTL
        cache_manager.cache_data("AAPL", sample_dataframe, ttl_hours=0)

        # Should return None for expired cache
        retrieved_data = cache_manager.get_data("AAPL")
        assert retrieved_data is None

    def test_clear_cache(self, cache_manager, sample_dataframe):
        """Test cache clearing."""
        # Cache some data
        cache_manager.cache_data("AAPL", sample_dataframe)
        cache_manager.cache_data("MSFT", sample_dataframe)

        # Clear all cache
        cache_manager.clear_cache()

        # Verify cache is cleared
        assert cache_manager.get_data("AAPL") is None
        assert cache_manager.get_data("MSFT") is None

    def test_clear_cache_by_type(
        self, cache_manager, sample_dataframe, sample_backtest_result
    ):
        """Test clearing cache by type."""
        # Cache different types of data
        cache_manager.cache_data("AAPL", sample_dataframe)
        cache_manager.cache_data("MSFT", sample_dataframe)
        cache_manager.cache_backtest_result(
            "AAPL", "rsi", {"period": 14}, sample_backtest_result
        )

        # Clear only data cache
        cache_manager.clear_cache(cache_type="data")

        # Verify only data cache is cleared
        assert cache_manager.get_data("AAPL") is None
        assert cache_manager.get_data("MSFT") is None
        assert (
            cache_manager.get_backtest_result("AAPL", "rsi", {"period": 14}) is not None
        )

    def test_clear_cache_older_than(self, cache_manager, sample_dataframe):
        """Test clearing cache older than specified days."""
        # Cache some data
        cache_manager.cache_data("AAPL", sample_dataframe)

        # Clear cache older than 5 days (should not clear recent data)
        cache_manager.clear_cache(older_than_days=5)

        # Recent data should still be there
        assert cache_manager.get_data("AAPL") is not None

    def test_get_cache_stats(self, cache_manager, sample_dataframe):
        """Test getting cache statistics."""
        # Add some cached data
        cache_manager.cache_data("AAPL", sample_dataframe)
        cache_manager.cache_data("MSFT", sample_dataframe)

        stats = cache_manager.get_cache_stats()

        assert isinstance(stats, dict)
        assert "total_size_gb" in stats
        assert "max_size_gb" in stats
        assert "utilization_percent" in stats
        assert "by_type" in stats
        assert "by_source" in stats

    def test_cache_size_management(self, cache_manager, sample_dataframe):
        """Test cache size management."""
        # Test with very small cache limit
        small_cache = UnifiedCacheManager(
            cache_dir=cache_manager.cache_dir,
            max_size_gb=0.001,  # Very small limit
        )

        # Try to cache data that exceeds limit
        result = small_cache.cache_data("AAPL", sample_dataframe)

        # Should handle gracefully
        assert isinstance(result, str)

    def test_compression(self, cache_manager, sample_dataframe):
        """Test data compression."""
        # Cache data (compression is always enabled)
        key = cache_manager.cache_data("AAPL", sample_dataframe)

        # Verify compressed file exists
        compressed_path = cache_manager.data_dir / f"{key}.gz"
        assert compressed_path.exists()

        # Verify we can retrieve the data correctly
        retrieved_data = cache_manager.get_data("AAPL")
        assert isinstance(retrieved_data, pd.DataFrame)
        assert len(retrieved_data) == len(sample_dataframe)

    def test_data_filtering(self, cache_manager, sample_dataframe):
        """Test data filtering by date range."""
        # Cache data
        cache_manager.cache_data("AAPL", sample_dataframe)

        # Test date range filtering
        start_date = "2023-01-15"
        end_date = "2023-01-25"

        filtered_data = cache_manager.get_data(
            "AAPL", start_date=start_date, end_date=end_date
        )

        if filtered_data is not None:
            assert len(filtered_data) <= len(sample_dataframe)
            assert filtered_data.index.min() >= pd.to_datetime(start_date)
            assert filtered_data.index.max() <= pd.to_datetime(end_date)

    def test_concurrent_access(self, cache_manager, sample_dataframe):
        """Test concurrent cache access."""
        import threading

        keys = [f"concurrent_test_{i}" for i in range(5)]
        results = {}

        def cache_worker(key):
            success = cache_manager.cache_data(key, sample_dataframe)
            results[key] = success

        # Start multiple threads
        threads = []
        for key in keys:
            thread = threading.Thread(target=cache_worker, args=(key,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all operations succeeded
        assert len(results) == 5
        assert all(results.values())

    def test_error_handling(self, cache_manager):
        """Test error handling in cache operations."""
        # Test getting non-existent cache
        result = cache_manager.get_data("non_existent")
        assert result is None

    def test_cache_key_collision_handling(self, cache_manager, sample_dataframe):
        """Test handling of cache key collisions."""
        symbol = "AAPL"

        # Cache first dataset
        cache_manager.cache_data(symbol, sample_dataframe)
        original_data = cache_manager.get_data(symbol)

        # Cache different dataset with same symbol (should overwrite)
        modified_data = sample_dataframe.copy()
        modified_data["New_Column"] = 1

        cache_manager.cache_data(symbol, modified_data)
        retrieved_data = cache_manager.get_data(symbol)

        # Should have the new data
        assert "New_Column" in retrieved_data.columns
        assert "New_Column" not in original_data.columns
