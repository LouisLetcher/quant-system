"""Unit tests for UnifiedCacheManager."""

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

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
        assert os.path.exists(cache_manager.metadata_db_path)

    def test_generate_cache_key(self, cache_manager):
        """Test cache key generation."""
        params = {
            "symbol": "AAPL",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "strategy": "rsi",
        }

        key1 = cache_manager._generate_cache_key("data", **params)
        key2 = cache_manager._generate_cache_key("data", **params)

        # Same parameters should generate same key
        assert key1 == key2

        # Different parameters should generate different keys
        params["symbol"] = "MSFT"
        key3 = cache_manager._generate_cache_key("data", **params)
        assert key1 != key3

    def test_cache_data(self, cache_manager, sample_dataframe):
        """Test caching DataFrame data."""
        key = "test_data_key"

        # Cache the data
        success = cache_manager.cache_data(key, sample_dataframe, ttl_hours=1)
        assert success == True

        # Verify file was created
        expected_path = os.path.join(
            cache_manager.cache_dir, "data", f"{key}.parquet.gz"
        )
        assert os.path.exists(expected_path)

        # Verify metadata was stored
        metadata = cache_manager._get_metadata(key)
        assert metadata is not None
        assert metadata["data_type"] == "data"
        assert metadata["compressed"] == True

    def test_get_data(self, cache_manager, sample_dataframe):
        """Test retrieving cached data."""
        key = "test_data_key"

        # Cache the data first
        cache_manager.cache_data(key, sample_dataframe)

        # Retrieve the data
        retrieved_data = cache_manager.get_data(key)

        assert isinstance(retrieved_data, pd.DataFrame)
        assert len(retrieved_data) == len(sample_dataframe)
        assert list(retrieved_data.columns) == list(sample_dataframe.columns)

    def test_cache_backtest_result(self, cache_manager, sample_backtest_result):
        """Test caching backtest results."""
        key = "test_backtest_key"

        # Cache the result
        success = cache_manager.cache_backtest_result(key, sample_backtest_result)
        assert success == True

        # Verify file was created
        expected_path = os.path.join(
            cache_manager.cache_dir, "backtests", f"{key}.json.gz"
        )
        assert os.path.exists(expected_path)

    def test_get_backtest_result(self, cache_manager, sample_backtest_result):
        """Test retrieving cached backtest results."""
        key = "test_backtest_key"

        # Cache the result first
        cache_manager.cache_backtest_result(key, sample_backtest_result)

        # Retrieve the result
        retrieved_result = cache_manager.get_backtest_result(key)

        assert isinstance(retrieved_result, dict)
        assert retrieved_result["symbol"] == sample_backtest_result["symbol"]
        assert (
            retrieved_result["total_return"] == sample_backtest_result["total_return"]
        )

    def test_cache_optimization_result(self, cache_manager):
        """Test caching optimization results."""
        key = "test_optimization_key"
        optimization_result = {
            "best_params": {"rsi_period": 14, "rsi_overbought": 70},
            "best_score": 1.5,
            "all_results": [
                {"params": {"rsi_period": 10}, "score": 1.2},
                {"params": {"rsi_period": 14}, "score": 1.5},
            ],
        }

        # Cache the result
        success = cache_manager.cache_optimization_result(key, optimization_result)
        assert success == True

        # Retrieve the result
        retrieved_result = cache_manager.get_optimization_result(key)

        assert isinstance(retrieved_result, dict)
        assert retrieved_result["best_score"] == 1.5

    def test_is_valid_cache(self, cache_manager, sample_dataframe):
        """Test cache validity checking."""
        key = "test_validity_key"

        # Non-existent cache should be invalid
        assert cache_manager.is_valid_cache(key) == False

        # Fresh cache should be valid
        cache_manager.cache_data(key, sample_dataframe, ttl_hours=1)
        assert cache_manager.is_valid_cache(key) == True

        # Expired cache should be invalid
        cache_manager.cache_data(key, sample_dataframe, ttl_hours=0)
        assert cache_manager.is_valid_cache(key) == False

    def test_delete_cache(self, cache_manager, sample_dataframe):
        """Test cache deletion."""
        key = "test_delete_key"

        # Cache some data
        cache_manager.cache_data(key, sample_dataframe)
        assert cache_manager.is_valid_cache(key) == True

        # Delete the cache
        success = cache_manager.delete_cache(key)
        assert success == True
        assert cache_manager.is_valid_cache(key) == False

    def test_clear_expired_cache(self, cache_manager, sample_dataframe):
        """Test clearing expired cache entries."""
        # Create some expired cache entries
        cache_manager.cache_data("expired_1", sample_dataframe, ttl_hours=0)
        cache_manager.cache_data("expired_2", sample_dataframe, ttl_hours=0)
        cache_manager.cache_data("valid_1", sample_dataframe, ttl_hours=24)

        # Clear expired entries
        cleared_count = cache_manager.clear_expired_cache()

        assert cleared_count == 2
        assert cache_manager.is_valid_cache("expired_1") == False
        assert cache_manager.is_valid_cache("expired_2") == False
        assert cache_manager.is_valid_cache("valid_1") == True

    def test_clear_cache_by_type(
        self, cache_manager, sample_dataframe, sample_backtest_result
    ):
        """Test clearing cache by type."""
        # Cache different types of data
        cache_manager.cache_data("data_1", sample_dataframe)
        cache_manager.cache_data("data_2", sample_dataframe)
        cache_manager.cache_backtest_result("backtest_1", sample_backtest_result)

        # Clear only data cache
        cleared_count = cache_manager.clear_cache_by_type("data")

        assert cleared_count == 2
        assert cache_manager.is_valid_cache("data_1") == False
        assert cache_manager.is_valid_cache("data_2") == False
        assert cache_manager.is_valid_cache("backtest_1") == True

    def test_clear_cache_older_than(self, cache_manager, sample_dataframe):
        """Test clearing cache older than specified days."""
        # Create cache entries with different ages
        cache_manager.cache_data("recent", sample_dataframe)

        # Manually update metadata to simulate old cache
        conn = sqlite3.connect(cache_manager.metadata_db_path)
        cursor = conn.cursor()
        old_timestamp = datetime.now() - timedelta(days=10)
        cursor.execute(
            """
            INSERT OR REPLACE INTO cache_metadata 
            (cache_key, data_type, file_path, created_at, expires_at, size_bytes, compressed, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                "old_cache",
                "data",
                "dummy_path",
                old_timestamp.isoformat(),
                (old_timestamp + timedelta(hours=24)).isoformat(),
                1000,
                True,
                "test",
            ),
        )
        conn.commit()
        conn.close()

        # Clear cache older than 5 days
        cleared_count = cache_manager.clear_cache_older_than(5)

        assert cleared_count == 1

    def test_get_cache_stats(self, cache_manager, sample_dataframe):
        """Test getting cache statistics."""
        # Add some cached data
        cache_manager.cache_data("test_1", sample_dataframe)
        cache_manager.cache_data("test_2", sample_dataframe)

        stats = cache_manager.get_cache_stats()

        assert isinstance(stats, dict)
        assert "total_size_gb" in stats
        assert "max_size_gb" in stats
        assert "utilization" in stats
        assert "by_type" in stats
        assert "by_source" in stats

    def test_cache_size_management(self, cache_manager, sample_dataframe):
        """Test cache size management."""
        # Test with very small cache limit
        small_cache = UnifiedCacheManager(
            cache_dir=cache_manager.cache_dir,
            max_size_gb=0.001,  # Very small limit
            default_ttl_hours=24,
        )

        # Try to cache data that exceeds limit
        result = small_cache.cache_data("large_data", sample_dataframe)

        # Should handle gracefully
        assert isinstance(result, bool)

    def test_compression(self, cache_manager, sample_dataframe):
        """Test data compression."""
        key = "compression_test"

        # Cache with compression
        cache_manager.cache_data(key, sample_dataframe, compress=True)

        # Verify compressed file exists
        compressed_path = os.path.join(
            cache_manager.cache_dir, "data", f"{key}.parquet.gz"
        )
        assert os.path.exists(compressed_path)

        # Verify we can retrieve the data correctly
        retrieved_data = cache_manager.get_data(key)
        assert isinstance(retrieved_data, pd.DataFrame)
        assert len(retrieved_data) == len(sample_dataframe)

    def test_metadata_integrity(self, cache_manager, sample_dataframe):
        """Test metadata database integrity."""
        key = "metadata_test"

        # Cache some data
        cache_manager.cache_data(key, sample_dataframe)

        # Verify metadata exists
        metadata = cache_manager._get_metadata(key)
        assert metadata is not None
        assert metadata["cache_key"] == key
        assert metadata["data_type"] == "data"
        assert "created_at" in metadata
        assert "expires_at" in metadata

    def test_concurrent_access(self, cache_manager, sample_dataframe):
        """Test concurrent cache access."""
        import threading
        import time

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
        # Test with invalid data
        invalid_data = "not a dataframe"

        with pytest.raises((TypeError, ValueError)):
            cache_manager.cache_data("invalid", invalid_data)

        # Test getting non-existent cache
        result = cache_manager.get_data("non_existent")
        assert result is None

    def test_cache_key_collision_handling(self, cache_manager, sample_dataframe):
        """Test handling of cache key collisions."""
        key = "collision_test"

        # Cache first dataset
        cache_manager.cache_data(key, sample_dataframe)
        original_data = cache_manager.get_data(key)

        # Cache different dataset with same key (should overwrite)
        modified_data = sample_dataframe.copy()
        modified_data["New_Column"] = 1

        cache_manager.cache_data(key, modified_data)
        retrieved_data = cache_manager.get_data(key)

        # Should have the new data
        assert "New_Column" in retrieved_data.columns
        assert "New_Column" not in original_data.columns
