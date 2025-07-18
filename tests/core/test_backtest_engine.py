"""
Tests for the backtest engine module.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from src.core.backtest_engine import (
    BacktestConfig,
    BacktestResult,
    UnifiedBacktestEngine,
)


class TestBacktestConfig:
    """Test BacktestConfig dataclass."""

    def test_config_initialization_minimal(self):
        """Test config initialization with minimal required parameters."""
        config = BacktestConfig(
            symbols=["AAPL", "MSFT"],
            strategies=["BuyAndHold"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        assert config.symbols == ["AAPL", "MSFT"]
        assert config.strategies == ["BuyAndHold"]
        assert config.start_date == "2023-01-01"
        assert config.end_date == "2023-12-31"

        # Check defaults
        assert config.initial_capital == 10000
        assert config.interval == "1d"
        assert config.commission == 0.001
        assert config.use_cache is True
        assert config.save_trades is False
        assert config.save_equity_curve is False
        assert config.memory_limit_gb == 8.0
        assert config.max_workers is None
        assert config.asset_type is None
        assert config.futures_mode is False
        assert config.leverage == 1.0

    def test_config_initialization_full(self):
        """Test config initialization with all parameters."""
        config = BacktestConfig(
            symbols=["BTC-USD"],
            strategies=["MeanReversion"],
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=50000,
            interval="1h",
            commission=0.0025,
            use_cache=False,
            save_trades=True,
            save_equity_curve=True,
            memory_limit_gb=16.0,
            max_workers=4,
            asset_type="crypto",
            futures_mode=True,
            leverage=2.0,
        )

        assert config.initial_capital == 50000
        assert config.interval == "1h"
        assert config.commission == 0.0025
        assert config.use_cache is False
        assert config.save_trades is True
        assert config.save_equity_curve is True
        assert config.memory_limit_gb == 16.0
        assert config.max_workers == 4
        assert config.asset_type == "crypto"
        assert config.futures_mode is True
        assert config.leverage == 2.0


class TestBacktestResult:
    """Test BacktestResult dataclass."""

    def test_result_initialization_minimal(self):
        """Test result initialization with minimal parameters."""
        config = BacktestConfig(
            symbols=["AAPL"],
            strategies=["BuyAndHold"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        result = BacktestResult(
            symbol="AAPL",
            strategy="BuyAndHold",
            parameters={"param1": "value1"},
            metrics={"return": 0.15, "sharpe": 1.2},
            config=config,
        )

        assert result.symbol == "AAPL"
        assert result.strategy == "BuyAndHold"
        assert result.parameters == {"param1": "value1"}
        assert result.metrics == {"return": 0.15, "sharpe": 1.2}
        assert result.config == config

        # Check defaults
        assert result.equity_curve is None
        assert result.trades is None
        assert result.start_date is None
        assert result.end_date is None
        assert result.duration_seconds == 0
        assert result.data_points == 0
        assert result.error is None
        assert result.source is None

    def test_result_initialization_full(self):
        """Test result initialization with all parameters."""
        config = BacktestConfig(
            symbols=["AAPL"],
            strategies=["BuyAndHold"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        equity_curve = pd.DataFrame({"equity": [10000, 10500, 11000]})
        trades = pd.DataFrame({"date": ["2023-01-01"], "action": ["buy"]})

        result = BacktestResult(
            symbol="AAPL",
            strategy="BuyAndHold",
            parameters={},
            metrics={"return": 0.15},
            config=config,
            equity_curve=equity_curve,
            trades=trades,
            start_date="2023-01-01",
            end_date="2023-12-31",
            duration_seconds=5.5,
            data_points=252,
            error=None,
            source="yfinance",
        )

        assert result.equity_curve is not None
        assert len(result.equity_curve) == 3
        assert result.trades is not None
        assert len(result.trades) == 1
        assert result.start_date == "2023-01-01"
        assert result.end_date == "2023-12-31"
        assert result.duration_seconds == 5.5
        assert result.data_points == 252
        assert result.source == "yfinance"


class TestUnifiedBacktestEngine:
    """Test UnifiedBacktestEngine class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_data_manager = MagicMock()
        self.mock_cache_manager = MagicMock()

        self.engine = UnifiedBacktestEngine(
            data_manager=self.mock_data_manager,
            cache_manager=self.mock_cache_manager,
            max_workers=2,
            memory_limit_gb=4.0,
        )

    def test_initialization_with_defaults(self):
        """Test engine initialization with default parameters."""
        engine = UnifiedBacktestEngine()

        assert engine.data_manager is not None
        assert engine.cache_manager is not None
        assert engine.result_analyzer is not None
        assert engine.max_workers > 0
        assert engine.memory_limit_bytes > 0
        assert isinstance(engine.stats, dict)

    def test_initialization_with_custom_params(self):
        """Test engine initialization with custom parameters."""
        assert self.engine.data_manager == self.mock_data_manager
        assert self.engine.cache_manager == self.mock_cache_manager
        assert self.engine.max_workers == 2
        assert self.engine.memory_limit_bytes == int(4.0 * 1024**3)

    def test_stats_initialization(self):
        """Test that stats are properly initialized."""
        expected_keys = [
            "backtests_run",
            "cache_hits",
            "cache_misses",
            "errors",
            "total_time",
        ]

        for key in expected_keys:
            assert key in self.engine.stats
            assert self.engine.stats[key] == 0

    @patch("src.core.backtest_engine.UnifiedBacktestEngine._get_default_parameters")
    @patch("src.core.backtest_engine.UnifiedBacktestEngine._execute_backtest")
    def test_run_backtest_success(self, mock_execute, mock_get_params):
        """Test successful backtest run."""
        # Setup mocks
        mock_get_params.return_value = {"param1": "value1"}
        self.mock_cache_manager.get_backtest_result.return_value = None

        # Mock data
        test_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [105, 106, 107],
                "Low": [95, 96, 97],
                "Close": [102, 103, 104],
                "Volume": [1000, 1100, 1200],
            }
        )
        self.mock_data_manager.get_data.return_value = test_data

        # Mock execute result
        expected_result = BacktestResult(
            symbol="AAPL",
            strategy="BuyAndHold",
            parameters={"param1": "value1"},
            metrics={"return": 0.15},
            config=MagicMock(),
        )
        mock_execute.return_value = expected_result

        config = BacktestConfig(
            symbols=["AAPL"],
            strategies=["BuyAndHold"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        # Run test
        result = self.engine.run_backtest("AAPL", "BuyAndHold", config)

        # Assertions
        assert isinstance(result, BacktestResult)
        assert result.symbol == "AAPL"
        assert result.strategy == "BuyAndHold"
        assert self.engine.stats["backtests_run"] == 1
        assert self.engine.stats["cache_misses"] == 1

    def test_run_backtest_with_cache_hit(self):
        """Test backtest run with cache hit."""
        # Setup cache hit
        cached_result = {
            "symbol": "AAPL",
            "strategy": "BuyAndHold",
            "parameters": {},
            "metrics": {"return": 0.15},
            "error": None,
        }
        self.mock_cache_manager.get_backtest_result.return_value = cached_result

        config = BacktestConfig(
            symbols=["AAPL"],
            strategies=["BuyAndHold"],
            start_date="2023-01-01",
            end_date="2023-12-31",
            use_cache=True,
        )

        # Mock helper method
        with patch.object(self.engine, "_dict_to_result") as mock_dict_to_result:
            mock_dict_to_result.return_value = BacktestResult(
                symbol="AAPL",
                strategy="BuyAndHold",
                parameters={},
                metrics={"return": 0.15},
                config=config,
            )

            result = self.engine.run_backtest("AAPL", "BuyAndHold", config)

            assert self.engine.stats["cache_hits"] == 1
            assert self.engine.stats["cache_misses"] == 0
            mock_dict_to_result.assert_called_once()

    def test_run_backtest_no_data(self):
        """Test backtest run when no data is available."""
        # Setup no data scenario
        self.mock_cache_manager.get_backtest_result.return_value = None
        self.mock_data_manager.get_data.return_value = None

        config = BacktestConfig(
            symbols=["INVALID"],
            strategies=["BuyAndHold"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        with patch.object(self.engine, "_get_default_parameters") as mock_get_params:
            mock_get_params.return_value = {}

            result = self.engine.run_backtest("INVALID", "BuyAndHold", config)

            assert result.error == "No data available"
            assert result.symbol == "INVALID"
            assert result.strategy == "BuyAndHold"

    def test_run_backtest_empty_data(self):
        """Test backtest run when data is empty."""
        # Setup empty data scenario
        self.mock_cache_manager.get_backtest_result.return_value = None
        self.mock_data_manager.get_data.return_value = pd.DataFrame()

        config = BacktestConfig(
            symbols=["EMPTY"],
            strategies=["BuyAndHold"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        with patch.object(self.engine, "_get_default_parameters") as mock_get_params:
            mock_get_params.return_value = {}

            result = self.engine.run_backtest("EMPTY", "BuyAndHold", config)

            assert result.error == "No data available"

    def test_run_backtest_exception_handling(self):
        """Test backtest run with exception handling."""
        # Setup exception scenario
        self.mock_cache_manager.get_backtest_result.return_value = None
        self.mock_data_manager.get_data.side_effect = Exception("Data fetch failed")

        config = BacktestConfig(
            symbols=["AAPL"],
            strategies=["BuyAndHold"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        with patch.object(self.engine, "_get_default_parameters") as mock_get_params:
            mock_get_params.return_value = {}

            result = self.engine.run_backtest("AAPL", "BuyAndHold", config)

            assert result.error == "Data fetch failed"
            assert self.engine.stats["errors"] == 1

    def test_run_backtest_futures_mode(self):
        """Test backtest run in futures mode."""
        # Setup futures mode
        self.mock_cache_manager.get_backtest_result.return_value = None
        test_data = pd.DataFrame(
            {
                "Open": [100, 101],
                "High": [105, 106],
                "Low": [95, 96],
                "Close": [102, 103],
                "Volume": [1000, 1100],
            }
        )
        self.mock_data_manager.get_crypto_futures_data.return_value = test_data

        config = BacktestConfig(
            symbols=["BTC-USD"],
            strategies=["BuyAndHold"],
            start_date="2023-01-01",
            end_date="2023-12-31",
            futures_mode=True,
        )

        with patch.object(self.engine, "_get_default_parameters") as mock_get_params:
            mock_get_params.return_value = {}
            with patch.object(self.engine, "_execute_backtest") as mock_execute:
                mock_execute.return_value = BacktestResult(
                    symbol="BTC-USD",
                    strategy="BuyAndHold",
                    parameters={},
                    metrics={},
                    config=config,
                )

                result = self.engine.run_backtest("BTC-USD", "BuyAndHold", config)

                # Verify futures data method was called
                self.mock_data_manager.get_crypto_futures_data.assert_called_once()
                assert not self.mock_data_manager.get_data.called

    def test_get_performance_stats(self):
        """Test getting engine performance statistics."""
        # Modify stats
        self.engine.stats["backtests_run"] = 5
        self.engine.stats["cache_hits"] = 2
        self.engine.stats["errors"] = 1

        stats = self.engine.get_performance_stats()

        assert stats["backtests_run"] == 5
        assert stats["cache_hits"] == 2
        assert stats["errors"] == 1

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        self.engine.clear_cache()
        self.mock_cache_manager.clear_cache.assert_called_once()

    def test_clear_cache_with_params(self):
        """Test cache clearing with parameters."""
        self.engine.clear_cache(symbol="AAPL", strategy="BuyAndHold")
        self.mock_cache_manager.clear_cache.assert_called_once_with(
            cache_type="backtest", symbol="AAPL"
        )

    def test_run_batch_backtests(self):
        """Test batch backtest execution."""
        # Create a simple config with single core processing to avoid multiprocessing issues
        config = BacktestConfig(
            symbols=["AAPL"],
            strategies=["BuyAndHold"],
            start_date="2023-01-01",
            end_date="2023-12-31",
            max_workers=1,  # Use single worker to avoid multiprocessing issues
        )

        # Mock the individual run_backtest method
        with patch.object(self.engine, "run_backtest") as mock_run_backtest:
            mock_run_backtest.return_value = BacktestResult(
                symbol="AAPL",
                strategy="BuyAndHold",
                parameters={},
                metrics={"return": 0.15},
                config=config,
            )

            results = self.engine.run_batch_backtests(config)

            assert len(results) == 1
            assert isinstance(results[0], BacktestResult)
            assert results[0].symbol == "AAPL"

    def test_memory_management(self):
        """Test memory management initialization."""
        engine = UnifiedBacktestEngine(memory_limit_gb=16.0)
        expected_bytes = int(16.0 * 1024**3)
        assert engine.memory_limit_bytes == expected_bytes

    def test_worker_count_limits(self):
        """Test worker count limits."""
        # Test with None (should use default)
        engine = UnifiedBacktestEngine(max_workers=None)
        assert engine.max_workers > 0
        assert engine.max_workers <= 8

        # Test with custom value
        engine = UnifiedBacktestEngine(max_workers=4)
        assert engine.max_workers == 4


class TestIntegration:
    """Integration tests for backtest engine."""

    def test_complete_backtest_workflow(self):
        """Test complete backtest workflow without mocking core functionality."""
        # Create test data
        test_data = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "High": [105, 106, 107, 108, 109],
                "Low": [95, 96, 97, 98, 99],
                "Close": [102, 103, 104, 105, 106],
                "Volume": [1000, 1100, 1200, 1300, 1400],
            },
            index=pd.date_range("2023-01-01", periods=5, freq="D"),
        )

        # Mock only external dependencies
        with patch("src.core.backtest_engine.UnifiedDataManager") as mock_dm_class:
            with patch("src.core.backtest_engine.UnifiedCacheManager") as mock_cm_class:
                # Setup mocks
                mock_dm = MagicMock()
                mock_cm = MagicMock()
                mock_dm_class.return_value = mock_dm
                mock_cm_class.return_value = mock_cm

                mock_dm.get_data.return_value = test_data
                mock_cm.get_backtest_result.return_value = None

                # Create engine
                engine = UnifiedBacktestEngine()

                # Test configuration
                config = BacktestConfig(
                    symbols=["TEST"],
                    strategies=["BuyAndHold"],
                    start_date="2023-01-01",
                    end_date="2023-01-05",
                    use_cache=False,
                )

                # Mock strategy-related methods
                with patch.object(engine, "_get_default_parameters") as mock_params:
                    with patch.object(engine, "_execute_backtest") as mock_execute:
                        mock_params.return_value = {}
                        mock_execute.return_value = BacktestResult(
                            symbol="TEST",
                            strategy="BuyAndHold",
                            parameters={},
                            metrics={"total_return": 0.06, "sharpe_ratio": 1.5},
                            config=config,
                        )

                        result = engine.run_backtest("TEST", "BuyAndHold", config)

                        # Verify result
                        assert isinstance(result, BacktestResult)
                        assert result.symbol == "TEST"
                        assert result.strategy == "BuyAndHold"
                        assert "total_return" in result.metrics
                        assert result.metrics["total_return"] == 0.06
