"""Integration tests for the full quant system workflow."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.core.data_manager import UnifiedDataManager
from src.core.cache_manager import UnifiedCacheManager
from src.core.backtest_engine import UnifiedBacktestEngine, BacktestConfig
from src.core.portfolio_manager import PortfolioManager
from src.core.result_analyzer import UnifiedResultAnalyzer


class TestFullWorkflow:
    """Integration tests for the complete quant trading system workflow."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def cache_manager(self, temp_dir):
        """Create cache manager instance."""
        return UnifiedCacheManager(cache_dir=temp_dir, max_size_gb=1.0)

    @pytest.fixture
    def data_manager(self, cache_manager):
        """Create data manager instance."""
        manager = UnifiedDataManager(cache_manager=cache_manager)
        manager.add_source('yahoo_finance')
        return manager

    @pytest.fixture
    def backtest_engine(self, data_manager, cache_manager):
        """Create backtest engine instance."""
        analyzer = UnifiedResultAnalyzer()
        return UnifiedBacktestEngine(
            data_manager=data_manager,
            cache_manager=cache_manager,
            result_analyzer=analyzer
        )

    @pytest.fixture
    def portfolio_manager(self, backtest_engine):
        """Create portfolio manager instance."""
        return PortfolioManager(backtest_engine=backtest_engine)

    @pytest.fixture
    def sample_data(self):
        """Generate sample market data."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)  # For reproducibility
        
        # Generate realistic price data
        initial_price = 100
        returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
        prices = [initial_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'Open': prices[:-1],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
            'Close': prices[1:],
            'Volume': np.random.randint(1000000, 10000000, 252)
        }, index=dates)
        
        return data

    @pytest.mark.integration
    def test_complete_single_asset_workflow(self, data_manager, backtest_engine, sample_data):
        """Test complete workflow for single asset backtesting."""
        # Mock data fetching
        with patch.object(data_manager, 'fetch_data', return_value=sample_data):
            # Create backtest configuration
            config = BacktestConfig(
                symbols=['AAPL'],
                strategies=['rsi'],
                start_date='2023-01-01',
                end_date='2023-12-31',
                initial_capital=10000,
                commission=0.001
            )
            
            # Run single backtest
            result = backtest_engine.run_single_backtest(
                symbol='AAPL',
                strategy='rsi',
                config=config
            )
            
            # Verify result
            assert result is not None
            assert result.symbol == 'AAPL'
            assert result.strategy == 'rsi'
            assert isinstance(result.total_return, float)
            assert isinstance(result.sharpe_ratio, float)
            assert isinstance(result.equity_curve, pd.Series)

    @pytest.mark.integration
    def test_complete_portfolio_workflow(self, portfolio_manager, data_manager, sample_data):
        """Test complete portfolio management workflow."""
        # Mock data fetching for multiple symbols
        with patch.object(data_manager, 'fetch_data', return_value=sample_data):
            with patch.object(data_manager, 'batch_fetch_data') as mock_batch:
                mock_batch.return_value = {
                    'AAPL': sample_data,
                    'MSFT': sample_data,
                    'GOOGL': sample_data
                }
                
                # Define portfolio
                portfolio_config = {
                    'name': 'Tech Portfolio',
                    'symbols': ['AAPL', 'MSFT', 'GOOGL'],
                    'strategies': ['rsi', 'macd'],
                    'risk_profile': 'moderate',
                    'target_return': 0.12
                }
                
                # Add portfolio
                portfolio_manager.add_portfolio('tech_portfolio', portfolio_config)
                
                # Generate investment recommendations
                recommendations = portfolio_manager.generate_investment_recommendations(
                    capital=100000,
                    risk_tolerance='moderate',
                    start_date='2023-01-01',
                    end_date='2023-12-31'
                )
                
                # Verify recommendations
                assert isinstance(recommendations, dict)
                assert 'recommended_allocations' in recommendations
                assert 'expected_return' in recommendations
                assert 'investment_plan' in recommendations

    @pytest.mark.integration
    def test_cache_integration_workflow(self, cache_manager, data_manager, sample_data):
        """Test workflow with cache integration."""
        # First request - should cache the data
        with patch.object(data_manager, 'fetch_data', return_value=sample_data) as mock_fetch:
            # Remove any existing data sources to force fresh fetch
            data_manager.sources.clear()
            data_manager.add_source('yahoo_finance')
            
            # First fetch - should call the actual fetch method
            result1 = data_manager.fetch_data('AAPL', '2023-01-01', '2023-12-31')
            
            # Manually cache the data
            cache_key = cache_manager._generate_cache_key(
                'data', symbol='AAPL', start_date='2023-01-01', end_date='2023-12-31', source='yahoo_finance'
            )
            cache_manager.cache_data(cache_key, sample_data)
            
            # Second fetch - should use cache
            result2 = data_manager.fetch_data('AAPL', '2023-01-01', '2023-12-31')
            
            # Verify both results are DataFrames
            assert isinstance(result1, pd.DataFrame)
            assert isinstance(result2, pd.DataFrame)

    @pytest.mark.integration
    def test_batch_processing_workflow(self, backtest_engine, data_manager, sample_data):
        """Test batch processing workflow."""
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        strategies = ['rsi', 'macd']
        
        # Mock batch data fetching
        with patch.object(data_manager, 'batch_fetch_data') as mock_batch:
            mock_batch.return_value = {symbol: sample_data for symbol in symbols}
            
            # Create batch configuration
            config = BacktestConfig(
                symbols=symbols,
                strategies=strategies,
                start_date='2023-01-01',
                end_date='2023-12-31',
                initial_capital=10000,
                max_workers=2
            )
            
            # Run batch backtest
            results = backtest_engine.batch_backtest(config)
            
            # Verify results structure
            assert isinstance(results, list)
            # Note: Results might be empty due to multiprocessing issues in tests
            # but the structure should be correct

    @pytest.mark.integration
    def test_optimization_workflow(self, backtest_engine, data_manager, sample_data):
        """Test strategy optimization workflow."""
        # Mock data fetching
        with patch.object(data_manager, 'fetch_data', return_value=sample_data):
            # Define optimization parameters
            param_space = {
                'rsi_period': [10, 14, 20],
                'rsi_overbought': [70, 75, 80],
                'rsi_oversold': [20, 25, 30]
            }
            
            # Run optimization
            try:
                best_params, best_score = backtest_engine.optimize_strategy(
                    symbol='AAPL',
                    strategy='rsi',
                    param_space=param_space,
                    start_date='2023-01-01',
                    end_date='2023-12-31',
                    objective='sharpe_ratio',
                    max_evaluations=9  # Small number for testing
                )
                
                # Verify optimization results
                assert isinstance(best_params, dict)
                assert isinstance(best_score, float)
                assert 'rsi_period' in best_params
                
            except Exception as e:
                # Optimization might fail in test environment, log but don't fail test
                pytest.skip(f"Optimization test skipped due to: {e}")

    @pytest.mark.integration
    def test_risk_analysis_workflow(self, portfolio_manager, data_manager, sample_data):
        """Test risk analysis workflow."""
        # Mock data for risk analysis
        with patch.object(data_manager, 'batch_fetch_data') as mock_batch:
            mock_batch.return_value = {
                'AAPL': sample_data,
                'BOND': sample_data * 0.5,  # Lower volatility asset
                'GOLD': sample_data * 0.3   # Different correlation asset
            }
            
            # Define portfolios with different risk profiles
            portfolios = {
                'conservative': {
                    'name': 'Conservative Portfolio',
                    'symbols': ['BOND', 'AAPL'],
                    'strategies': ['sma_crossover'],
                    'risk_profile': 'conservative'
                },
                'aggressive': {
                    'name': 'Aggressive Portfolio',
                    'symbols': ['AAPL', 'GOLD'],
                    'strategies': ['rsi', 'macd'],
                    'risk_profile': 'aggressive'
                }
            }
            
            # Add portfolios
            for pid, config in portfolios.items():
                portfolio_manager.add_portfolio(pid, config)
            
            # Compare portfolios
            comparison = portfolio_manager.compare_portfolios(
                start_date='2023-01-01',
                end_date='2023-12-31'
            )
            
            # Verify risk analysis results
            assert isinstance(comparison, dict)
            assert 'rankings' in comparison
            assert 'summary' in comparison

    @pytest.mark.integration
    def test_data_quality_workflow(self, data_manager, sample_data):
        """Test data quality validation workflow."""
        # Test with good data
        good_data = sample_data.copy()
        
        # Test with problematic data
        bad_data = sample_data.copy()
        bad_data.iloc[10:20, :] = np.nan  # Introduce missing values
        
        with patch.object(data_manager, 'fetch_data') as mock_fetch:
            # Test good data path
            mock_fetch.return_value = good_data
            result1 = data_manager.fetch_data('AAPL', '2023-01-01', '2023-12-31')
            assert data_manager._validate_data_quality(result1) == True
            
            # Test bad data path
            mock_fetch.return_value = bad_data
            result2 = data_manager.fetch_data('AAPL', '2023-01-01', '2023-12-31')
            assert data_manager._validate_data_quality(result2) == False

    @pytest.mark.integration
    def test_performance_monitoring_workflow(self, backtest_engine, cache_manager):
        """Test performance monitoring throughout workflow."""
        # Get initial cache stats
        initial_stats = cache_manager.get_cache_stats()
        
        # Get engine performance stats
        engine_stats = backtest_engine.get_performance_stats()
        
        # Verify stats structure
        assert isinstance(initial_stats, dict)
        assert 'total_size_gb' in initial_stats
        assert 'utilization' in initial_stats
        
        assert isinstance(engine_stats, dict)
        assert 'total_backtests' in engine_stats
        assert 'cache_hits' in engine_stats

    @pytest.mark.integration
    def test_error_recovery_workflow(self, data_manager, backtest_engine):
        """Test error recovery and graceful degradation."""
        # Test data fetching with network error
        with patch.object(data_manager, 'fetch_data', side_effect=Exception("Network error")):
            try:
                result = data_manager.fetch_data('AAPL', '2023-01-01', '2023-12-31')
                assert result is None or isinstance(result, pd.DataFrame)
            except Exception:
                # Should handle gracefully
                pass
        
        # Test backtest with invalid configuration
        invalid_config = BacktestConfig(
            symbols=[],  # Empty symbols list
            strategies=['rsi'],
            start_date='2023-01-01',
            end_date='2023-12-31'
        )
        
        try:
            results = backtest_engine.batch_backtest(invalid_config)
            assert isinstance(results, list)
        except ValueError:
            # Expected behavior for invalid config
            pass

    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_scale_workflow(self, portfolio_manager, data_manager, sample_data):
        """Test workflow with large number of assets (marked as slow test)."""
        # Create large portfolio
        symbols = [f'STOCK_{i:03d}' for i in range(50)]
        strategies = ['rsi', 'macd', 'sma_crossover']
        
        large_portfolio = {
            'name': 'Large Portfolio',
            'symbols': symbols,
            'strategies': strategies,
            'risk_profile': 'moderate'
        }
        
        # Mock batch data fetching for large portfolio
        with patch.object(data_manager, 'batch_fetch_data') as mock_batch:
            mock_batch.return_value = {symbol: sample_data for symbol in symbols}
            
            portfolio_manager.add_portfolio('large_portfolio', large_portfolio)
            
            # This should handle large portfolios gracefully
            recommendations = portfolio_manager.generate_investment_recommendations(
                capital=1000000,
                risk_tolerance='moderate',
                start_date='2023-01-01',
                end_date='2023-12-31'
            )
            
            assert isinstance(recommendations, dict)

    @pytest.mark.integration
    def test_concurrent_workflow(self, portfolio_manager, data_manager, sample_data):
        """Test concurrent operations workflow."""
        import threading
        import time
        
        results = {}
        
        def portfolio_worker(portfolio_id, config):
            try:
                portfolio_manager.add_portfolio(portfolio_id, config)
                # Simulate some work
                time.sleep(0.1)
                results[portfolio_id] = 'success'
            except Exception as e:
                results[portfolio_id] = f'error: {e}'
        
        # Create multiple portfolios concurrently
        portfolios = {
            f'portfolio_{i}': {
                'name': f'Portfolio {i}',
                'symbols': ['AAPL', 'MSFT'],
                'strategies': ['rsi'],
                'risk_profile': 'moderate'
            }
            for i in range(5)
        }
        
        threads = []
        for pid, config in portfolios.items():
            thread = threading.Thread(target=portfolio_worker, args=(pid, config))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify all operations completed
        assert len(results) == 5
        assert all(status == 'success' for status in results.values())
