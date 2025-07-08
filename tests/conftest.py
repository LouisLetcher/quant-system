"""Pytest configuration and shared fixtures."""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture for test data directory."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def reports_dir():
    """Fixture for reports output directory."""
    reports_dir = Path(__file__).parent.parent / "reports_output"
    reports_dir.mkdir(exist_ok=True)
    return reports_dir


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    np.random.seed(42)  # For reproducible tests

    base_price = 100
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = [base_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    data = pd.DataFrame(
        {
            "Open": [p * (1 + np.random.normal(0, 0.001)) for p in prices],
            "High": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            "Low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            "Close": prices,
            "Volume": [1000 + np.random.randint(-200, 200) for _ in prices],
        },
        index=dates,
    )

    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    data["High"] = data[["Open", "High", "Close"]].max(axis=1)
    data["Low"] = data[["Open", "Low", "Close"]].min(axis=1)

    return data


@pytest.fixture
def sample_portfolio_config():
    """Sample portfolio configuration for testing."""
    return {
        "name": "Test Portfolio",
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "initial_capital": 100000,
        "commission": 0.001,
        "strategy": {"name": "BuyAndHold", "parameters": {}},
        "risk_management": {
            "max_position_size": 0.1,
            "stop_loss": 0.05,
            "take_profit": 0.15,
        },
        "data_source": {
            "primary_source": "yahoo",
            "fallback_sources": ["alpha_vantage"],
        },
    }


@pytest.fixture
def mock_api_keys(monkeypatch):
    """Mock API keys for testing."""
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test_key")
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "test_key")
    monkeypatch.setenv("POLYGON_API_KEY", "test_key")
    monkeypatch.setenv("TIINGO_API_KEY", "test_key")
    monkeypatch.setenv("FINNHUB_API_KEY", "test_key")


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create temporary cache directory for testing."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def crypto_symbols():
    """Sample crypto symbols for testing."""
    return ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD"]


@pytest.fixture
def forex_symbols():
    """Sample forex symbols for testing."""
    return ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]


@pytest.fixture
def stock_symbols():
    """Sample stock symbols for testing."""
    return ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]


@pytest.fixture(autouse=True)
def setup_test_environment(reports_dir):
    """Setup test environment automatically."""
    # Ensure reports directory exists
    reports_dir.mkdir(exist_ok=True)

    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"

    yield

    # Cleanup after tests
    if "TESTING" in os.environ:
        del os.environ["TESTING"]


# Pytest markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "network: mark test as requiring network access")
