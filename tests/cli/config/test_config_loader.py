"""Test suite for CLI config loader."""

from __future__ import annotations

from unittest.mock import mock_open, patch

import pytest

from src.cli.config.config_loader import (
    get_default_parameters,
    get_portfolio_config,
    load_assets_config,
)


class TestConfigLoader:
    """Test class for CLI config loader functionality."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        return {
            "portfolios": {
                "tech_stocks": {
                    "description": "Technology sector stocks",
                    "assets": [
                        {
                            "ticker": "AAPL",
                            "commission": 0.001,
                            "initial_capital": 10000,
                        },
                        {
                            "ticker": "MSFT",
                            "commission": 0.001,
                            "initial_capital": 10000,
                        },
                    ],
                }
            }
        }

    @patch("builtins.open", new_callable=mock_open, read_data='{"portfolios": {}}')
    @patch("json.load")
    def test_load_assets_config(self, mock_json_load, mock_file_open, sample_config):
        """Test loading assets configuration."""
        mock_json_load.return_value = sample_config

        result = load_assets_config()

        mock_file_open.assert_called_with("config/assets_config.json", "r")
        mock_json_load.assert_called_once()
        assert result == sample_config

        # Test with file not found
        mock_file_open.side_effect = FileNotFoundError()
        result = load_assets_config()
        assert result == {"portfolios": {}}

    @patch("src.cli.config.config_loader.load_assets_config")
    def test_get_portfolio_config(self, mock_load_config, sample_config):
        """Test getting portfolio configuration."""
        mock_load_config.return_value = sample_config

        result = get_portfolio_config("tech_stocks")

        # Assertions
        mock_load_config.assert_called_once()
        assert result["description"] == "Technology sector stocks"
        assert len(result["assets"]) == 2
        assert result["assets"][0]["ticker"] == "AAPL"

        # Test with non-existent portfolio
        result = get_portfolio_config("non_existent")
        assert result is None

    def test_get_default_parameters(self):
        """Test getting default parameters."""
        defaults = get_default_parameters()

        # Assertions
        assert isinstance(defaults, dict)
        assert "commission" in defaults
        assert "initial_capital" in defaults
        assert "period" in defaults
        assert "interval" in defaults
