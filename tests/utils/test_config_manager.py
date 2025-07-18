"""
Tests for the config manager module.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils.config_manager import ConfigManager


class TestConfigManager:
    """Test ConfigManager class."""

    def test_initialization_default(self):
        """Test default initialization."""
        config = ConfigManager()

        assert isinstance(config.config, dict)
        assert "data" in config.config
        assert "backtest" in config.config
        assert "logging" in config.config

        # Check default values
        assert config.config["data"]["default_interval"] == "1d"
        assert config.config["backtest"]["default_commission"] == 0.001
        assert config.config["backtest"]["initial_capital"] == 10000
        assert config.config["logging"]["level"] == "INFO"

    def test_initialization_with_nonexistent_file(self):
        """Test initialization with nonexistent config file."""
        with pytest.raises(FileNotFoundError):
            ConfigManager(config_path="/nonexistent/config.json")

    def test_load_json_config_file(self):
        """Test loading JSON configuration file."""
        # Create temporary JSON config file
        config_data = {
            "data": {"default_interval": "1h", "cache_dir": "/tmp/custom_cache"},
            "backtest": {"initial_capital": 50000},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_file = f.name

        try:
            config = ConfigManager(config_path=temp_file)

            # Check that values were loaded and merged
            assert config.config["data"]["default_interval"] == "1h"
            assert config.config["data"]["cache_dir"] == "/tmp/custom_cache"
            assert config.config["backtest"]["initial_capital"] == 50000
            # Default commission should still be there
            assert config.config["backtest"]["default_commission"] == 0.001

        finally:
            os.unlink(temp_file)

    def test_load_yaml_config_file(self):
        """Test loading YAML configuration file."""
        # Create temporary YAML config file
        config_data = {
            "data": {"default_interval": "5m", "new_setting": "test_value"},
            "logging": {"level": "DEBUG"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name

        try:
            config = ConfigManager(config_path=temp_file)

            # Check that values were loaded
            assert config.config["data"]["default_interval"] == "5m"
            assert config.config["data"]["new_setting"] == "test_value"
            assert config.config["logging"]["level"] == "DEBUG"

        finally:
            os.unlink(temp_file)

    def test_load_unsupported_file_format(self):
        """Test loading unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"some content")
            temp_file = f.name

        try:
            with pytest.raises(
                ValueError, match="Unsupported configuration file format"
            ):
                ConfigManager(config_path=temp_file)
        finally:
            os.unlink(temp_file)

    def test_load_from_environment_variables(self):
        """Test loading configuration from environment variables."""
        # Set test environment variables
        test_env_vars = {
            "QUANTPY_DATA_CACHE_SIZE": "1000",
            "QUANTPY_BACKTEST_COMMISSION": "0.005",
            "QUANTPY_LOGGING_DEBUG": "true",
            "QUANTPY_TEST_SECTION_ENABLED": "false",
            "QUANTPY_API_TIMEOUT": "30.5",
        }

        # Store original values to restore later
        original_values = {}
        for key in test_env_vars:
            original_values[key] = os.environ.get(key)
            os.environ[key] = test_env_vars[key]

        try:
            config = ConfigManager()

            # Check that environment variables were loaded
            assert (
                config.config["data"]["cache_size"] == 1000
            )  # String converted to int
            assert (
                config.config["backtest"]["commission"] == 0.005
            )  # String converted to float
            assert config.config["logging"]["debug"] is True  # String converted to bool
            assert (
                config.config["test"]["section_enabled"] is False
            )  # String converted to bool
            assert config.config["api"]["timeout"] == 30.5  # String converted to float

        finally:
            # Restore original environment
            for key, original_value in original_values.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value

    def test_get_method_dot_notation(self):
        """Test get method with dot notation."""
        config = ConfigManager()

        # Test existing values
        assert config.get("data.default_interval") == "1d"
        assert config.get("backtest.initial_capital") == 10000
        assert config.get("logging.level") == "INFO"

        # Test nonexistent values
        assert config.get("nonexistent.key") is None
        assert config.get("data.nonexistent") is None

        # Test with default values
        assert config.get("nonexistent.key", "default_value") == "default_value"
        assert config.get("data.nonexistent", 42) == 42

    def test_get_method_root_level(self):
        """Test get method for root level keys."""
        config = ConfigManager()

        # Test getting entire sections
        data_section = config.get("data")
        assert isinstance(data_section, dict)
        assert "default_interval" in data_section

    def test_set_method(self):
        """Test set method."""
        config = ConfigManager()

        # Set new value in existing section
        config.set("data", "new_key", "new_value")
        assert config.config["data"]["new_key"] == "new_value"

        # Set value in new section
        config.set("new_section", "test_key", 123)
        assert config.config["new_section"]["test_key"] == 123

    def test_save_to_json_file(self):
        """Test saving configuration to JSON file."""
        config = ConfigManager()
        config.set("test", "value", "test_data")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            config.save_to_file(temp_file)

            # Verify file was saved correctly
            with open(temp_file) as f:
                saved_config = json.load(f)

            assert saved_config["test"]["value"] == "test_data"
            assert "data" in saved_config
            assert "backtest" in saved_config

        finally:
            os.unlink(temp_file)

    def test_save_to_yaml_file(self):
        """Test saving configuration to YAML file."""
        config = ConfigManager()
        config.set("test", "yaml_value", "yaml_data")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_file = f.name

        try:
            config.save_to_file(temp_file)

            # Verify file was saved correctly
            with open(temp_file) as f:
                saved_config = yaml.safe_load(f)

            assert saved_config["test"]["yaml_value"] == "yaml_data"
            assert "data" in saved_config

        finally:
            os.unlink(temp_file)

    def test_save_unsupported_format(self):
        """Test saving to unsupported file format."""
        config = ConfigManager()

        with pytest.raises(ValueError, match="Unsupported file format"):
            config.save_to_file("config.txt")

    def test_nested_dict_update(self):
        """Test nested dictionary update functionality."""
        config = ConfigManager()

        # Create test configuration with nested structure
        test_config = {
            "data": {
                "cache_dir": "/new/cache/dir",
                "new_nested": {"deep_value": "test"},
            },
            "completely_new": {"setting": "value"},
        }

        config._update_nested_dict(config.config, test_config)

        # Check that nested values were updated correctly
        assert config.config["data"]["cache_dir"] == "/new/cache/dir"
        assert config.config["data"]["new_nested"]["deep_value"] == "test"
        assert config.config["completely_new"]["setting"] == "value"

        # Check that existing values not in update dict are preserved
        assert config.config["data"]["default_interval"] == "1d"
        assert config.config["backtest"]["initial_capital"] == 10000

    def test_environment_variable_type_conversion(self):
        """Test type conversion for environment variables."""
        test_cases = [
            ("QUANTPY_TEST_BOOL_TRUE", "true", True),
            ("QUANTPY_TEST_BOOL_YES", "yes", True),
            ("QUANTPY_TEST_BOOL_1", "1", True),
            ("QUANTPY_TEST_BOOL_FALSE", "false", False),
            ("QUANTPY_TEST_BOOL_NO", "no", False),
            ("QUANTPY_TEST_BOOL_0", "0", False),
            ("QUANTPY_TEST_INT", "42", 42),
            ("QUANTPY_TEST_FLOAT", "3.14", 3.14),
            ("QUANTPY_TEST_STRING", "hello", "hello"),
        ]

        # Store original values
        original_values = {}
        for env_var, value, _ in test_cases:
            original_values[env_var] = os.environ.get(env_var)
            os.environ[env_var] = value

        try:
            config = ConfigManager()

            # Check type conversions
            assert config.config["test"]["bool_true"] is True
            assert config.config["test"]["bool_yes"] is True
            assert config.config["test"]["bool_1"] is True
            assert config.config["test"]["bool_false"] is False
            assert config.config["test"]["bool_no"] is False
            assert config.config["test"]["bool_0"] is False
            assert config.config["test"]["int"] == 42
            assert config.config["test"]["float"] == 3.14
            assert config.config["test"]["string"] == "hello"

        finally:
            # Restore environment
            for env_var, _, _ in test_cases:
                original_value = original_values[env_var]
                if original_value is None:
                    os.environ.pop(env_var, None)
                else:
                    os.environ[env_var] = original_value

    def test_directory_creation(self):
        """Test that necessary directories are created."""
        config = ConfigManager()

        # Check that cache directory path exists
        cache_dir = Path(config.config["data"]["cache_dir"])
        assert cache_dir.exists()

        # Check that log directory path exists
        log_file = Path(config.config["logging"]["log_file"])
        assert log_file.parent.exists()

    def test_get_with_invalid_path(self):
        """Test get method with invalid path structures."""
        config = ConfigManager()

        # Test empty path
        assert config.get("") is None

        # Test path that goes through non-dict value
        config.set("test", "simple_value", "not_a_dict")
        assert config.get("test.simple_value.nonexistent") is None

    def test_configuration_persistence(self):
        """Test that configuration changes persist through save/load cycle."""
        # Create initial config
        config1 = ConfigManager()
        config1.set("test", "persist_value", "original")
        config1.set("new_section", "new_key", 999)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            # Save configuration
            config1.save_to_file(temp_file)

            # Load configuration in new instance
            config2 = ConfigManager(config_path=temp_file)

            # Verify values persisted
            assert config2.config["test"]["persist_value"] == "original"
            assert config2.config["new_section"]["new_key"] == 999

            # Verify defaults are still there
            assert config2.config["data"]["default_interval"] == "1d"

        finally:
            os.unlink(temp_file)


class TestIntegration:
    """Integration tests for ConfigManager."""

    def test_complete_workflow(self):
        """Test complete configuration workflow."""
        # Set up environment variables
        os.environ["QUANTPY_API_KEY"] = "test_key_123"
        os.environ["QUANTPY_DATA_PROVIDER"] = "yahoo"

        # Create config file
        config_data = {
            "backtest": {"initial_capital": 100000, "commission": 0.002},
            "strategies": {
                "buy_and_hold": {"enabled": True},
                "mean_reversion": {"enabled": False, "window": 20},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            # Initialize with all sources
            config = ConfigManager(config_path=config_file)

            # Test that all sources were loaded correctly

            # Defaults
            assert config.get("data.default_interval") == "1d"
            assert config.get("logging.level") == "INFO"

            # File config
            assert config.get("backtest.initial_capital") == 100000
            assert config.get("strategies.buy_and_hold.enabled") is True
            assert config.get("strategies.mean_reversion.window") == 20

            # Environment variables
            assert config.get("api.key") == "test_key_123"
            assert config.get("data.provider") == "yahoo"

            # Test precedence (env vars should override file config)
            os.environ["QUANTPY_BACKTEST_INITIAL_CAPITAL"] = "75000"
            config = ConfigManager(config_path=config_file)
            assert config.get("backtest.initial_capital") == 75000

            # Test modification and persistence
            config.set("runtime", "test_run", True)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                output_file = f.name

            config.save_to_file(output_file)

            # Load saved config
            config2 = ConfigManager(config_path=output_file)
            assert config2.get("runtime.test_run") is True

            os.unlink(output_file)

        finally:
            os.unlink(config_file)
            # Clean up environment
            os.environ.pop("QUANTPY_API_KEY", None)
            os.environ.pop("QUANTPY_DATA_PROVIDER", None)
            os.environ.pop("QUANTPY_BACKTEST_INITIAL_CAPITAL", None)

    def test_complex_nested_configuration(self):
        """Test handling of complex nested configurations."""
        complex_config = {
            "data_sources": {
                "primary": {
                    "type": "yahoo",
                    "settings": {
                        "timeout": 30,
                        "retries": 3,
                        "cache": {"enabled": True, "ttl": 3600},
                    },
                },
                "secondary": {
                    "type": "alpha_vantage",
                    "settings": {"api_key": "demo", "premium": False},
                },
            },
            "strategies": {
                "momentum": {
                    "params": {
                        "lookback": 252,
                        "top_n": 10,
                        "rebalance_freq": "monthly",
                    }
                }
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(complex_config, f)
            config_file = f.name

        try:
            config = ConfigManager(config_path=config_file)

            # Test deep nested access
            assert config.get("data_sources.primary.type") == "yahoo"
            assert config.get("data_sources.primary.settings.timeout") == 30
            assert config.get("data_sources.primary.settings.cache.enabled") is True
            assert config.get("data_sources.secondary.settings.premium") is False
            assert config.get("strategies.momentum.params.lookback") == 252

            # Test nonexistent deep paths
            assert config.get("data_sources.primary.settings.cache.nonexistent") is None
            assert (
                config.get("strategies.nonexistent.params.value", "default")
                == "default"
            )

        finally:
            os.unlink(config_file)
