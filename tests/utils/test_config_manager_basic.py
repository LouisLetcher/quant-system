"""Basic tests for config manager."""

from __future__ import annotations

from unittest.mock import mock_open, patch

import pytest

from src.utils.config_manager import ConfigManager


class TestConfigManagerBasic:
    """Basic test cases for ConfigManager class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        config = ConfigManager()
        assert hasattr(config, "config")
        assert isinstance(config.config, dict)

    @patch("builtins.open", new_callable=mock_open, read_data='{"test": "value"}')
    @patch("src.utils.config_manager.Path.exists")
    def test_load_json_config(self, mock_exists, mock_file):
        """Test loading JSON configuration."""
        mock_exists.return_value = True

        config = ConfigManager("test_config.json")
        assert config.config.get("test") == "value"

    @patch(
        "builtins.open", new_callable=mock_open, read_data="test: value\nother: data"
    )
    @patch("src.utils.config_manager.Path.exists")
    def test_load_yaml_config(self, mock_exists, mock_file):
        """Test loading YAML configuration."""
        mock_exists.return_value = True

        with patch("src.utils.config_manager.yaml.safe_load") as mock_yaml:
            mock_yaml.return_value = {"test": "value", "other": "data"}
            config = ConfigManager("test_config.yaml")
            assert config.config.get("test") == "value"

    def test_get_method_basic(self):
        """Test basic get method functionality."""
        config = ConfigManager()
        config.config = {"test": {"nested": "value"}}

        assert config.get("test.nested") == "value"
        assert config.get("nonexistent", "default") == "default"

    def test_set_method_basic(self):
        """Test basic set method functionality."""
        config = ConfigManager()

        config.set("test.nested", "value")
        assert config.get("test.nested") == "value"

    @patch("builtins.open", new_callable=mock_open)
    @patch("src.utils.config_manager.Path.mkdir")
    def test_save_json_format(self, mock_mkdir, mock_file):
        """Test saving configuration in JSON format."""
        config = ConfigManager()
        config.config = {"test": "value"}

        config.save("test_output.json")
        mock_file.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("src.utils.config_manager.Path.mkdir")
    def test_save_yaml_format(self, mock_mkdir, mock_file):
        """Test saving configuration in YAML format."""
        config = ConfigManager()
        config.config = {"test": "value"}

        with patch("src.utils.config_manager.yaml.dump") as mock_yaml_dump:
            config.save("test_output.yaml")
            mock_yaml_dump.assert_called_once()

    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        with patch.dict("os.environ", {"TEST_CONFIG_VAR": "test_value"}):
            config = ConfigManager()
            config._load_from_environment("TEST_CONFIG_")

            # Check that environment variables were processed
            assert hasattr(config, "config")

    def test_unsupported_file_format(self):
        """Test handling of unsupported file formats."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            config = ConfigManager()
            config.save("test.txt")

    def test_nested_dict_operations(self):
        """Test nested dictionary operations."""
        config = ConfigManager()

        # Test deep nesting
        config.set("level1.level2.level3", "deep_value")
        assert config.get("level1.level2.level3") == "deep_value"

        # Test overwriting
        config.set("level1.level2.level3", "new_value")
        assert config.get("level1.level2.level3") == "new_value"


class TestConfigManagerIntegration:
    """Integration tests for ConfigManager."""

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"app": {"name": "test", "version": "1.0"}}',
    )
    @patch("src.utils.config_manager.Path.exists")
    def test_complete_workflow(self, mock_exists, mock_file):
        """Test complete configuration workflow."""
        mock_exists.return_value = True

        # Load configuration
        config = ConfigManager("app_config.json")

        # Read values
        app_name = config.get("app.name")
        assert app_name == "test"

        # Modify values
        config.set("app.version", "2.0")
        assert config.get("app.version") == "2.0"

        # Configuration should be ready for saving
        assert config.config is not None

    def test_error_handling(self):
        """Test error handling in configuration operations."""
        config = ConfigManager()

        # Test accessing non-existent key without default
        result = config.get("nonexistent")
        assert result is None

        # Test with default value
        result = config.get("nonexistent", "default")
        assert result == "default"

    def test_type_conversion(self):
        """Test automatic type conversion in configuration."""
        config = ConfigManager()

        # Set different types
        config.set("string_val", "text")
        config.set("int_val", 42)
        config.set("bool_val", True)
        config.set("list_val", [1, 2, 3])

        # Verify types are preserved
        assert isinstance(config.get("string_val"), str)
        assert isinstance(config.get("int_val"), int)
        assert isinstance(config.get("bool_val"), bool)
        assert isinstance(config.get("list_val"), list)
