from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import yaml


class ConfigManager:
    """
    Manages configuration settings for the application.
    Supports loading from JSON or YAML files and environment variables.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the config manager.

        Args:
            config_path: Optional path to a configuration file
        """
        self.config: Dict[str, Any] = {}

        # Load default configuration
        self._load_defaults()

        # Load from file if provided
        if config_path:
            self.load_config_file(config_path)

        # Override with environment variables
        self._load_from_env()

    def _load_defaults(self) -> None:
        """Load default configuration values."""
        self.config = {
            "data": {
                "default_interval": "1d",
                "cache_dir": os.path.join(
                    os.path.expanduser("~"), ".quant-py", "cache"
                ),
            },
            "backtest": {
                "default_commission": 0.001,  # 0.1% commission
                "initial_capital": 10000,
            },
            "logging": {
                "level": "INFO",
                "log_file": os.path.join(
                    os.path.expanduser("~"), ".quant-py", "logs", "quant-py.log"
                ),
                "debug_backtest": False,  # Add debug flag for backtest operations
            },
        }

        # Create necessary directories
        os.makedirs(self.config["data"]["cache_dir"], exist_ok=True)
        os.makedirs(os.path.dirname(self.config["logging"]["log_file"]), exist_ok=True)

    def load_config_file(self, config_path: str) -> None:
        """
        Load configuration from a file.

        Args:
            config_path: Path to the configuration file (JSON or YAML)
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Determine file type and load accordingly
        if config_path.endswith(".json"):
            with open(config_path) as f:
                file_config = json.load(f)
        elif config_path.endswith((".yaml", ".yml")):
            with open(config_path) as f:
                file_config = yaml.safe_load(f)
        else:
            raise ValueError(
                "Unsupported configuration file format. Use .json, .yaml, or .yml"
            )

        # Update configuration
        self._update_nested_dict(self.config, file_config)

    def _load_from_env(self) -> None:
        """
        Load configuration from environment variables.

        Environment variables should be in the format:
        QUANTPY_SECTION_KEY=value
        """
        prefix = "QUANTPY_"

        for env_var, value in os.environ.items():
            if env_var.startswith(prefix):
                # Remove prefix and split into parts
                parts = env_var[len(prefix) :].lower().split("_")

                if len(parts) >= 2:
                    section = parts[0]
                    key = "_".join(parts[1:])

                    # Create section if it doesn't exist
                    if section not in self.config:
                        self.config[section] = {}

                    # Convert value to appropriate type if possible
                    if value.lower() in ("true", "yes", "1"):
                        value = True
                    elif value.lower() in ("false", "no", "0"):
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    else:
                        try:
                            value = float(value)
                        except ValueError:
                            pass  # Keep as string

                    self.config[section][key] = value

    def _update_nested_dict(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
        """
        Update a nested dictionary with values from another dictionary.

        Args:
            d: Target dictionary to update
            u: Source dictionary with new values
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation for nested dictionaries.

        Args:
            path: Path to the configuration value using dot notation (e.g., 'data.cache_dir')
            default: Default value to return if the path doesn't exist

        Returns:
            The configuration value or the default
        """
        keys = path.split(".")
        result = self.config

        for key in keys:
            if not isinstance(result, dict) or key not in result:
                return default
            result = result[key]

        return result

    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            section: Configuration section
            key: Configuration key
            value: Value to set
        """
        if section not in self.config:
            self.config[section] = {}

        self.config[section][key] = value

    def save_to_file(self, file_path: str) -> None:
        """
        Save the current configuration to a file.

        Args:
            file_path: Path to save the configuration to
        """
        # Determine file type based on extension
        if file_path.endswith(".json"):
            with open(file_path, "w") as f:
                json.dump(self.config, f, indent=4)
        elif file_path.endswith((".yaml", ".yml")):
            with open(file_path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
        else:
            raise ValueError("Unsupported file format. Use .json, .yaml, or .yml")
