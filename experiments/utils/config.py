"""Configuration utilities for experiments.

Supports:
- Loading YAML configs
- Merging multiple configs (base + overrides)
- CLI argument overrides
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import copy


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to YAML file

    Returns:
        Configuration dictionary
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config or {}


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries.

    Values in override take precedence. Nested dicts are merged recursively.

    Args:
        base: Base dictionary
        override: Override dictionary

    Returns:
        Merged dictionary
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def merge_configs(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple configs in order (later configs override earlier).

    Args:
        configs: List of config dictionaries

    Returns:
        Merged configuration
    """
    result = {}
    for config in configs:
        result = deep_merge(result, config)
    return result


def parse_cli_overrides(args: List[str]) -> Dict[str, Any]:
    """Parse CLI override arguments.

    Supports nested keys with dot notation:
        --model.hidden_dim=512
        --training.learning_rate=0.001
        --data.n_samples=10000

    Args:
        args: List of CLI arguments

    Returns:
        Dictionary of overrides
    """
    overrides = {}

    for arg in args:
        if not arg.startswith("--"):
            continue

        arg = arg[2:]  # Remove --

        if "=" not in arg:
            # Boolean flag
            key = arg
            value = True
        else:
            key, value = arg.split("=", 1)
            value = _parse_value(value)

        # Handle nested keys
        _set_nested(overrides, key.split("."), value)

    return overrides


def _parse_value(value: str) -> Any:
    """Parse a string value to its appropriate type."""
    # None
    if value.lower() in ("null", "none"):
        return None

    # Boolean
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    # Number
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        pass

    # List (simple comma-separated)
    if value.startswith("[") and value.endswith("]"):
        items = value[1:-1].split(",")
        return [_parse_value(item.strip()) for item in items if item.strip()]

    # String
    return value


def _set_nested(d: Dict, keys: List[str], value: Any):
    """Set a nested dictionary value.

    Args:
        d: Dictionary to modify
        keys: List of nested keys
        value: Value to set
    """
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def get_nested(d: Dict, keys: List[str], default: Any = None) -> Any:
    """Get a nested dictionary value.

    Args:
        d: Dictionary to read from
        keys: List of nested keys
        default: Default value if not found

    Returns:
        Value at nested key path or default
    """
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


def config_to_flat_dict(config: Dict, prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested config to dot-notation keys.

    Args:
        config: Nested configuration dictionary
        prefix: Key prefix

    Returns:
        Flat dictionary with dot-notation keys
    """
    result = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(config_to_flat_dict(value, full_key))
        else:
            result[full_key] = value
    return result


class Config:
    """Configuration container with attribute access."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary.

        Args:
            config_dict: Configuration dictionary
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default."""
        return getattr(self, key, default)

    def __repr__(self):
        return f"Config({self.to_dict()})"
