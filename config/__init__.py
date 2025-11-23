"""
YAML-based configuration loader with environment variable overrides.
"""

import os
import yaml


class Config:
    """Singleton configuration class that loads from YAML and supports env var overrides."""

    _instance = None

    def __new__(cls, config_path="config.yaml"):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load(config_path)
        return cls._instance

    def _load(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found.")

        with open(config_path, "r", encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f) or {}

        # Simple environment variable overrides
        # API_KEY_HF -> api_key_hf, REDIS_URL -> redis_url, etc.
        for key, value in os.environ.items():
            if key.startswith(('API_', 'REDIS_', 'LOG_')):
                self._cfg[key.lower()] = value

    def get(self, *keys, default=None):
        """Get nested config value by keys, e.g., get('agents', 0, 'name')"""
        d = self._cfg
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d

    def all(self):
        """Return entire config dict."""
        return self._cfg
