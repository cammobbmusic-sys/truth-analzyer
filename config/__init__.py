"""
YAML-based configuration loader with environment variable overrides.
Also provides backward compatibility with old Config interface.
"""

import os
import yaml
from typing import Dict, List, Any
from dataclasses import dataclass

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, skip automatic loading
    pass


# Backward compatibility classes
@dataclass
class ModelConfig:
    """Configuration for individual AI models."""
    name: str
    model_name: str
    role: str = "general"
    temperature: float = 0.7
    max_tokens: int = 2000


@dataclass
class VerificationConfig:
    """Configuration for verification process."""
    consensus_threshold: float = 0.8
    min_models: int = 3
    max_iterations: int = 5
    conflict_resolution_method: str = "weighted_average"


@dataclass
class BrainstormingConfig:
    """Configuration for brainstorming process."""
    num_ideas_per_iteration: int = 10
    max_iterations: int = 3
    diversity_threshold: float = 0.6
    clustering_method: str = "semantic"


class YAMLConfig:
    """Singleton YAML-based configuration class."""

    _instance = None

    def __new__(cls, config_path="config.yaml"):
        if cls._instance is None:
            cls._instance = super(YAMLConfig, cls).__new__(cls)
            cls._instance._load(config_path)
        return cls._instance

    def _load(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found.")

        with open(config_path, "r", encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f) or {}

        self.apply_env_overrides(self._cfg)

    @staticmethod
    def apply_env_overrides(config_dict):
        """Apply environment variable overrides to config dict."""
        for key, value in os.environ.items():
            if key.startswith(('API_', 'REDIS_', 'LOG_')):
                config_dict[key.lower()] = value

    def get(self, *keys, default=None) -> Any:
        """Get nested config value by keys."""
        d = self._cfg
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d

    def all(self) -> Dict[str, Any]:
        """Return entire config dict."""
        return self._cfg


class Config:
    """Backward-compatible Config class that mimics the old interface."""

    def __init__(self):
        # Load YAML config directly (not using singleton to avoid initialization issues)
        yaml_config = self._load_yaml_config()

        # Create old-style objects for compatibility
        self.models = self._load_model_configs(yaml_config)
        self.verification = VerificationConfig()
        self.brainstorming = BrainstormingConfig()
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.output_format = os.getenv("OUTPUT_FORMAT", "json")
        self.prompts = self._load_prompt_templates()

        # Override with YAML values if available
        yaml_verification = yaml_config.get('verification')
        if yaml_verification:
            for key, value in yaml_verification.items():
                if hasattr(self.verification, key):
                    setattr(self.verification, key, value)

        yaml_brainstorming = yaml_config.get('brainstorming')
        if yaml_brainstorming:
            for key, value in yaml_brainstorming.items():
                if hasattr(self.brainstorming, key):
                    setattr(self.brainstorming, key, value)

    def _load_yaml_config(self):
        """Load YAML config directly."""
        config_path = "config.yaml"
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            YAMLConfig.apply_env_overrides(cfg)
            return cfg
        return {}

    def _load_model_configs(self, yaml_config) -> List[Dict[str, Any]]:
        """Load AI model configurations from YAML as dictionaries for factory compatibility."""
        agents = yaml_config.get('agents', [])
        models = []
        for agent in agents:
            if isinstance(agent, dict):
                models.append({
                    'name': agent.get('name', agent.get('model', 'unnamed')),
                    'model': agent.get('model', ''),
                    'provider': agent.get('provider', 'generic'),
                    'role': agent.get('role', 'general'),
                    'timeout': agent.get('timeout', 15),
                })
        return models

    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates (placeholder for compatibility)."""
        return {
            "verification_base": "Analyze the following question: {question}",
            "brainstorming_base": "Generate ideas for: {topic}",
            "meta_prompt_refinement": "Refine: {original_prompt}"
        }

    def get_active_models(self) -> List[ModelConfig]:
        """Get all configured models with safe attribute access."""
        models = getattr(self, "models", [])
        return models

    def get_models_safe(self) -> List[ModelConfig]:
        """Safely get models list with defensive programming pattern."""
        models = getattr(self, "models", [])
        return models

    def validate_config(self) -> List[str]:
        """Validate configuration and return any issues with safe attribute access."""
        issues = []
        models = self.get_models_safe()
        verification = getattr(self, "verification", None)

        if len(models) < getattr(verification, "min_models", 3):
            issues.append(f"Insufficient models: {len(models)} < {getattr(verification, 'min_models', 3)}")

        if verification and hasattr(verification, "consensus_threshold"):
            threshold = getattr(verification, "consensus_threshold", 0.8)
            if not 0.5 <= threshold <= 1.0:
                issues.append("Consensus threshold must be between 0.5 and 1.0")
        return issues


# Export both old and new interfaces
__all__ = ['Config', 'YAMLConfig', 'ModelConfig', 'VerificationConfig', 'BrainstormingConfig']
