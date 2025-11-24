

from typing import Dict, Any
from agents.base import ModelAgent
from agents.adapters.http_generic import HTTPGenericAdapter
from agents.adapters.huggingface import HuggingFaceAdapter
from agents.adapters.groq_adapter import GroqAdapter
from agents.adapters.openrouter_adapter import OpenRouterAdapter
from agents.adapters.together_adapter import TogetherAdapter
from agents.adapters.cohere_adapter import CohereAdapter


ADAPTER_MAP = {

    "generic": HTTPGenericAdapter,

    "huggingface": HuggingFaceAdapter,

    "groq": GroqAdapter,

    "openrouter": OpenRouterAdapter,  # PRIORITY: Use OpenRouter over Together

    "together": TogetherAdapter,      # DEPRECATED: Focus on OpenRouter instead

    "cohere": CohereAdapter           # NEW: Cohere AI for diverse responses

}


def normalize_agent_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and validate agent configuration.
    Sets defaults and validates required fields.
    """
    if not isinstance(config, dict):
        raise ValueError("Agent config must be a dictionary")

    # Set defaults
    defaults = {
        'role': 'agent',
        'timeout': 15,
        'model': '',
        'name': 'unnamed-agent',
        'provider': 'generic'
    }

    for key, default in defaults.items():
        config.setdefault(key, default)

    # Validate required fields
    required = ['name', 'provider', 'model']
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"Missing required agent config fields: {missing}")

    return config


def create_agent(config: Dict[str, Any]) -> ModelAgent:

    '''

    Factory to instantiate agents based on config dict:

    Expected keys: name, provider, model, role, timeout

    '''

    # Normalize and validate config
    normalized_config = normalize_agent_config(config)

    adapter_class = ADAPTER_MAP.get(normalized_config["provider"].lower(), HTTPGenericAdapter)

    return adapter_class(

        name=normalized_config["name"],

        provider=normalized_config["provider"],

        model=normalized_config["model"],

        role=normalized_config["role"],

        timeout=normalized_config["timeout"]

    )

