

from agents.base import ModelAgent

from agents.adapters.http_generic import HTTPGenericAdapter

from agents.adapters.huggingface import HuggingFaceAdapter



ADAPTER_MAP = {

    "generic": HTTPGenericAdapter,

    "huggingface": HuggingFaceAdapter

}


def normalize_agent_config(config: dict) -> dict:
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


def create_agent(config: dict) -> ModelAgent:

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

