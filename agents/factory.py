

from agents.base import ModelAgent

from agents.adapters.http_generic import HTTPGenericAdapter

from agents.adapters.huggingface import HuggingFaceAdapter



ADAPTER_MAP = {

    "generic": HTTPGenericAdapter,

    "huggingface": HuggingFaceAdapter

}



def create_agent(config: dict) -> ModelAgent:

    '''

    Factory to instantiate agents based on config dict:

    Expected keys: name, provider, model, role, timeout

    '''

    adapter_class = ADAPTER_MAP.get(config.get("provider").lower(), HTTPGenericAdapter)

    return adapter_class(

        name=config.get("name", "unnamed-agent"),

        provider=config.get("provider", "generic"),

        model=config.get("model", ""),

        role=config.get("role", "agent"),

        timeout=config.get("timeout", 15)

    )

