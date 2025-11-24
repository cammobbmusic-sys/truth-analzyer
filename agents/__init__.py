"""
Agents module for the Multi-AI System.
Contains AI model wrappers, factory, and meta-prompt logic.
Orchestrator functionality moved to orchestrator module.
"""

from .model_agent import ModelAgent
from .meta_prompt import MetaPrompt
from .factory import create_agent, normalize_agent_config

__all__ = ["ModelAgent", "MetaPrompt", "create_agent", "normalize_agent_config"]
