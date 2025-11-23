"""
Configuration settings for the Multi-AI System using Cursor 2.0 AI integration.
Contains model configurations, prompt templates, and thresholds.
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for individual Cursor AI models."""
    name: str
    model_name: str  # Cursor AI model identifier
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


class Config:
    """Main configuration class."""

    def __init__(self):
        # Model configurations for Cursor AI
        self.models = self._load_model_configs()

        # Verification settings
        self.verification = VerificationConfig()

        # Brainstorming settings
        self.brainstorming = BrainstormingConfig()

        # General settings
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.output_format = os.getenv("OUTPUT_FORMAT", "json")

        # Prompt templates
        self.prompts = self._load_prompt_templates()

    def _load_model_configs(self) -> List[ModelConfig]:
        """Load AI model configurations."""
        return [
            ModelConfig(
                name="gemini-3-pro",
                model_name="gemini-3-pro",
                role="expert",
                temperature=0.3,
                max_tokens=3000
            ),
            ModelConfig(
                name="gpt-5.1-codex",
                model_name="gpt-5.1-codex",
                role="creative",
                temperature=0.7,
                max_tokens=3000
            ),
            ModelConfig(
                name="sonnet-4.5",
                model_name="sonnet-4.5",
                role="analyst",
                temperature=0.5,
                max_tokens=3000
            ),
        ]

    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates."""
        templates = {}

        # Default templates
        templates["verification_base"] = """
Analyze the following question and provide a well-reasoned answer.
Question: {question}

Provide your response with:
1. Direct answer
2. Explanation with evidence
3. Confidence level (0-1)
4. Key assumptions
"""

        templates["brainstorming_base"] = """
Generate creative ideas for the following topic. Focus on diverse perspectives and innovative approaches.
Topic: {topic}

Generate {num_ideas} distinct ideas, each with:
1. Idea title
2. Brief description
3. Potential benefits
4. Implementation considerations
"""

        templates["meta_prompt_refinement"] = """
Review and refine the following prompt to improve clarity and effectiveness:
Original prompt: {original_prompt}

Provide:
1. Refined version
2. Key improvements made
3. Rationale for changes
"""

        return templates

    def get_active_models(self) -> List[ModelConfig]:
        """Get all configured Cursor AI models (no API key validation needed)."""
        return self.models

    def validate_config(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []

        # Check for at least one model
        if len(self.models) < self.verification.min_models:
            issues.append(
                f"Insufficient models configured: {len(self.models)} "
                f"(minimum required: {self.verification.min_models})"
            )

        # Check verification threshold
        if not 0.5 <= self.verification.consensus_threshold <= 1.0:
            issues.append("Consensus threshold must be between 0.5 and 1.0")

        return issues
