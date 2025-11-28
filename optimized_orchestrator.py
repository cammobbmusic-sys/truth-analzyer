"""
Optimized Orchestrator with Caching Layer

This module provides optimized versions of the core pipelines that use
semantic similarity caching to reduce API costs and improve response times.
"""

from typing import Dict, Any, Optional
from optimization_layer import optimized_pipeline
from orchestrator.pipelines.debate_orchestrator import DebateOrchestrator
import json


class OptimizedDebateOrchestrator:
    """Enhanced DebateOrchestrator with caching capabilities."""

    def __init__(self, max_rounds: int = 3, delay_between_agents: float = 1.0,
                 absolute_truth_mode: bool = False, use_cache: bool = True):
        self.orchestrator = DebateOrchestrator(
            max_rounds=max_rounds,
            delay_between_agents=delay_between_agents,
            absolute_truth_mode=absolute_truth_mode
        )
        self.use_cache = use_cache

    def _debate_pipeline_adapter(self, topic: str) -> Dict[str, Any]:
        """
        Adapter function that makes the debate orchestrator compatible
        with the optimization layer interface.

        Args:
            topic: Debate topic string

        Returns:
            Dict containing debate results
        """
        # For caching, we need a simple signature that takes just a query string
        # and returns a dict. We'll create a wrapper that uses default agents.
        agent_configs = [
            {"name": "Agent1", "provider": "groq", "model": "llama-3.1-8b-instant"},
            {"name": "Agent2", "provider": "openrouter", "model": "anthropic/claude-3-haiku"},
            {"name": "Agent3", "provider": "cohere", "model": "command-nightly"}
        ]

        result = self.orchestrator.run(
            topic=topic,
            agent_configs=agent_configs,
            dry_run=False,
            absolute_truth_mode=self.orchestrator.absolute_truth_mode
        )

        # Convert the result to a cacheable format
        return {
            "topic": topic,
            "transcript": result.get("transcript", []),
            "final_summary": result.get("final_summary", {}),
            "total_rounds": len(result.get("transcript", [])),
            "total_messages": sum(len(round.get("messages", [])) for round in result.get("transcript", [])),
            "absolute_truth_mode": self.orchestrator.absolute_truth_mode,
            "cached": False
        }

    def run_optimized_debate(self, topic: str) -> Dict[str, Any]:
        """
        Run debate with caching optimization.

        Args:
            topic: Debate topic

        Returns:
            Dict with optimization results and debate data
        """
        if not self.use_cache:
            # Run without caching
            result = self._debate_pipeline_adapter(topic)
            return {
                "source": "PIPELINE",
                "result": result
            }

        # Use optimized pipeline with caching
        return optimized_pipeline(topic, self._debate_pipeline_adapter)

    def run(self, *args, **kwargs):
        """Run the optimized debate orchestrator."""
        pass

    def execute(self, *args, **kwargs):
        """Execute the optimized debate orchestrator."""
        pass

    def coordinate(self, *args, **kwargs):
        """Coordinate the optimized debate orchestrator."""
        pass


def create_optimized_debate_orchestrator(max_rounds: int = 3,
                                       absolute_truth_mode: bool = False,
                                       use_cache: bool = True) -> OptimizedDebateOrchestrator:
    """
    Factory function to create an optimized debate orchestrator.

    Args:
        max_rounds: Number of debate rounds
        absolute_truth_mode: Whether to use absolute truth mode
        use_cache: Whether to use semantic caching

    Returns:
        OptimizedDebateOrchestrator instance
    """
    return OptimizedDebateOrchestrator(
        max_rounds=max_rounds,
        absolute_truth_mode=absolute_truth_mode,
        use_cache=use_cache
    )


# Example usage functions
def run_optimized_debate_example():
    """Example of running an optimized debate."""
    # Create optimized orchestrator
    optimized_orchestrator = create_optimized_debate_orchestrator(
        max_rounds=2,
        absolute_truth_mode=False,
        use_cache=True
    )

    # Run debate
    topic = "Should social media platforms be regulated by governments?"
    result = optimized_orchestrator.run_optimized_debate(topic)

    print(f"Source: {result['source']}")  # CACHE or PIPELINE
    print(f"Topic: {result['result']['topic']}")
    print(f"Total Rounds: {result['result']['total_rounds']}")
    print(f"Total Messages: {result['result']['total_messages']}")

    return result


if __name__ == "__main__":
    # Run example
    result = run_optimized_debate_example()
    print("\nDebate completed!")
    print(f"Result: {json.dumps(result, indent=2)[:500]}...")
