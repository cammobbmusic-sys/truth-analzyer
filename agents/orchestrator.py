"""
Orchestrator - Multi-agent orchestration for parallel/sequential execution.
"""

from concurrent.futures import ThreadPoolExecutor
from .model_agent import ModelAgent
from utils.cross_validation import analyze_consensus
from utils.idea_aggregation import aggregate_ideas, rank_ideas_by_novelty, generate_idea_report


class Orchestrator:
    """Orchestrates multiple AI agents in parallel or sequential execution."""

    def __init__(self, agents: list):
        """
        Initialize orchestrator with a list of agents.

        Args:
            agents: List of ModelAgent instances
        """
        self.agents = agents

    def run_parallel(self, prompt: str) -> dict:
        """
        Runs multiple agents in parallel and returns their outputs.

        Args:
            prompt: The prompt to send to all agents

        Returns:
            Dictionary mapping agent roles to their outputs
        """
        outputs = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(agent.run_prompt, prompt): agent for agent in self.agents}
            for future in futures:
                agent = futures[future]
                outputs[agent.role] = future.result()
        return outputs

    def run_sequential(self, prompts: list) -> dict:
        """
        Runs agents sequentially, passing outputs from one as context to the next.

        Args:
            prompts: List of prompts, one for each agent

        Returns:
            Dictionary mapping agent roles to their outputs
        """
        context = ""
        results = {}
        for agent, prompt in zip(self.agents, prompts):
            full_prompt = f"{prompt}\nPrevious context:\n{context}"
            output = agent.run_prompt(full_prompt)
            results[agent.role] = output
            context += "\n" + output
        return results

    def run_analysis(self, query: str, mode: str = "parallel") -> dict:
        """
        High-level analysis method supporting different execution modes.

        Args:
            query: The question or topic to analyze
            mode: Execution mode - 'parallel', 'sequential', or 'verification'

        Returns:
            Analysis results
        """
        if mode == "parallel":
            # Run all agents in parallel with the same prompt
            return self.run_parallel(f"Please analyze: {query}")

        elif mode == "sequential":
            # Run agents sequentially with different prompts
            prompts = [
                f"Initial analysis of: {query}",
                f"Build upon the previous analysis for: {query}",
                f"Provide final synthesis for: {query}"
            ]
            return self.run_sequential(prompts[:len(self.agents)])

        elif mode == "verification":
            # Cross-verification approach
            verification_prompt = f"Verify and analyze this claim/question: {query}\nProvide evidence and reasoning."
            parallel_results = self.run_parallel(verification_prompt)

            # Add consensus analysis
            return {
                "individual_responses": parallel_results,
                "consensus": self._analyze_consensus(parallel_results),
                "query": query,
                "mode": mode
            }

        elif mode == "brainstorming":
            # Idea generation and aggregation
            brainstorming_prompt = f"Generate creative ideas for: {query}\nProvide diverse perspectives and approaches."
            parallel_results = self.run_parallel(brainstorming_prompt)

            # Aggregate and cluster ideas
            idea_clusters = aggregate_ideas(parallel_results)
            ranked_ideas = rank_ideas_by_novelty(dict(idea_clusters))

            return {
                "individual_responses": parallel_results,
                "idea_clusters": idea_clusters,
                "ranked_ideas": ranked_ideas,
                "query": query,
                "mode": mode
            }

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _analyze_consensus(self, responses: dict) -> dict:
        """Comprehensive consensus analysis using cross-validation."""
        return analyze_consensus(responses)
