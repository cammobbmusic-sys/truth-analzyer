#!/usr/bin/env python3
"""
Multi-AI System - Truth Analyzer
Entry point for orchestrating verification and brainstorming processes.
"""

import argparse
import sys
from typing import Optional

from agents.orchestrator import Orchestrator
from agents.model_agent import ModelAgent


def create_agents() -> list:
    """Create a list of ModelAgent instances for the orchestrator."""
    return [
        ModelAgent("gemini-3-pro", "expert"),
        ModelAgent("gpt-5.1-codex", "creative"),
        ModelAgent("sonnet-4.5", "analyst")
    ]


def main(query: Optional[str] = None, mode: str = "parallel") -> None:
    """
    Main entry point for the multi-AI system.

    Args:
        query: The question or topic to analyze
        mode: Operation mode - 'parallel', 'sequential', or 'verification'
    """
    try:
        # Create agents
        agents = create_agents()

        # Initialize orchestrator with agents
        orchestrator = Orchestrator(agents)

        # Get query if not provided
        if not query:
            query = input("Enter your question or topic: ").strip()
            if not query:
                print("Error: No query provided.")
                sys.exit(1)

        print(f"Processing query: {query}")
        print(f"Mode: {mode}")
        print(f"Using {len(agents)} agents: {[f'{agent.model_name}({agent.role})' for agent in agents]}")

        # Run the analysis
        results = orchestrator.run_analysis(query, mode)

        # Display results
        print("\n" + "="*50)
        print("ANALYSIS RESULTS")
        print("="*50)

        if mode == "verification" and "consensus" in results:
            from utils.cross_validation import generate_consensus_report

            # Generate detailed consensus report
            report = generate_consensus_report(results['individual_responses'])
            print(report)

            # Show summary
            consensus = results['consensus']
            print(f"\n📊 SUMMARY:")
            print(f"   Query: {results['query']}")
            print(f"   Consensus Achieved: {consensus['consensus_achieved']}")
            print(f"   Agreement Score: {consensus['agreement_score']:.2f}")
            print(f"   Consensus Score: {consensus['consensus_score']:.2f}")
            print(f"   Coherence Score: {consensus['coherence_score']:.2f}")
            print(f"   Confidence Score: {consensus['confidence_score']:.2f}")

        elif mode == "brainstorming" and "ranked_ideas" in results:
            from utils.idea_aggregation import generate_idea_report

            # Generate detailed idea aggregation report
            report = generate_idea_report(results['ranked_ideas'], top_n=3)
            print(report)

            # Show summary
            from utils.metrics import diversity_score
            print(f"\n📊 SUMMARY:")
            print(f"   Query: {results['query']}")
            print(f"   Total Idea Clusters: {len(results['idea_clusters'])}")
            print(f"   Overall Diversity Score: {diversity_score(results['ranked_ideas'])}")
            print(f"   Ranked Ideas: {len(results['ranked_ideas'])}")

        else:
            print(f"Query: {query}")
            print(f"Mode: {mode}")
            print("\nAgent Responses:")
            for role, response in results.items():
                print(f"\n{role.upper()}:")
                print("-" * 20)
                print(response[:300] + "..." if len(response) > 300 else response)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-AI Orchestrator")
    parser.add_argument("query", nargs="?", help="Question or topic to analyze")
    parser.add_argument(
        "--mode",
        choices=["parallel", "sequential", "verification", "brainstorming"],
        default="parallel",
        help="Execution mode (default: parallel)"
    )

    args = parser.parse_args()
    main(query=args.query, mode=args.mode)
