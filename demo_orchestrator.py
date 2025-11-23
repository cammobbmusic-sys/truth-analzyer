#!/usr/bin/env python3
"""
Demo script showing the simplified Orchestrator functionality.
"""

from agents.orchestrator import Orchestrator
from agents.model_agent import ModelAgent


def main():
    print("🔧 Orchestrator Demo")
    print("=" * 50)

    # Create agents
    agents = [
        ModelAgent("gemini-3-pro", "expert"),
        ModelAgent("gpt-5.1-codex", "creative"),
        ModelAgent("sonnet-4.5", "analyst")
    ]

    # Initialize orchestrator
    orchestrator = Orchestrator(agents)

    print(f"Created orchestrator with {len(agents)} agents:")
    for agent in agents:
        print(f"  • {agent.model_name} ({agent.role})")
    print()

    # Demo 1: Parallel execution
    print("🚀 Demo 1: Parallel Execution")
    print("-" * 30)
    query = "What are the benefits of renewable energy?"
    print(f"Query: {query}")

    parallel_results = orchestrator.run_parallel(f"Please analyze: {query}")
    print("\nResults:")
    for role, response in parallel_results.items():
        print(f"\n{role.upper()}:")
        print(response[:150] + "..." if len(response) > 150 else response)
    print()

    # Demo 2: Sequential execution
    print("🔄 Demo 2: Sequential Execution")
    print("-" * 30)
    prompts = [
        "Initial analysis of artificial intelligence",
        "Build upon the previous analysis with ethical considerations",
        "Provide final synthesis and future outlook"
    ]
    print("Prompts:")
    for i, prompt in enumerate(prompts, 1):
        print(f"  {i}. {prompt}")

    sequential_results = orchestrator.run_sequential(prompts)
    print("\nResults:")
    for role, response in sequential_results.items():
        print(f"\n{role.upper()}:")
        print(response[:200] + "..." if len(response) > 200 else response)
    print()

    # Demo 3: Verification mode
    print("✅ Demo 3: Verification Mode")
    print("-" * 30)
    verification_query = "Machine learning is a subset of artificial intelligence"
    print(f"Claim to verify: {verification_query}")

    verification_results = orchestrator.run_analysis(verification_query, mode="verification")
    print("\nConsensus Analysis:")
    print(f"  Agreement Score: {verification_results['consensus']['agreement_score']:.2f}")
    print(f"  Total Agents: {verification_results['consensus']['total_agents']}")
    print(f"  Avg Response Length: {verification_results['consensus']['average_response_length']:.0f} chars")

    print("\nIndividual Verifications:")
    for role, response in verification_results['individual_responses'].items():
        print(f"\n{role.upper()}:")
        print(response[:150] + "..." if len(response) > 150 else response)


if __name__ == "__main__":
    main()
