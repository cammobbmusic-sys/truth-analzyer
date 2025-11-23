#!/usr/bin/env python3
"""
Complete workflow example showing how to use the multi-AI system.
This demonstrates the full pipeline from prompt generation to analysis.
"""

from agents.model_agent import ModelAgent
from agents.meta_prompt import MetaPrompt
from agents.orchestrator import Orchestrator
from utils.cross_validation import cross_validate
from utils.idea_aggregation import aggregate_ideas


def main():
    print("🚀 Multi-AI System Complete Workflow Demo")
    print("=" * 50)

    # Step 1: Initialize agents with different roles
    print("Step 1: Initializing agents...")
    agents = [
        ModelAgent("gemini-3-pro", role="expert"),      # For verification/analysis
        ModelAgent("gpt-5.1-codex", role="creative"),   # For brainstorming
        ModelAgent("sonnet-4.5", role="analyst")        # For balanced analysis
    ]

    orchestrator = Orchestrator(agents)
    meta_prompt = MetaPrompt()

    print("✓ Agents initialized:")
    for agent in agents:
        provider = "Google" if "gemini" in agent.model_name else "OpenAI" if "gpt" in agent.model_name else "Anthropic"
        print(f"  • {agent.model_name} ({provider}) - {agent.role}")
    print()

    # Step 2: Generate prompt using MetaPrompt
    print("Step 2: Generating prompt...")
    context = "Recent news report on AI model hallucinations."
    prompt = meta_prompt.generate_prompt(
        task="verification",
        context=context,
        model_role="expert"
    )

    print(f"Context: {context}")
    print(f"Generated Prompt: {prompt[:100]}...")
    print()

    # Step 3: Run agents in parallel
    print("Step 3: Running agents in parallel...")
    outputs = orchestrator.run_parallel(prompt)

    print("Agent Outputs:")
    for role, response in outputs.items():
        print(f"\n{role.upper()}:")
        print(f"  {response[:150]}...")
    print()

    # Step 4: Cross-validate outputs
    print("Step 4: Cross-validating outputs...")
    verified, flagged = cross_validate(outputs, threshold=0.7)

    print(f"✓ Verified statements: {len(verified)}")
    if verified:
        for i, statement in enumerate(verified[:3], 1):  # Show up to 3
            print(f"  {i}. {statement[:80]}...")

    print(f"⚠️ Flagged discrepancies: {len(flagged)}")
    if flagged:
        for role1, role2, text1, text2 in flagged[:2]:  # Show up to 2
            print(f"  • {role1} vs {role2}: Low similarity")
    print()

    # Step 5: Aggregate ideas for brainstorming
    print("Step 5: Aggregating ideas for brainstorming...")
    ideas = aggregate_ideas(outputs)

    print(f"✓ Generated {len(ideas)} idea clusters")
    for i, (cluster_key, cluster_ideas) in enumerate(ideas[:3], 1):  # Show top 3 clusters
        print(f"  Cluster {i}: {len(cluster_ideas)} ideas")
        print(f"    Key: {cluster_key[:60]}...")
    print()

    # Alternative: Use the orchestrator's high-level methods
    print("🎯 Alternative: Using Orchestrator High-Level Methods")
    print("-" * 50)

    # Verification mode
    print("Verification Mode:")
    verification_results = orchestrator.run_analysis(context, mode="verification")
    consensus = verification_results['consensus']
    print(f"  Consensus Score: {consensus['consensus_score']:.2f}")
    print(f"  Agreement Score: {consensus['agreement_score']:.2f}")
    print(f"  Coherence Score: {consensus['coherence_score']:.2f}")
    print(f"  Consensus Achieved: {consensus['consensus_achieved']}")
    print()

    # Brainstorming mode
    print("Brainstorming Mode:")
    brainstorming_results = orchestrator.run_analysis("Smart city solutions", mode="brainstorming")
    print(f"  Idea Clusters: {len(brainstorming_results['idea_clusters'])}")
    print(f"  Ranked Ideas: {len(brainstorming_results['ranked_ideas'])}")
    if brainstorming_results['ranked_ideas']:
        top_idea = brainstorming_results['ranked_ideas'][0]
        print(f"  Top Idea Quality: {top_idea['quality_score']:.2f}")
        print(f"  Top Idea Novelty: {top_idea['novelty_score']:.2f}")
    print()

    print("✅ Complete workflow demonstration finished!")


if __name__ == "__main__":
    main()
