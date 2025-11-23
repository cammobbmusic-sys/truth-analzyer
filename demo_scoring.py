#!/usr/bin/env python3
"""
Demo script showing the new scoring functions.
"""

from utils.metrics import consensus_score, diversity_score, coherence_score
from utils.cross_validation import analyze_consensus
from utils.idea_aggregation import aggregate_ideas, rank_ideas_by_novelty


def demo_scoring_functions():
    """Demo the individual scoring functions."""
    print("📊 Scoring Functions Demo")
    print("=" * 40)

    # Test consensus_score
    print("1. Consensus Score:")
    print(f"   3 verified out of 5 total: {consensus_score(3, 5):.2f}")
    print(f"   0 verified out of 3 total: {consensus_score(0, 3):.2f}")
    print(f"   5 verified out of 5 total: {consensus_score(5, 5):.2f}")
    print()

    # Test diversity_score
    print("2. Diversity Score:")
    clusters1 = ["cluster1", "cluster2", "cluster3"]
    clusters2 = ["cluster1"]
    print(f"   3 clusters: {diversity_score(clusters1)}")
    print(f"   1 cluster: {diversity_score(clusters2)}")
    print()

    # Test coherence_score
    print("3. Coherence Score:")
    coherent_outputs = {
        "agent1": "Machine learning is a powerful technology",
        "agent2": "ML represents a significant technological advancement",
        "agent3": "Machine learning technology is highly capable"
    }

    incoherent_outputs = {
        "agent1": "Cats are mammals",
        "agent2": "The moon is made of cheese",
        "agent3": "Water boils at 100 degrees Celsius"
    }

    coherent_score = coherence_score(coherent_outputs)
    incoherent_score = coherence_score(incoherent_outputs)

    print(f"   Coherent outputs: {coherent_score:.2f}")
    print(f"   Incoherent outputs: {incoherent_score:.2f}")
    print()


def demo_integrated_scoring():
    """Demo how scoring functions are integrated into the analysis pipeline."""
    print("🔗 Integrated Scoring Demo")
    print("=" * 40)

    # Sample agent outputs for verification
    verification_outputs = {
        "expert": "Based on available evidence, verify and analyze this can be evaluated as scientifically accurate. Key supporting factors include peer-reviewed research.",
        "creative": "From multiple perspectives, verify and analyze this can be evaluated as practically viable. Critical considerations include practical applications.",
        "analyst": "Based on available evidence, verify and analyze this can be evaluated as well-supported. Key supporting factors include theoretical foundations."
    }

    print("Verification Analysis with Enhanced Scoring:")
    analysis = analyze_consensus(verification_outputs)

    print(f"   Agreement Score: {analysis['agreement_score']:.2f}")
    print(f"   Consensus Score: {analysis['consensus_score']:.2f}")
    print(f"   Coherence Score: {analysis['coherence_score']:.2f}")
    print(f"   Confidence Score: {analysis['confidence_score']:.2f}")
    print(f"   Consensus Achieved: {analysis['consensus_achieved']}")
    print()

    # Sample outputs for idea aggregation
    brainstorming_outputs = {
        "expert": """1. Implement comprehensive cybersecurity protocols
2. Develop automated testing frameworks
3. Establish continuous integration pipelines""",

        "creative": """• Create immersive virtual reality training environments
• Deploy AI-powered code review assistants
• Build collaborative real-time editing platforms""",

        "analyst": """- Optimize database query performance
- Implement predictive analytics dashboards
- Develop automated reporting systems"""
    }

    print("Brainstorming Analysis with Diversity Scoring:")
    clusters = aggregate_ideas(brainstorming_outputs)
    ranked_ideas = rank_ideas_by_novelty(dict(clusters))

    print(f"   Total Idea Clusters: {len(clusters)}")
    print(f"   Overall Diversity Score: {diversity_score(ranked_ideas)}")

    if ranked_ideas:
        top_cluster = ranked_ideas[0]
        print(f"   Top Cluster Quality: {top_cluster['quality_score']:.2f}")
        print(f"   Top Cluster Novelty: {top_cluster['novelty_score']:.2f}")
        print(f"   Top Cluster Diversity: {top_cluster['diversity_score']:.2f}")

    print()


def demo_scoring_comparison():
    """Compare scoring across different scenarios."""
    print("⚖️ Scoring Comparison Demo")
    print("=" * 40)

    scenarios = [
        {
            "name": "High Agreement",
            "outputs": {
                "agent1": "Solar power is renewable energy",
                "agent2": "Solar energy represents a renewable source",
                "agent3": "Solar power is a form of renewable energy"
            }
        },
        {
            "name": "Mixed Agreement",
            "outputs": {
                "agent1": "Electric cars reduce emissions",
                "agent2": "EVs help the environment",
                "agent3": "Nuclear power is controversial"
            }
        },
        {
            "name": "Low Agreement",
            "outputs": {
                "agent1": "Cats are pets",
                "agent2": "Dogs are loyal",
                "agent3": "Fish live in water"
            }
        }
    ]

    for scenario in scenarios:
        print(f"{scenario['name']}:")
        coherence = coherence_score(scenario['outputs'])
        analysis = analyze_consensus(scenario['outputs'])
        print(f"   Coherence: {coherence:.2f}")
        print(f"   Confidence: {analysis['confidence_score']:.2f}")
        print()


if __name__ == "__main__":
    demo_scoring_functions()
    demo_integrated_scoring()
    demo_scoring_comparison()
