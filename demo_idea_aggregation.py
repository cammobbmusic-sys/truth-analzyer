#!/usr/bin/env python3
"""
Demo script showing the idea aggregation functionality.
"""

from utils.idea_aggregation import (
    aggregate_ideas,
    rank_ideas_by_novelty,
    generate_idea_report,
    find_contradictory_ideas
)


def demo_basic_aggregation():
    """Demo basic idea aggregation functionality."""
    print("🧠 Basic Idea Aggregation Demo")
    print("=" * 40)

    # Sample outputs from different agents brainstorming transportation solutions
    outputs = {
        "expert": """1. Implement autonomous vehicle infrastructure with dedicated lanes
2. Develop integrated public transit networks with real-time scheduling
3. Invest in high-speed rail connections between major cities
4. Create pedestrian-friendly urban planning with bike lanes""",

        "creative": """• Transform cities with hyperloop networks for instant travel
• Deploy drone taxis for personal air transportation
• Build underwater tunnels for coastal city connections
• Create floating cities to reduce urban congestion""",

        "analyst": """- Establish comprehensive data analytics for traffic optimization
- Implement congestion pricing to manage peak-hour travel
- Develop multimodal transportation hubs at key locations
- Create AI-powered route optimization for all transport modes"""
    }

    print("Raw Agent Outputs:")
    for role, text in outputs.items():
        print(f"\n{role.upper()}:")
        print(text)
    print()

    # Aggregate ideas into clusters
    clusters = aggregate_ideas(outputs)

    print("Idea Clustering Results:")
    print(f"Total Clusters: {len(clusters)}")
    for i, (key, ideas) in enumerate(clusters[:3], 1):  # Show top 3 clusters
        print(f"\n{i}. CLUSTER ({len(ideas)} ideas):")
        print(f"   Key: {key[:60]}...")
        print(f"   Ideas: {len(ideas)}")

    print("\n" + "="*60 + "\n")


def demo_advanced_aggregation():
    """Demo advanced idea aggregation with ranking and analysis."""
    print("🚀 Advanced Idea Aggregation Demo")
    print("=" * 40)

    outputs = {
        "expert": """1. Implement comprehensive cybersecurity measures for IoT devices
2. Develop blockchain-based supply chain tracking systems
3. Create AI-powered predictive maintenance for industrial equipment
4. Establish quantum-resistant encryption standards""",

        "creative": """• Build holographic telepresence systems for remote collaboration
• Deploy swarm robotics for large-scale manufacturing automation
• Create neural interfaces for direct brain-computer communication
• Develop self-repairing infrastructure using nanotechnology""",

        "analyst": """- Optimize cloud computing architectures for edge processing
- Implement zero-trust security models across enterprise networks
- Develop sustainable computing practices to reduce carbon footprint
- Create automated compliance monitoring and reporting systems"""
    }

    print("Brainstorming Topic: Future of Technology Solutions")
    print()

    # Aggregate and rank ideas
    clusters = aggregate_ideas(outputs)
    ranked_ideas = rank_ideas_by_novelty(dict(clusters))

    # Generate comprehensive report
    report = generate_idea_report(ranked_ideas, top_n=3)
    print(report)

    # Find contradictory ideas
    contradictions = find_contradictory_ideas(outputs)
    if contradictions:
        print(f"\n⚠️ POTENTIAL CONTRADICTIONS FOUND: {len(contradictions)}")
        for i, contra in enumerate(contradictions[:2], 1):  # Show first 2
            print(f"\n{i}. {contra['agent1']} vs {contra['agent2']}:")
            print(f"   {contra['contradiction_type']}")
            print(f"   • {contra['idea1'][:50]}...")
            print(f"   • {contra['idea2'][:50]}...")

    print("\n" + "="*60 + "\n")


def demo_novelty_ranking():
    """Demo novelty ranking with base ideas."""
    print("🎯 Novelty Ranking Demo")
    print("=" * 40)

    # Base ideas (existing concepts)
    base_ideas = [
        "Use solar panels for energy generation",
        "Implement recycling programs",
        "Build electric vehicle infrastructure"
    ]

    # New outputs to compare
    outputs = {
        "innovator": """1. Deploy space-based solar power stations beaming energy to Earth
2. Create molecular assembly for zero-waste manufacturing
3. Build quantum teleportation networks for instant communication
4. Develop consciousness uploading for digital immortality""",

        "pragmatist": """- Install additional solar panels on existing buildings
- Expand recycling centers with better sorting technology
- Add more electric vehicle charging stations
- Implement smart grid energy distribution"""
    }

    print("Base Ideas (existing concepts):")
    for idea in base_ideas:
        print(f"  • {idea}")
    print()

    print("New Agent Outputs:")
    for role, text in outputs.items():
        print(f"\n{role.upper()}:")
        for line in text.split('\n')[:2]:  # Show first 2 ideas
            if line.strip():
                print(f"  {line}")
    print()

    # Aggregate and rank with novelty scoring
    clusters = aggregate_ideas(outputs)
    ranked_ideas = rank_ideas_by_novelty(dict(clusters), base_ideas)

    print("Novelty Ranking Results:")
    for i, cluster in enumerate(ranked_ideas[:3], 1):
        print(f"\n{i}. NOVELTY: {cluster['novelty_score']:.2f} | QUALITY: {cluster['quality_score']:.2f}")
        print(f"   {cluster['cluster_representative'][:80]}...")
        print(f"   Cluster Size: {cluster['cluster_size']} ideas")


if __name__ == "__main__":
    demo_basic_aggregation()
    demo_advanced_aggregation()
    demo_novelty_ranking()
