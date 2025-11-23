#!/usr/bin/env python3
"""
Demo script showing the cross-validation functionality.
"""

from utils.cross_validation import cross_validate, analyze_consensus, generate_consensus_report
from agents.orchestrator import Orchestrator
from agents.model_agent import ModelAgent


def demo_basic_cross_validation():
    """Demo basic cross-validation functionality."""
    print("🔍 Basic Cross-Validation Demo")
    print("=" * 40)

    # Sample outputs from different agents
    outputs = {
        "expert": "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
        "creative": "ML represents the creative application of AI techniques to solve complex problems through data-driven insights.",
        "analyst": "Machine learning constitutes a subset of AI, utilizing statistical methods and algorithms for pattern recognition."
    }

    print("Sample Outputs:")
    for role, text in outputs.items():
        print(f"\n{role.upper()}:")
        print(f"  {text}")
    print()

    # Basic cross-validation
    verified, flagged = cross_validate(outputs, threshold=0.7)

    print("Cross-Validation Results:")
    print(f"Verified statements: {len(verified)}")
    print(f"Flagged discrepancies: {len(flagged)}")
    print()

    if verified:
        print("✅ VERIFIED STATEMENTS:")
        for i, stmt in enumerate(verified, 1):
            print(f"{i}. {stmt}")
    print()

    if flagged:
        print("⚠️ FLAGGED DISCREPANCIES:")
        for role1, role2, text1, text2 in flagged:
            print(f"• {role1} vs {role2}:")
            print(f"  Similarity below threshold")
    print()


def demo_orchestrator_integration():
    """Demo cross-validation integrated with orchestrator."""
    print("🔗 Orchestrator Integration Demo")
    print("=" * 40)

    # Create agents
    agents = [
        ModelAgent("cursor-fast", "expert"),
        ModelAgent("cursor-slow", "creative"),
        ModelAgent("cursor-balanced", "analyst")
    ]

    orchestrator = Orchestrator(agents)

    # Run verification
    claim = "Quantum computing will solve all computational problems"
    print(f"Verification Claim: {claim}")

    results = orchestrator.run_analysis(claim, mode="verification")

    # Generate detailed consensus report
    report = generate_consensus_report(results['individual_responses'])

    print("\n" + report)
    print()

    # Show consensus analysis
    consensus = results['consensus']
    print("📊 Consensus Summary:")
    print(f"  Agreement Score: {consensus['agreement_score']:.2f}")
    print(f"  Confidence Score: {consensus['confidence_score']:.2f}")
    print(f"  Consensus Achieved: {consensus['consensus_achieved']}")
    print(f"  Verified Statements: {len(consensus['verified_statements'])}")
    print(f"  Discrepancies Found: {consensus['discrepancies_found']}")


def demo_similarity_thresholds():
    """Demo how different similarity thresholds affect results."""
    print("🎯 Similarity Threshold Effects Demo")
    print("=" * 40)

    outputs = {
        "agent1": "The weather is sunny today",
        "agent2": "It's a sunny day outside",
        "agent3": "The forecast shows rain tomorrow"
    }

    thresholds = [0.5, 0.7, 0.9]

    for threshold in thresholds:
        print(f"\nThreshold: {threshold}")
        analysis = analyze_consensus(outputs, threshold=threshold)
        print(f"  Agreement Score: {analysis['agreement_score']:.2f}")
        print(f"  Verified: {len(analysis['verified_statements'])}")
        print(f"  Flagged: {len(analysis['flagged_discrepancies'])}")


if __name__ == "__main__":
    demo_basic_cross_validation()
    print("\n" + "="*60 + "\n")
    demo_orchestrator_integration()
    print("\n" + "="*60 + "\n")
    demo_similarity_thresholds()
