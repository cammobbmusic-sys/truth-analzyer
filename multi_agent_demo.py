#!/usr/bin/env python3
"""
Multi-Agent Orchestrator Setup: Groq + HuggingFace + OpenRouter
Safe to run; set dry_run=False after adding valid API keys
"""

import os
from config import Config
from agents.factory import create_agent
from orchestrator.pipelines.verify_pipeline import VerificationPipeline
from orchestrator.pipelines.triangulation_orchestrator import TriangulationOrchestrator
from orchestrator.pipelines.brainstorm_orchestrator import BrainstormOrchestrator

def setup_agents():
    """Create agents from available providers"""
    agents = []

    # Try to create Groq agent
    try:
        groq_agent = create_agent({
            'name': 'groq-agent',
            'provider': 'groq',
            'model': 'llama-3.1-8b-instant'
        })
        agents.append(groq_agent)
        print("✓ Groq agent created")
    except Exception as e:
        print(f"⚠️  Groq agent failed: {str(e)[:50]}...")
        # Fallback to generic
        generic_agent = create_agent({
            'name': 'generic-groq',
            'provider': 'generic',
            'model': 'test'
        })
        agents.append(generic_agent)
        print("✓ Using generic agent as fallback for Groq")

    # Try to create HuggingFace agent
    try:
        hf_agent = create_agent({
            'name': 'hf-agent',
            'provider': 'huggingface',
            'model': 'tiiuae/falcon-7b-instruct'
        })
        agents.append(hf_agent)
        print("✓ HuggingFace agent created")
    except Exception as e:
        print(f"⚠️  HuggingFace agent failed: {str(e)[:50]}...")
        # Fallback to generic
        generic_agent = create_agent({
            'name': 'generic-hf',
            'provider': 'generic',
            'model': 'test'
        })
        agents.append(generic_agent)
        print("✓ Using generic agent as fallback for HuggingFace")

    # Try to create OpenRouter agent
    try:
        or_agent = create_agent({
            'name': 'or-agent',
            'provider': 'openrouter',
            'model': 'openchat/openchat-7b'
        })
        agents.append(or_agent)
        print("✓ OpenRouter agent created")
    except Exception as e:
        print(f"⚠️  OpenRouter agent failed: {str(e)[:50]}...")
        # Fallback to generic
        generic_agent = create_agent({
            'name': 'generic-or',
            'provider': 'generic',
            'model': 'test'
        })
        agents.append(generic_agent)
        print("✓ Using generic agent as fallback for OpenRouter")

    return agents

def main():
    print("🤖 MULTI-AGENT ORCHESTRATOR DEMO")
    print("=" * 50)
    print()

    # Setup agents with fallbacks
    print("Setting up agents...")
    agents = setup_agents()
    print(f"Created {len(agents)} agents")
    print()

    # Verification Pipeline Example
    print("1️⃣ TESTING VERIFICATION PIPELINE")
    print("-" * 30)

    verify_pipeline = VerificationPipeline(similarity_threshold=0.75)
    verify_report = verify_pipeline.run(
        text="The Eiffel Tower is in Paris and was completed in 1889.",
        agent_instances=agents,
        dry_run=True  # Safe mode
    )

    print("✅ Verification Report:")
    print(f"  Verdict: {verify_report.get('verdict', 'unknown')}")
    print(".3f")
    print(f"  Agents: {verify_report.get('agents_count', 0)}")
    print()

    # Triangulation Orchestrator Example
    print("2️⃣ TESTING TRIANGULATION ORCHESTRATOR")
    print("-" * 30)

    tri_orch = TriangulationOrchestrator(max_agents=3, max_retries=2, similarity_threshold=0.75)
    tri_report = tri_orch.run(
        text="Paris is the capital of France.",
        agent_instances=agents,
        dry_run=True  # Safe mode
    )

    print("✅ Triangulation Report:")
    print(f"  Verdict: {tri_report.get('verdict', 'unknown')}")
    print(".3f")
    print(f"  Agent reports: {len(tri_report.get('agent_reports', []))}")
    print()

    # Brainstorm Orchestrator Example
    print("3️⃣ TESTING BRAINSTORM ORCHESTRATOR")
    print("-" * 30)

    brain_orch = BrainstormOrchestrator(max_agents=3, dry_run=True)
    brain_report = brain_orch.run(
        prompt="Generate innovative ideas for reducing plastic waste in cities",
        agent_instances=agents,
        run_verification=False  # Skip verification for speed
    )

    print("✅ Brainstorm Report:")
    print(f"  Prompt: {brain_report.get('prompt', '')[:50]}...")
    print(f"  Ideas generated: {len(brain_report.get('ideas', []))}")

    for idx, idea in enumerate(brain_report.get('ideas', [])[:3]):  # Show first 3
        print(f"  Agent {idx} Idea: {idea.get('idea', '')[:80]}...")
    print()

    print("=" * 50)
    print("🎉 MULTI-AGENT SYSTEM DEMO COMPLETE!")
    print()
    print("To enable live API calls:")
    print("1. Set your API keys as environment variables:")
    print("   export GROQ_API_KEY=your_key")
    print("   export HF_TOKEN=your_token")
    print("   export OPENROUTER_API_KEY=your_key")
    print("2. Change dry_run=False in the function calls")
    print("3. Re-run for real AI responses!")

if __name__ == "__main__":
    main()
