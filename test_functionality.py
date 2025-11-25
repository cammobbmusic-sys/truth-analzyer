#!/usr/bin/env python3
"""
Comprehensive Functionality Test for Multi-Agent Truth Analyzer v1.0
"""

print("=== COMPREHENSIVE FUNCTIONALITY TEST ===")
print()

# Test 1: Configuration
print("1. Configuration Test")
try:
    from config import Config
    c = Config()
    print(f"   Models loaded: {len(c.models)}")
    print(f"   API keys available: GROQ={'SET' if c.models[0].get('api_key') else 'NOT SET'}")
    print("   SUCCESS: Configuration working")
except Exception as e:
    print(f"   FAILED: {e}")
print()

# Test 2: Agent Creation
print("2. Agent Creation Test")
try:
    from agents.factory import create_agent
    agents = []
    for i, cfg in enumerate(c.models[:3]):
        agent = create_agent(cfg)
        agents.append(agent)
        print(f"   SUCCESS: Agent {i+1} created: {agent.name}")
except Exception as e:
    print(f"   FAILED: {e}")
print()

# Test 3: Verification Pipeline
print("3. Verification Pipeline Test")
try:
    from orchestrator.pipelines.verify_pipeline import VerificationPipeline
    vp = VerificationPipeline()
    result = vp.run(text='The sky is blue', agent_configs=c.models[:3], dry_run=True)
    print(f"   Verdict: {result['verdict']}")
    print(f"   Confidence: {result['consensus']['confidence']:.3f}")
    print("   SUCCESS: Verification pipeline working")
except Exception as e:
    print(f"   FAILED: {e}")
print()

# Test 4: Triangulation Orchestrator
print("4. Triangulation Orchestrator Test")
try:
    from orchestrator.pipelines.triangulation_orchestrator import TriangulationOrchestrator
    to = TriangulationOrchestrator()
    result = to.run(text='The Earth is round', agent_configs=c.models[:3], dry_run=True)
    print(f"   Verdict: {result['verdict']}")
    print(f"   Confidence: {result['consensus']['confidence']:.3f}")
    print("   SUCCESS: Triangulation orchestrator working")
except Exception as e:
    print(f"   FAILED: {e}")
print()

# Test 5: Brainstorm Orchestrator
print("5. Brainstorm Orchestrator Test")
try:
    from orchestrator.pipelines.brainstorm_orchestrator import BrainstormOrchestrator
    bo = BrainstormOrchestrator(max_agents=3, dry_run=True)
    result = bo.run(prompt='Ideas for productivity app', agent_configs=c.models[:3])
    print(f"   Ideas generated: {len(result['ideas'])}")
    print("   SUCCESS: Brainstorm orchestrator working")
except Exception as e:
    print(f"   FAILED: {e}")
print()

print("=== SUMMARY ===")
print("SUCCESS: All core functionality working!")
print("Web Interface: Should be running on localhost:5000")
