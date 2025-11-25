from orchestrator.pipelines.debate_orchestrator import DebateOrchestrator

# === Debate Configuration ===
debate_topic = "Electric vehicles are better for the environment than gasoline cars."
num_rounds = 2
absolute_truth = True  # Enable strict verification

# Define agent configurations (for debate orchestrator)
agent_configs = [
    {"name": "Groq Advocate", "provider": "groq", "model": "llama-3.1-8b-instant"},
    {"name": "OpenRouter Skeptic", "provider": "openrouter", "model": "anthropic/claude-3-haiku"},
    {"name": "Cohere Synthesizer", "provider": "cohere", "model": "command-nightly"},
]

# Initialize Debate Orchestrator
debate = DebateOrchestrator(
    max_rounds=num_rounds,
    absolute_truth_mode=absolute_truth
)

# Run debate
try:
    result = debate.run(
        topic=debate_topic,
        agent_configs=agent_configs,
        dry_run=True,  # Set to False for real API calls
        absolute_truth_mode=absolute_truth
    )

    # Check for errors
    if 'error' in result:
        print(f"❌ Debate failed: {result['error']}")
        exit(1)

    # === Output Results ===
    print("\n=== DEBATE RESULTS ===")
    print(f"Topic: {result.get('topic', 'Unknown')}")
    print(f"Total Rounds: {result.get('total_rounds', 0)}")
    print(f"Total Messages: {result.get('total_messages', 0)}")
    print(f"Absolute Truth Mode: {'Enabled' if absolute_truth else 'Disabled'}")

except Exception as e:
    print(f"❌ Error running debate: {e}")
    exit(1)

if 'rounds' in result and result['rounds']:
    print("\n=== DEBATE TRANSCRIPT ===")
    for round_data in result['rounds']:
        print(f"\n--- Round {round_data['round_number']} ---")
        for message in round_data['messages']:
            if message.get('error') and message['error'] != 'agent_unavailable':
                print(f"❌ {message.get('agent', 'Unknown')}: ERROR - {message['error']}")
            else:
                role_indicator = ""
                if message.get('role'):
                    role_indicator = f" ({message['role'].replace('_', ' ')})"
                msg_text = message.get('message', 'No message')
                print(f"🤖 {message.get('agent', 'Unknown')}{role_indicator}: {msg_text[:150]}{'...' if len(msg_text) > 150 else ''}")
else:
    print("\n=== DEBATE TRANSCRIPT ===")
    print("❌ No debate rounds available - agents may have failed to initialize")

print("\n=== FINAL SUMMARY ===")
if result.get('final_summary'):
    summary = result['final_summary']
    print(f"Summarized by: {summary.get('summarizer_agent', 'Unknown')}")

    if summary.get('verification_result'):
        verification = summary['verification_result']
        consensus = verification.get('consensus', {})
        print(f"Absolute Truth Verdict: {summary.get('absolute_truth_verdict', 'N/A')}")
        print(f"Confidence: {consensus.get('confidence', 0):.3f}")

    print(f"\nSummary:\n{summary.get('summary_text', 'No summary available')}")
else:
    print("No final summary available")

print("\n=== DEBATE STATISTICS ===")
print(f"Total Response Time: {result.get('total_response_time', 0):.2f} seconds")
print(f"Mode: {'Simulation' if result.get('dry_run') else 'Live API'}")
