from orchestrator.pipelines.debate_orchestrator import DebateOrchestrator
from debate_scoring_system import DebateScoringSystem, integrate_scoring_with_debate

# === Debate Config ===
debate_topic = "Electric vehicles are better for the environment than gasoline cars."
num_rounds = 2
absolute_truth = True

# Agent configurations (not instances - the orchestrator creates them)
agent_configs = [
    {"name": "Groq Advocate", "provider": "groq", "model": "llama-3.1-8b-instant"},
    {"name": "OpenRouter Skeptic", "provider": "openrouter", "model": "anthropic/claude-3-haiku"},
    {"name": "Cohere Synthesizer", "provider": "cohere", "model": "command-nightly"},
]

# Initialize Debate Orchestrator (corrected constructor)
debate = DebateOrchestrator(
    max_rounds=num_rounds,
    absolute_truth_mode=absolute_truth
)

# Run debate (corrected method call)
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

# === Evaluation by Agent 3 (Cohere Synthesizer) ===
def evaluate_debate_winner(result):
    """
    Cohere Synthesizer evaluates the debate based on:
      - Absolute truth compliance
      - Persuasiveness and evidence quality
      - Logical consistency
      - Engagement with counterpoints
      - Balance and synthesis of arguments

    Returns: winner assessment and detailed rationale
    """

    # Extract debate data
    rounds = result.get('rounds', [])
    final_summary = result.get('final_summary', {})

    # Initialize scoring
    agent_scores = {
        "Groq Advocate": {"score": 0, "criteria": []},
        "OpenRouter Skeptic": {"score": 0, "criteria": []},
        "Cohere Synthesizer": {"score": 0, "criteria": []}
    }

    # Analyze each round
    for round_data in rounds:
        for message in round_data['messages']:
            agent_name = message.get('agent', '')
            if agent_name not in agent_scores:
                continue

            score_entry = agent_scores[agent_name]
            content = message.get('message', '')

            # Scoring criteria
            score = 0
            criteria = []

            # Evidence quality (mentions studies, data, facts)
            if any(word in content.lower() for word in ['study', 'research', 'data', 'evidence', 'according to']):
                score += 2
                criteria.append("Strong evidence use")

            # Logical consistency (uses words like therefore, however, consequently)
            if any(word in content.lower() for word in ['therefore', 'however', 'consequently', 'thus', 'hence']):
                score += 1
                criteria.append("Logical reasoning")

            # Engagement (responds to other agents)
            if any(name in content for name in ["Groq Advocate", "OpenRouter Skeptic", "Cohere Synthesizer"]):
                score += 1
                criteria.append("Direct engagement")

            # Synthesis ability (finds common ground, balances views)
            if any(word in content.lower() for word in ['balance', 'common ground', 'synthesis', 'both sides', 'comprehensive']):
                score += 1
                criteria.append("Balanced synthesis")

            # Absolute truth compliance (high confidence, factual accuracy)
            if message.get('error') is None:  # No errors = good execution
                score += 1
                criteria.append("Error-free execution")

            score_entry['score'] += score
            score_entry['criteria'].extend(criteria)

    # Determine winner based on total score
    winner = max(agent_scores.keys(), key=lambda x: agent_scores[x]['score'])
    winner_score = agent_scores[winner]['score']

    # Check for ties
    tied_agents = [agent for agent, data in agent_scores.items() if data['score'] == winner_score]
    if len(tied_agents) > 1:
        winner = "Tie between " + ", ".join(tied_agents)

    # Generate detailed rationale as Cohere Synthesizer
    rationale = f"""As the Cohere Synthesizer, I have evaluated this debate using multiple criteria:

**Scoring Results:**
{chr(10).join(f"• {agent}: {data['score']} points {data['criteria']}" for agent, data in agent_scores.items())}

**Winner Determination:**
{winner} emerged as the strongest debater with {winner_score} total points.

**Key Strengths Observed:**
"""

    # Add specific strengths based on winner
    if "Groq Advocate" in winner:
        rationale += "- Exceptional evidence-based argumentation\n- Strong focus on empirical data and research\n- Effective counterpoint construction"
    elif "OpenRouter Skeptic" in winner:
        rationale += "- Rigorous questioning of assumptions\n- Critical analysis of evidence quality\n- Effective challenge of established viewpoints"
    elif "Cohere Synthesizer" in winner:
        rationale += "- Balanced integration of multiple perspectives\n- Effective synthesis of complex arguments\n- Comprehensive conflict resolution"
    else:
        rationale += "- All agents demonstrated strong debate skills\n- Multiple valid perspectives presented\n- No clear dominant strategy emerged"

    # Add final assessment
    rationale += f"""

**Final Assessment:**
This debate demonstrates the power of multi-agent AI discussion, with each agent bringing unique strengths to create a more comprehensive analysis than any single agent could achieve alone."""

    return winner, rationale

# === Advanced Debate Evaluation using Scoring System ===
def evaluate_debate_with_scoring_system(result):
    """
    Use the comprehensive DebateScoringSystem to evaluate debate performance.
    This provides detailed, criteria-based scoring across multiple rounds.
    """
    # Use the integrated scoring system
    debate_scores = integrate_scoring_with_debate(result)

    winner = debate_scores['winner']
    rationale = debate_scores['rationale']

    # Format the results in a Cohere Synthesizer style
    synthesizer_summary = f"""**COHERE SYNTHESIZER EVALUATION REPORT**

**Debate Winner:** {winner}

**Scoring Methodology:**
- **Evidence (25%)**: Quality and relevance of cited data and research
- **Logic (25%)**: Coherence and soundness of reasoning
- **Engagement (20%)**: Direct responses to other agents' arguments
- **Synthesis (20%)**: Integration of multiple perspectives
- **Truth Compliance (10%)**: Accuracy and factual correctness

**Detailed Rationale:**
{rationale}

**Final Assessment:**
This evaluation demonstrates the power of structured, multi-criteria assessment in AI debate analysis. The scoring system provides objective metrics while preserving the nuanced understanding that only an AI synthesizer can achieve."""

    return winner, synthesizer_summary, debate_scores

# Run evaluation
winner, rationale, detailed_scores = evaluate_debate_with_scoring_system(result)

# === Output ===
print("=== DEBATE RESULTS ===")
print(f"Topic: {result.get('topic', 'Unknown')}")
print(f"Total Rounds: {result.get('total_rounds', 0)}")
print(f"Total Messages: {result.get('total_messages', 0)}")
print(f"Absolute Truth Mode: {'Enabled' if absolute_truth else 'Disabled'}")

if 'rounds' in result and result['rounds']:
    print("\n=== DEBATE TRANSCRIPT ===")
    for round_data in result['rounds']:
        print(f"\n--- Round {round_data['round_number']} ---")
        for message in round_data['messages']:
            role_indicator = ""
            if message.get('role'):
                role_indicator = f" ({message['role'].replace('_', ' ')})"
            msg_text = message.get('message', 'No message')
            print(f"[AI] {message.get('agent', 'Unknown')}{role_indicator}: {msg_text[:150]}{'...' if len(msg_text) > 150 else ''}")

print("\n=== FINAL SUMMARY ===")
if result.get('final_summary'):
    summary = result['final_summary']
    print(f"Summarized by: {summary.get('summarizer_agent', 'Unknown')}")

    if summary.get('verification_result'):
        verification = summary['verification_result']
        consensus = verification.get('consensus', {})
        print(f"Absolute Truth Verdict: {summary.get('absolute_truth_verdict', 'N/A')}")
        print(".3f")

    print(f"\nSummary:\n{summary.get('summary_text', 'No summary available')}")

print("\n=== ADVANCED DEBATE SCORING ANALYSIS ===")
print(f"Winner: {winner}")
print(f"Rationale:\n{rationale}")

# Show detailed scoring breakdown
print(f"\n=== DETAILED SCORING BREAKDOWN ===")
print(f"Total Rounds Scored: {detailed_scores['round_count']}")
print("\nFinal Scores by Agent:")
for agent, score in sorted(detailed_scores['totals'].items(), key=lambda x: x[1], reverse=True):
    print(f"  {agent}: {score:.1f} points")

print("\nAverage Scores per Criterion:")
criteria_totals = detailed_scores['criteria_totals']
for agent in criteria_totals:
    print(f"  {agent}:")
    agent_criteria = criteria_totals[agent]
    avg_scores = {criterion: score / detailed_scores['round_count']
                 for criterion, score in agent_criteria.items()}
    for criterion, avg_score in sorted(avg_scores.items(), key=lambda x: x[1], reverse=True):
        criterion_name = criterion.replace('_', ' ').title()
        print(f"    - {criterion_name}: {avg_score:.1f}/10")

print("\n=== DEBATE STATISTICS ===")
print(f"Total Response Time: {result.get('total_response_time', 0):.2f} seconds")
print(f"Mode: {'Simulation' if result.get('dry_run') else 'Live API'}")
