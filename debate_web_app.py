from flask import Flask, render_template, request, jsonify
from orchestrator.pipelines.debate_orchestrator import DebateOrchestrator
from debate_scoring_system import integrate_scoring_with_debate, DebateScoringSystem
import os

app = Flask(__name__)

# Ensure environment variables are loaded
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ----------------------------
# Enhanced Scoring Functions using AI Analysis
# ----------------------------

def evaluate_evidence(message_content, agent_name):
    """Evaluate evidence quality in message content."""
    content = message_content.lower()

    score = 5  # Base score

    # Evidence indicators
    evidence_keywords = ['study', 'research', 'data', 'evidence', 'according to', 'statistics', 'report']
    if any(word in content for word in evidence_keywords):
        score += 2

    # Citations or specific references
    if any(char in content for char in ['[', ']', '(', ')']) or 'et al' in content:
        score += 1

    # Factual claims vs opinions
    opinion_words = ['i think', 'i believe', 'probably', 'maybe', 'perhaps']
    if not any(word in content for word in opinion_words):
        score += 1

    return min(10, max(0, score))

def evaluate_logic(message_content, agent_name):
    """Evaluate logical consistency in message content."""
    content = message_content.lower()

    score = 5  # Base score

    # Logical connectors
    logic_words = ['therefore', 'however', 'consequently', 'thus', 'hence', 'because', 'since']
    if any(word in content for word in logic_words):
        score += 2

    # Structured argumentation
    if any(word in content for word in ['first', 'second', 'third', 'finally', 'moreover']):
        score += 1

    # Counterargument handling
    if any(word in content for word in ['while', 'although', 'despite', 'notwithstanding']):
        score += 1

    return min(10, max(0, score))

def evaluate_engagement(message_content, agent_name, all_messages):
    """Evaluate how well agent engages with other participants."""
    content = message_content.lower()

    score = 3  # Base score

    # Direct references to other agents
    other_agents = [msg['agent'] for msg in all_messages if msg['agent'] != agent_name]
    for other_agent in other_agents:
        if other_agent.lower() in content or other_agent.split()[0].lower() in content:
            score += 2
            break

    # Response to previous arguments
    response_words = ['you said', 'you mentioned', 'your point', 'agree with', 'disagree with']
    if any(word in content for word in response_words):
        score += 1

    # Questions asked
    if '?' in content:
        score += 1

    return min(10, max(0, score))

def evaluate_synthesis(message_content, agent_name, all_messages):
    """Evaluate ability to synthesize multiple viewpoints."""
    content = message_content.lower()

    score = 3  # Base score

    # Synthesis indicators
    synthesis_words = ['balance', 'both sides', 'comprehensive', 'synthesis', 'integration',
                      'common ground', 'middle ground', 'nuanced view']
    if any(word in content for word in synthesis_words):
        score += 2

    # Multiple perspective recognition
    perspective_words = ['different perspective', 'other viewpoint', 'alternative view', 'complex issue']
    if any(word in content for word in perspective_words):
        score += 1

    # Resolution attempts
    resolution_words = ['compromise', 'solution', 'resolution', 'bridge the gap']
    if any(word in content for word in resolution_words):
        score += 1

    return min(10, max(0, score))

def evaluate_truth_compliance(message_content, agent_name):
    """Evaluate factual accuracy and truthfulness."""
    content = message_content.lower()

    score = 5  # Base score (neutral)

    # Factual disclaimers
    uncertain_words = ['i think', 'i believe', 'probably', 'maybe', 'perhaps', 'allegedly']
    if any(word in content for word in uncertain_words):
        score -= 1

    # Confidence indicators
    confident_words = ['clearly', 'obviously', 'definitely', 'certainly', 'undoubtedly']
    if any(word in content for word in confident_words):
        score += 1

    # Qualification of claims
    qualification_words = ['generally', 'typically', 'often', 'usually', 'in most cases']
    if any(word in content for word in qualification_words):
        score += 1

    return min(10, max(0, score))

def score_debate_round(round_messages):
    """
    Score all agents in a single debate round.

    Args:
        round_messages: List of message dictionaries from the round

    Returns:
        Dict with agent scores for each criterion
    """
    scores = {}

    # Group messages by agent
    agent_messages = {}
    for msg in round_messages:
        agent = msg.get('agent', 'Unknown')
        if agent not in agent_messages:
            agent_messages[agent] = []
        agent_messages[agent].append(msg)

    # Score each agent
    for agent, messages in agent_messages.items():
        # Combine all messages from this agent in this round
        combined_content = ' '.join([msg.get('text', '') for msg in messages])

        # Evaluate each criterion
        evidence_score = evaluate_evidence(combined_content, agent)
        logic_score = evaluate_logic(combined_content, agent)
        engagement_score = evaluate_engagement(combined_content, agent, round_messages)
        synthesis_score = evaluate_synthesis(combined_content, agent, round_messages)
        truth_score = evaluate_truth_compliance(combined_content, agent)

        scores[agent] = {
            "evidence": evidence_score,
            "logic": logic_score,
            "engagement": engagement_score,
            "synthesis": synthesis_score,
            "truth_compliance": truth_score
        }

    return scores

def create_enhanced_debate_response(debate_result, scoring_system=None):
    """
    Create an enhanced response that includes both debate results and scoring.
    """
    response = {
        "topic": debate_result.get("topic"),
        "total_rounds": debate_result.get("total_rounds"),
        "total_messages": debate_result.get("total_messages"),
        "dry_run": debate_result.get("dry_run"),
        "rounds": debate_result.get("rounds", []),
        "transcript": debate_result.get("transcript", []),
    }

    # Add final summary if available
    if "final_summary" in debate_result:
        summary = debate_result["final_summary"]
        response["final_summary"] = {
            "summary_text": summary.get("summary_text"),
            "summarizer_agent": summary.get("summarizer_agent"),
            "error": summary.get("error")
        }

        # Add absolute truth verification if available
        if "verification_result" in summary:
            verification = summary["verification_result"]
            consensus = verification.get("consensus", {})
            response["final_summary"]["absolute_truth_verdict"] = summary.get("absolute_truth_verdict")
            response["final_summary"]["confidence"] = consensus.get("confidence")

    # Add comprehensive scoring if requested
    if scoring_system:
        try:
            debate_scores = integrate_scoring_with_debate(debate_result)
            response["scoring"] = {
                "round_scores": debate_scores.get("rounds", []),
                "totals": debate_scores.get("totals", {}),
                "winner": debate_scores.get("winner"),
                "rationale": debate_scores.get("rationale"),
                "criteria_breakdown": debate_scores.get("criteria_totals", {})
            }
        except Exception as e:
            response["scoring_error"] = f"Scoring failed: {str(e)}"

    return response

# ----------------------------
# Flask Routes
# ----------------------------

@app.route('/')
def index():
    """Serve the debate dashboard."""
    return render_template("debate_dashboard.html")

@app.route('/run_debate', methods=['POST'])
def run_debate():
    """Run a debate and return results with optional scoring."""
    try:
        data = request.json
        topic = data.get('topic', 'Default Debate Topic')
        rounds = int(data.get('rounds', 2))
        absolute_truth = data.get('absolute_truth', False)
        include_scoring = data.get('include_scoring', True)  # New parameter

        # Validate inputs
        if not topic.strip():
            return jsonify({"error": "Topic cannot be empty"}), 400

        if not (1 <= rounds <= 5):
            return jsonify({"error": "Rounds must be between 1 and 5"}), 400

        # Configure agents
        agent_configs = [
            {"name": "Debater Alpha", "provider": "groq", "model": "llama-3.1-8b-instant"},
            {"name": "Debater Beta", "provider": "openrouter", "model": "anthropic/claude-3-haiku"},
            {"name": "Debater Gamma", "provider": "cohere", "model": "command-nightly"}
        ]

        # Run the debate
        debate = DebateOrchestrator(max_rounds=rounds, absolute_truth_mode=absolute_truth)
        result = debate.run(
            topic=topic,
            agent_configs=agent_configs,
            dry_run=True,  # Use dry_run for web interface
            absolute_truth_mode=absolute_truth
        )

        # Check for errors
        if 'error' in result:
            return jsonify({"error": result['error']}), 500

        # Create enhanced response
        if include_scoring:
            response = create_enhanced_debate_response(result, scoring_system=True)
        else:
            response = create_enhanced_debate_response(result, scoring_system=False)

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Debate execution failed: {str(e)}"}), 500

@app.route('/debate_stats', methods=['GET'])
def debate_stats():
    """Return general debate statistics."""
    return jsonify({
        "total_debates_supported": "Unlimited",
        "max_rounds": 5,
        "max_agents": 3,
        "scoring_criteria": ["evidence", "logic", "engagement", "synthesis", "truth_compliance"],
        "supported_providers": ["groq", "openrouter", "cohere"],
        "features": [
            "Multi-agent debates",
            "Round-based discussions",
            "Absolute Truth verification",
            "Comprehensive scoring",
            "Real-time evaluation"
        ]
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
