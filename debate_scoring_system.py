"""
Advanced Debate Scoring System for Multi-Agent Truth Analyzer

This module provides comprehensive evaluation metrics for AI debates,
tracking performance across multiple criteria and rounds.
"""

from typing import Dict, List, Any, Optional
import json
from datetime import datetime

class DebateScoringSystem:
    """
    Comprehensive scoring system for evaluating AI agent debate performance.
    """

    def __init__(self):
        # Scoring criteria weights
        self.criteria_weights = {
            "evidence": 0.25,      # Quality and quantity of evidence cited
            "logic": 0.25,         # Logical consistency and reasoning
            "engagement": 0.20,    # Direct engagement with other agents
            "synthesis": 0.20,     # Ability to synthesize multiple viewpoints
            "truth_compliance": 0.10  # Adherence to factual accuracy
        }

        # Score ranges for each criterion
        self.score_ranges = {
            "evidence": (0, 10),      # 0-10 scale
            "logic": (0, 10),         # 0-10 scale
            "engagement": (0, 10),    # 0-10 scale
            "synthesis": (0, 10),     # 0-10 scale
            "truth_compliance": (0, 5) # 0-5 scale (bonus factor)
        }

    def initialize_debate_scores(self, agent_names: List[str]) -> Dict[str, Any]:
        """
        Initialize scoring structure for a debate.
        """
        return {
            "rounds": [],
            "totals": {agent: 0 for agent in agent_names},
            "criteria_totals": {agent: {criterion: 0 for criterion in self.criteria_weights.keys()}
                              for agent in agent_names},
            "round_count": 0,
            "winner": None,
            "rationale": "",
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "criteria_weights": self.criteria_weights,
                "agent_names": agent_names
            }
        }

    def score_round(self, debate_scores: Dict[str, Any], round_number: int,
                   agent_scores: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        """
        Score a single debate round for all agents.
        """
        # Validate agent scores
        for agent_name, scores in agent_scores.items():
            for criterion, score in scores.items():
                if criterion not in self.criteria_weights:
                    raise ValueError(f"Unknown criterion: {criterion}")
                min_score, max_score = self.score_ranges[criterion]
                if not (min_score <= score <= max_score):
                    raise ValueError(f"Score {score} for {criterion} out of range [{min_score}, {max_score}]")

        # Add round to debate scores
        round_data = {
            "round_number": round_number,
            "agents": agent_scores.copy(),
            "timestamp": datetime.now().isoformat()
        }
        debate_scores["rounds"].append(round_data)
        debate_scores["round_count"] += 1

        # Update totals
        for agent_name, scores in agent_scores.items():
            # Calculate weighted score for this round
            weighted_score = 0
            for criterion, score in scores.items():
                weight = self.criteria_weights[criterion]
                weighted_score += score * weight

            debate_scores["totals"][agent_name] += weighted_score

            # Update criteria totals
            for criterion, score in scores.items():
                debate_scores["criteria_totals"][agent_name][criterion] += score

        return debate_scores

    def determine_winner(self, debate_scores: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the debate winner based on total scores.
        """
        totals = debate_scores["totals"]
        max_score = max(totals.values())

        # Find all agents with the maximum score
        winners = [agent for agent, score in totals.items() if score == max_score]

        if len(winners) == 1:
            winner = winners[0]
        else:
            winner = f"Tie between {', '.join(winners)}"

        # Generate detailed rationale
        rationale = self._generate_winner_rationale(debate_scores, winner, max_score)

        debate_scores["winner"] = winner
        debate_scores["rationale"] = rationale
        debate_scores["metadata"]["completed_at"] = datetime.now().isoformat()

        return debate_scores

    def _generate_winner_rationale(self, debate_scores: Dict[str, Any],
                                 winner: str, winner_score: float) -> str:
        """
        Generate detailed rationale for winner determination.
        """
        if "Tie between" in winner:
            tied_agents = winner.replace("Tie between ", "").split(", ")
            rationale = f"This debate resulted in a tie between {', '.join(tied_agents)}, each achieving a total score of {winner_score:.1f} points. "
        else:
            rationale = f"{winner} emerged as the winner with a total score of {winner_score:.1f} points. "

        # Analyze performance by criteria
        criteria_totals = debate_scores["criteria_totals"]
        rounds_completed = debate_scores["round_count"]

        rationale += f"\n\n**Performance Analysis (across {rounds_completed} rounds):**\n"

        # Sort agents by total score
        sorted_agents = sorted(debate_scores["totals"].items(), key=lambda x: x[1], reverse=True)

        for agent, total_score in sorted_agents:
            rationale += f"\n**{agent}** (Total: {total_score:.1f}):\n"

            # Show criteria breakdown
            agent_criteria = criteria_totals[agent]
            avg_scores = {criterion: score / rounds_completed
                         for criterion, score in agent_criteria.items()}

            sorted_criteria = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
            for criterion, avg_score in sorted_criteria:
                rationale += f"  - {criterion.replace('_', ' ').title()}: {avg_score:.1f}/10\n"

        # Add final assessment
        rationale += "\n**Key Factors in Winner Determination:**\n"
        rationale += "- **Evidence Quality**: Strength and relevance of cited data and research\n"
        rationale += "- **Logical Consistency**: Coherence and soundness of arguments\n"
        rationale += "- **Engagement Level**: Direct responses to other agents' points\n"
        rationale += "- **Synthesis Ability**: Integration of multiple perspectives\n"
        rationale += "- **Truth Compliance**: Accuracy and factual correctness\n"

        return rationale

    def export_scores(self, debate_scores: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Export debate scores to JSON format.
        """
        if filename:
            with open(filename, 'w') as f:
                json.dump(debate_scores, f, indent=2, default=str)
            return f"Scores exported to {filename}"
        else:
            return json.dumps(debate_scores, indent=2, default=str)

    def get_performance_summary(self, debate_scores: Dict[str, Any]) -> str:
        """
        Generate a human-readable performance summary.
        """
        summary = "DEBATE PERFORMANCE SUMMARY\n"
        summary += "=" * 40 + "\n\n"

        summary += f"Winner: {debate_scores['winner']}\n"
        summary += f"Rounds Completed: {debate_scores['round_count']}\n\n"

        summary += "FINAL SCORES:\n"
        sorted_scores = sorted(debate_scores['totals'].items(), key=lambda x: x[1], reverse=True)
        for agent, score in sorted_scores:
            summary += f"  {agent}: {score:.1f} points\n"

        summary += f"\nRATIONALE:\n{debate_scores['rationale']}\n"

        return summary


# Example usage and integration with DebateOrchestrator
def integrate_scoring_with_debate(debate_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Integrate the scoring system with actual debate results.
    This function demonstrates how to apply scoring to real debate transcripts.
    """
    scoring_system = DebateScoringSystem()

    # Extract agent names from debate result
    agent_names = []
    if 'rounds' in debate_result and debate_result['rounds']:
        # Get agent names from first round messages
        first_round = debate_result['rounds'][0]
        agent_names = list(set(msg.get('agent', 'Unknown')
                             for msg in first_round.get('messages', [])))

    # Initialize scoring
    debate_scores = scoring_system.initialize_debate_scores(agent_names)

    # For demonstration, create sample scores based on message content
    # In a real implementation, this would analyze the actual message content
    for round_idx, round_data in enumerate(debate_result.get('rounds', [])):
        round_scores = {}

        for agent in agent_names:
            # Analyze messages for this agent in this round
            agent_messages = [msg for msg in round_data.get('messages', [])
                            if msg.get('agent') == agent]

            # Simple scoring logic (replace with actual NLP analysis)
            evidence_score = 7  # Default good score
            logic_score = 8
            engagement_score = 6
            synthesis_score = 5
            truth_score = 4

            # Adjust scores based on message content (simplified)
            for msg in agent_messages:
                content = msg.get('message', '').lower()

                # Evidence indicators
                if any(word in content for word in ['study', 'research', 'data', 'evidence']):
                    evidence_score = min(10, evidence_score + 2)

                # Logic indicators
                if any(word in content for word in ['therefore', 'consequently', 'thus']):
                    logic_score = min(10, logic_score + 1)

                # Engagement indicators
                if any(name.lower() in content for name in agent_names if name != agent):
                    engagement_score = min(10, engagement_score + 2)

                # Synthesis indicators
                if any(word in content for word in ['balance', 'both sides', 'comprehensive']):
                    synthesis_score = min(10, synthesis_score + 2)

            round_scores[agent] = {
                "evidence": evidence_score,
                "logic": logic_score,
                "engagement": engagement_score,
                "synthesis": synthesis_score,
                "truth_compliance": truth_score
            }

        # Score the round
        debate_scores = scoring_system.score_round(debate_scores, round_idx + 1, round_scores)

    # Determine winner
    debate_scores = scoring_system.determine_winner(debate_scores)

    return debate_scores


# Demonstration function
def demonstrate_scoring_system():
    """
    Demonstrate the complete scoring system.
    """
    scoring_system = DebateScoringSystem()

    # Initialize for 3 agents
    agent_names = ["Groq Advocate", "OpenRouter Skeptic", "Cohere Synthesizer"]
    debate_scores = scoring_system.initialize_debate_scores(agent_names)

    # Score Round 1
    round1_scores = {
        "Groq Advocate": {"evidence": 8, "logic": 7, "engagement": 9, "synthesis": 6, "truth_compliance": 4},
        "OpenRouter Skeptic": {"evidence": 7, "logic": 8, "engagement": 7, "synthesis": 5, "truth_compliance": 5},
        "Cohere Synthesizer": {"evidence": 6, "logic": 9, "engagement": 8, "synthesis": 10, "truth_compliance": 4}
    }
    debate_scores = scoring_system.score_round(debate_scores, 1, round1_scores)

    # Score Round 2
    round2_scores = {
        "Groq Advocate": {"evidence": 9, "logic": 8, "engagement": 8, "synthesis": 7, "truth_compliance": 5},
        "OpenRouter Skeptic": {"evidence": 8, "logic": 7, "engagement": 9, "synthesis": 6, "truth_compliance": 4},
        "Cohere Synthesizer": {"evidence": 7, "logic": 8, "engagement": 9, "synthesis": 9, "truth_compliance": 5}
    }
    debate_scores = scoring_system.score_round(debate_scores, 2, round2_scores)

    # Determine winner
    debate_scores = scoring_system.determine_winner(debate_scores)

    # Display results
    print("DEBATE SCORING RESULTS")
    print("=" * 30)
    print(f"Winner: {debate_scores['winner']}")
    print(f"Total Rounds: {debate_scores['round_count']}")
    print("\nFinal Scores:")
    for agent, score in debate_scores['totals'].items():
        print(f"  {agent}: {score:.1f} points")

    print(f"\nWinner Rationale:\n{debate_scores['rationale']}")

    return debate_scores


if __name__ == "__main__":
    # Run demonstration
    demonstrate_scoring_system()
