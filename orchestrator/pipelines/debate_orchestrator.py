"""
Debate Orchestrator - Multi-Agent Debate System

This module implements a debate system where multiple AI agents engage in
structured debate rounds, building on each other's arguments to reach
a more refined conclusion.
"""

import time
from typing import List, Dict, Optional, Any
import logging

# Import project utilities
try:
    from agents.factory import create_agent
except Exception:
    create_agent = None

# Import verification pipeline for absolute truth mode
try:
    from orchestrator.pipelines.verify_pipeline import VerificationPipeline
except Exception:
    VerificationPipeline = None

# Ensure environment variables are loaded
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass  # dotenv not available, assume env vars are set

logger = logging.getLogger(__name__)


# ---------------------------------------------
#  MESSAGE-PASSING LAYER (shared chatroom)
# ---------------------------------------------

class DebateRoom:
    """Manages the shared conversation transcript for debates."""

    def __init__(self, topic: str, max_rounds: int = 3):
        self.topic = topic
        self.max_rounds = max_rounds
        self.transcript: List[Dict[str, str]] = []  # [{"agent": name, "text": message}]

    def add_message(self, agent: str, text: str):
        """Add a message to the transcript."""
        self.transcript.append({"agent": agent, "text": text})

    def get_chatroom_context(self) -> str:
        """Return the entire debate transcript as a formatted string."""
        if not self.transcript:
            return "No messages yet."

        out = "### Shared Debate Room Transcript\n\n"
        for i, t in enumerate(self.transcript, 1):
            out += f"**{t['agent']}** (Round {((i-1)//3)+1}):\n{t['text']}\n\n"
        return out

    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages in the transcript."""
        return self.transcript.copy()


# ---------------------------------------------
#  DEBATE PROMPT BUILDER
# ---------------------------------------------

def build_debate_prompt(agent_name: str, agent_role: str, debate_room: DebateRoom) -> str:
    """Constructs the prompt each agent receives for the debate."""

    transcript = debate_room.get_chatroom_context()

    return f"""You are **{agent_name}** in a 3-agent AI debate.

Your role: {agent_role}

Topic: **{debate_room.topic}**

**DEBATE RULES:**
- Attack weak arguments with evidence-based reasoning
- Defend strong claims logically
- Correct factual errors
- Build upon or challenge previous arguments
- Respond directly to specific claims made by other agents
- Quote or reference lines you agree/disagree with

At the end of your message:
- Ask ONE direct and provocative question aimed at another agent to continue the debate

### CURRENT DEBATE TRANSCRIPT:

{transcript}

### Your Response:
"""


# ---------------------------------------------
#  DEBATE ORCHESTRATOR
# ---------------------------------------------

class DebateOrchestrator:
    """
    Orchestrates multi-agent debates with structured rounds and message passing.
    """

    def __init__(self, max_rounds: int = 3, delay_between_agents: float = 0.5, absolute_truth_mode: bool = False):
        self.max_rounds = max_rounds
        self.delay_between_agents = delay_between_agents
        self.absolute_truth_mode = absolute_truth_mode

    def _instantiate_agents(self, agent_configs: List[Dict[str, Any]]) -> List[Any]:
        """Create agent instances from configurations."""
        agents = []
        if not agent_configs:
            return agents

        if create_agent is None:
            logger.warning('agents.factory.create_agent not importable; using placeholders.')
            return [None for _ in agent_configs]

        for cfg in agent_configs:
            try:
                agents.append(create_agent(cfg))
            except Exception as e:
                logger.exception('Failed to create agent: %s', e)
                agents.append(None)
        return agents

    def _get_agent_role(self, index: int) -> str:
        """Assign specialized roles to agents for debate diversity."""
        roles = [
            "Skeptical Analyst - Question assumptions and demand evidence",
            "Evidence Advocate - Provide data and research-based arguments",
            "Critical Synthesizer - Find common ground and resolve contradictions"
        ]
        return roles[index % len(roles)]

    def run_debate_round(self, agents: List[Any], debate_room: DebateRoom, round_no: int) -> Dict[str, Any]:
        """Run a single round of debate with all agents."""
        round_messages = []

        for idx, agent in enumerate(agents):
            if agent is None:
                # Fallback for missing agents
                fallback_msg = f"[AGENT {idx+1} UNAVAILABLE] Cannot participate in debate round {round_no}."
                debate_room.add_message(f"Agent {idx+1} (Unavailable)", fallback_msg)
                round_messages.append({"agent": f"Agent {idx+1}", "message": fallback_msg, "error": "agent_unavailable"})
                continue

            agent_name = getattr(agent, 'name', f'Agent {idx+1}')
            agent_role = self._get_agent_role(idx)

            try:
                # Build debate prompt
                prompt = build_debate_prompt(agent_name, agent_role, debate_room)

                # Get agent response
                start_time = time.time()
                response = agent.generate(prompt, max_tokens=512, temperature=0.8)
                response_time = time.time() - start_time

                # Add to debate room
                debate_room.add_message(agent_name, response)

                round_messages.append({
                    "agent": agent_name,
                    "role": agent_role,
                    "message": response,
                    "response_time": response_time,
                    "error": None
                })

                # Delay between agents for more natural conversation flow
                if idx < len(agents) - 1:
                    time.sleep(self.delay_between_agents)

            except Exception as e:
                error_msg = f"[ERROR] Agent {agent_name} failed in round {round_no}: {str(e)}"
                debate_room.add_message(agent_name, error_msg)
                round_messages.append({
                    "agent": agent_name,
                    "role": agent_role,
                    "message": error_msg,
                    "response_time": 0,
                    "error": str(e)
                })
                logger.exception(f'Agent {agent_name} failed in debate round {round_no}')

        return {
            "round_number": round_no,
            "messages": round_messages,
            "total_participants": len([m for m in round_messages if m["error"] != "agent_unavailable"])
        }

    def run(self,
            topic: str,
            agent_configs: Optional[List[Dict[str, Any]]] = None,
            agent_instances: Optional[List[Any]] = None,
            dry_run: bool = True,
            absolute_truth_mode: bool = False) -> Dict[str, Any]:
        """
        Run a complete multi-agent debate.

        Args:
            topic: The debate topic/question
            agent_configs: List of agent configurations
            agent_instances: Pre-instantiated agent objects
            dry_run: If True, avoid real API calls

        Returns:
            Complete debate results including transcript and summary
        """

        # Initialize debate room
        debate_room = DebateRoom(topic, max_rounds=self.max_rounds)

        # Get agents
        if agent_instances:
            agents = agent_instances[:3]  # Use up to 3 agents
        elif agent_configs:
            agents = self._instantiate_agents(agent_configs[:3])
        else:
            return {"error": "no_agents", "topic": topic, "dry_run": dry_run}

        # Ensure we have at least 2 agents for meaningful debate
        if len([a for a in agents if a is not None]) < 2:
            return {"error": "insufficient_agents", "topic": topic, "dry_run": dry_run}

        # Run debate rounds
        rounds_data = []
        total_response_time = 0

        for round_no in range(1, self.max_rounds + 1):
            logger.info(f'Starting debate round {round_no} on topic: {topic}')

            round_result = self.run_debate_round(agents, debate_room, round_no)
            rounds_data.append(round_result)

            # Accumulate response times
            for msg in round_result["messages"]:
                total_response_time += msg.get("response_time", 0)

            # Check if we should continue (could add convergence logic here)
            if round_no < self.max_rounds:
                time.sleep(1.0)  # Brief pause between rounds

        # Generate final summary
        final_summary = self._generate_debate_summary(debate_room, agents, self.absolute_truth_mode)

        # Compile results
        result = {
            "topic": topic,
            "total_rounds": len(rounds_data),
            "total_messages": len(debate_room.transcript),
            "rounds": rounds_data,
            "final_summary": final_summary,
            "transcript": debate_room.get_messages(),
            "total_response_time": total_response_time,
            "dry_run": dry_run
        }

        return result

    def _generate_debate_summary(self, debate_room: DebateRoom, agents: List[Any], absolute_truth_mode: bool = False) -> Dict[str, Any]:
        """Generate a final summary of the debate."""
        try:
            if absolute_truth_mode and VerificationPipeline:
                # Use verification pipeline for absolute truth mode
                logger.info('Using Absolute Truth Mode for debate summary verification')

                # Create a verification claim from the debate
                debate_summary = self._generate_basic_summary(debate_room, agents)
                verification_claim = f"Based on this debate, {debate_summary}"

                # Use verification pipeline with absolute truth mode
                pipeline = VerificationPipeline(absolute_truth_mode=True)
                verification_result = pipeline.run(
                    text=verification_claim,
                    agent_instances=agents,
                    dry_run=True,
                    absolute_truth_mode=True
                )

                # Combine debate summary with verification result
                return {
                    "summary_text": debate_summary,
                    "verification_result": verification_result,
                    "absolute_truth_verdict": verification_result.get('consensus', {}).get('verdict'),
                    "summarizer_agent": getattr(agents[0], 'name', 'Unknown') if agents else 'Unknown',
                    "error": None
                }

            else:
                # Standard debate summary
                return self._generate_basic_summary(debate_room, agents)

        except Exception as e:
            logger.exception('Failed to generate debate summary')
            return {
                "summary_text": f"[SUMMARY UNAVAILABLE] Error generating summary: {str(e)}",
                "summarizer_agent": "Error",
                "error": str(e)
            }

    def _generate_basic_summary(self, debate_room: DebateRoom, agents: List[Any]) -> Dict[str, Any]:
        """Generate a basic debate summary without verification."""
        try:
            # Use the first available agent to summarize
            summarizer = next((agent for agent in agents if agent is not None), None)
            if summarizer is None:
                return {"error": "no_summarizer_available"}

            full_transcript = debate_room.get_chatroom_context()

            summary_prompt = f"""
You are the neutral final judge of this AI debate.

Given the debate transcript below, produce a structured summary:

1. **FINAL CONCLUSION**: The main agreed-upon answer or position
2. **KEY ARGUMENTS**: The strongest points made by each agent
3. **RESOLVED ISSUES**: Points where agents reached agreement
4. **REMAINING DISAGREEMENTS**: Any unresolved conflicts
5. **CONFIDENCE SCORE**: Overall confidence in the conclusion (0-100)

Topic: {debate_room.topic}

{full_transcript}

Provide your summary in a clear, structured format.
"""

            summary = summarizer.generate(summary_prompt, max_tokens=1024, temperature=0.3)

            return {
                "summary_text": summary,
                "summarizer_agent": getattr(summarizer, 'name', 'Unknown'),
                "error": None
            }

        except Exception as e:
            logger.exception('Failed to generate basic debate summary')
            return {
                "summary_text": f"[SUMMARY UNAVAILABLE] Error generating summary: {str(e)}",
                "summarizer_agent": "Error",
                "error": str(e)
            }


# ---------------------------------------------
#  STANDALONE DEBATE FUNCTION
# ---------------------------------------------

def run_standalone_debate(topic: str, rounds: int = 3):
    """
    Run a debate using the provided simulated agents.
    This is the standalone version from the user's code.
    """
    room = DebateRoom(topic, max_rounds=rounds)

    # Simulated agent functions (replace with real API calls)
    def call_agent_groq(prompt: str) -> str:
        return f"[GROQ SIMULATED RESPONSE] {prompt[:200]}..."

    def call_agent_openrouter(prompt: str) -> str:
        return f"[OPENROUTER SIMULATED RESPONSE] {prompt[:200]}..."

    def call_agent_cohere(prompt: str) -> str:
        return f"[COHERE SIMULATED RESPONSE] {prompt[:200]}..."

    for round_no in range(1, rounds + 1):
        print(f"\n\n=== ROUND {round_no} ===\n")

        # Agent 1 - Groq
        p1 = build_debate_prompt("Agent 1 (Groq)", "Skeptical Analyst", room)
        a1 = call_agent_groq(p1)
        room.add_message("Agent 1 (Groq)", a1)
        print("A1:", a1[:180], "...\n")

        time.sleep(0.3)

        # Agent 2 - OpenRouter
        p2 = build_debate_prompt("Agent 2 (OpenRouter)", "Evidence Advocate", room)
        a2 = call_agent_openrouter(p2)
        room.add_message("Agent 2 (OpenRouter)", a2)
        print("A2:", a2[:180], "...\n")

        time.sleep(0.3)

        # Agent 3 - Cohere
        p3 = build_debate_prompt("Agent 3 (Cohere)", "Critical Synthesizer", room)
        a3 = call_agent_cohere(p3)
        room.add_message("Agent 3 (Cohere)", a3)
        print("A3:", a3[:180], "...\n")

        time.sleep(0.3)

    return room


def summarize_final_verdict(debate_room: DebateRoom) -> str:
    """Ask Groq (or any model) to summarize the debate into a final answer."""

    def call_agent_groq(prompt: str) -> str:
        return f"[GROQ SIMULATED SUMMARY] {prompt[:200]}..."

    full_transcript = debate_room.get_chatroom_context()

    prompt = f"""
You are the neutral final judge.

Given the 3-agent debate transcript below, produce:

1. The final agreed-upon answer
2. Any disagreements and how they were resolved
3. A confidence score (0–100)

Transcript:

{full_transcript}
"""

    return call_agent_groq(prompt)


if __name__ == "__main__":
    # Example standalone usage
    room = run_standalone_debate("What are the environmental impacts of electric vehicles?", rounds=2)
    print("\n\n=== FINAL VERDICT ===\n")
    result = summarize_final_verdict(room)
    print(result)
