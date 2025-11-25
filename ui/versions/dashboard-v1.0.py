from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from config import Config

from orchestrator.pipelines.verify_pipeline import VerificationPipeline

from orchestrator.pipelines.triangulation_orchestrator import TriangulationOrchestrator

from orchestrator.pipelines.brainstorm_orchestrator import BrainstormOrchestrator

from agents.adapters.groq_adapter import GroqAdapter
from agents.adapters.openrouter_adapter import OpenRouterAdapter
from agents.adapters.cohere_adapter import CohereAdapter

import os



# Load API keys from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not GROQ_API_KEY:
    print("⚠️ Warning: GROQ_API_KEY not set. Live API calls will fail.")

if not OPENROUTER_API_KEY:
    print("⚠️ Warning: OPENROUTER_API_KEY not set. Some live API calls will fail.")

if not COHERE_API_KEY:
    print("⚠️ Warning: COHERE_API_KEY not set. Some live API calls will fail.")



app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

conf = Config()



def create_mixed_agents():
    """Factory for mixed live agents with specialized roles."""
    agents = []

    # Agent roles for diversity
    agent_configs = [
        {
            "name": "brainstorm-agent",
            "provider": "groq",
            "model": "llama-3.1-8b-instant",
            "role": "brainstorm",
            "focus": "Generate multiple ideas, explore possibilities, think outside the box"
        },
        {
            "name": "creative-agent",
            "provider": "openrouter",
            "model": "anthropic/claude-3-haiku",
            "role": "creative",
            "focus": "Be imaginative, artistic, and innovative in your approach"
        },
        {
            "name": "professional-agent",
            "provider": "cohere",
            "model": "command-nightly",
            "role": "professional",
            "focus": "Be structured, logical, practical, and business-oriented"
        }
    ]

    for config in agent_configs:
        try:
            if config["provider"] == "groq" and GROQ_API_KEY:
                agent = GroqAdapter(
                    name=config["name"],
                    provider=config["provider"],
                    model=config["model"]
                )
                # Store the role info for later use
                agent.role = config["role"]
                agent.focus = config["focus"]
                agents.append(agent)

            elif config["provider"] == "openrouter" and OPENROUTER_API_KEY:
                agent = OpenRouterAdapter(
                    name=config["name"],
                    provider=config["provider"],
                    model=config["model"]
                )
                agent.role = config["role"]
                agent.focus = config["focus"]
                agents.append(agent)

            elif config["provider"] == "cohere" and COHERE_API_KEY:
                agent = CohereAdapter(
                    name=config["name"],
                    provider=config["provider"],
                    model=config["model"]
                )
                agent.role = config["role"]
                agent.focus = config["focus"]
                agents.append(agent)

        except Exception as e:
            print(f"Failed to create {config['provider']} agent: {e}")

    # If no agents were created, raise error
    if not agents:
        raise RuntimeError("No API keys available. Cannot create live agents.")

    return agents


class SpecializedAgent:
    """Wrapper for agents with role-specific prompt modification."""
    def __init__(self, base_agent, original_query):
        self.base_agent = base_agent
        self.original_query = original_query
        self.name = getattr(base_agent, 'name', 'unknown-agent')
        self.provider = getattr(base_agent, 'provider', 'unknown')
        self.model = getattr(base_agent, 'model', 'unknown')
        self.role = getattr(base_agent, 'role', 'general')
        self.focus = getattr(base_agent, 'focus', 'Provide helpful responses')

    def generate(self, prompt, **kwargs):
        """Generate response with role-specific context."""
        role_instruction = f"You are a {self.role} specialist. {self.focus}. "

        # Modify the prompt based on the agent's role
        if self.role == "brainstorm":
            specialized_prompt = f"{role_instruction}Brainstorm multiple creative ideas and possibilities. Think expansively and generate diverse options. Original query: {prompt}"
        elif self.role == "creative":
            specialized_prompt = f"{role_instruction}Approach this with creativity and imagination. Think artistically and innovatively. Original query: {prompt}"
        elif self.role == "professional":
            specialized_prompt = f"{role_instruction}Be structured, logical, and professional in your analysis. Focus on practical, business-oriented solutions. Original query: {prompt}"
        else:
            specialized_prompt = f"{role_instruction}Original query: {prompt}"

        return self.base_agent.generate(specialized_prompt, **kwargs)

    def __getattr__(self, name):
        """Delegate other attributes to the base agent."""
        return getattr(self.base_agent, name)


def create_groq_agent(name="groq-agent"):
    """Factory for a live Groq agent (backward compatibility)."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set. Cannot create live agent.")
    return GroqAdapter(name=name, provider="groq", model="llama-3.1-8b-instant")



@app.route("/")

def index():

    return render_template("index.html")



@app.route("/verify", methods=["POST"])

def verify():

    query = request.json.get("query")

    live = request.json.get("live", False)

    pipeline = VerificationPipeline(similarity_threshold=0.75)



    if live:

        # Use mixed real agents (Groq, OpenRouter, Cohere)

        agents = create_mixed_agents()

        # Ensure we have at least 3 agents, pad with duplicates if needed

        while len(agents) < 3:
            agents.append(agents[0])  # Duplicate first agent if needed

        agents = agents[:3]  # Take first 3 agents

        # Create specialized agents with role-specific prompts
        specialized_agents = [SpecializedAgent(agent, query) for agent in agents]

        report = pipeline.run(text=query, agent_instances=specialized_agents, dry_run=False)

    else:

        # Safe simulation

        agent_configs = conf.models[:3]

        report = pipeline.run(text=query, agent_configs=agent_configs, dry_run=True)



    return jsonify(report)



@app.route("/triangulate", methods=["POST"])

def triangulate():

    query = request.json.get("query")

    live = request.json.get("live", False)

    tri = TriangulationOrchestrator(max_agents=5, max_retries=2)



    if live:

        # Use mixed real agents (Groq, OpenRouter, Cohere)

        agents = create_mixed_agents()

        report = tri.run(text=query, agent_instances=agents, dry_run=False)

    else:

        agent_configs = conf.models[:5]

        report = tri.run(text=query, agent_configs=agent_configs, dry_run=True)



    return jsonify(report)



@app.route("/brainstorm", methods=["POST"])

def brainstorm():

    prompt = request.json.get("prompt")

    live = request.json.get("live", False)

    brainstormer = BrainstormOrchestrator(max_agents=5, dry_run=not live)



    if live:

        # Use mixed real agents (Groq, OpenRouter, Cohere)

        agents = create_mixed_agents()

        report = brainstormer.run(prompt=prompt, agent_instances=agents, run_verification=True)

    else:

        agent_configs = conf.models[:5]

        report = brainstormer.run(prompt=prompt, agent_configs=agent_configs, run_verification=False)



    return jsonify(report)



if __name__ == "__main__":

    app.run(debug=True)
