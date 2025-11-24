from flask import Flask, render_template, request, jsonify

from config import Config

from orchestrator.pipelines.verify_pipeline import VerificationPipeline

from orchestrator.pipelines.triangulation_orchestrator import TriangulationOrchestrator

from orchestrator.pipelines.brainstorm_orchestrator import BrainstormOrchestrator

from agents.adapters.groq_adapter import GroqAdapter

import os



# Load your Groq API key from environment

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:

    print("⚠️ Warning: GROQ_API_KEY not set. Live API calls will fail.")



app = Flask(__name__)

conf = Config()



def create_groq_agent(name="groq-agent"):

    """Factory for a live Groq agent."""

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

        # Use real Groq agent

        agents = [create_groq_agent() for _ in range(3)]

        report = pipeline.run(text=query, agent_instances=agents, dry_run=False)

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

        agents = [create_groq_agent() for _ in range(3)]

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

        agents = [create_groq_agent() for _ in range(3)]

        report = brainstormer.run(prompt=prompt, agent_instances=agents, run_verification=True)

    else:

        agent_configs = conf.models[:5]

        report = brainstormer.run(prompt=prompt, agent_configs=agent_configs, run_verification=False)



    return jsonify(report)



if __name__ == "__main__":

    app.run(debug=True)
