from flask import Flask, render_template, request, jsonify
from orchestrator.pipelines.triangulation_orchestrator import TriangulationOrchestrator
from orchestrator.pipelines.brainstorm_orchestrator import BrainstormOrchestrator
from orchestrator.pipelines.verify_pipeline import VerificationPipeline
from config import Config

app = Flask(__name__)
conf = Config()

# Initialize orchestrators with dry_run=True for safety
tri_orch = TriangulationOrchestrator()
brain_orch = BrainstormOrchestrator(dry_run=True)
verify_pipeline = VerificationPipeline()

@app.route("/")
def index():
    return render_template("index.html")  # basic HTML page, create templates/index.html

@app.route("/verify", methods=["POST"])
def verify():
    query = request.json.get("query", "")
    report = verify_pipeline.run(text=query, agent_configs=conf.models[:3])
    return jsonify(report)

@app.route("/triangulate", methods=["POST"])
def triangulate():
    query = request.json.get("query", "")
    report = tri_orch.run(text=query, agent_configs=conf.models[:5])
    return jsonify(report)

@app.route("/brainstorm", methods=["POST"])
def brainstorm():
    prompt = request.json.get("prompt", "")
    report = brain_orch.run(prompt=prompt, agent_configs=conf.models[:5], run_verification=False)
    return jsonify(report)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
