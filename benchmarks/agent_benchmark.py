"""
Agent Benchmark Runner (auto-generated)

- Safe runner: DRY_RUN=True prevents live network calls by default.

- Edit DRY_RUN=False only after you confirm credentials and endpoints.

- Outputs results into reports/benchmarks_<timestamp>.json
"""

import time
import json
import os
import requests
from pathlib import Path
from datetime import datetime

ROOT = Path.cwd()
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# CAUTION: Prevent accidental live calls
DRY_RUN = True

# Configuration: override via environment or edit after review
AGENTS = [
    {
        "name": "gemini-3-pro",
        "provider": "google",
        "model": "gemini-3-pro",
        "role": "expert",
        "timeout": 15
    },
    {
        "name": "gpt-5.1-codex",
        "provider": "openai",
        "model": "gpt-5.1-codex",
        "role": "creative",
        "timeout": 15
    },
    {
        "name": "sonnet-4.5",
        "provider": "anthropic",
        "model": "sonnet-4.5",
        "role": "analyst",
        "timeout": 15
    }
]

# Example prompts (you can extend or replace)
PROMPTS = {
    "verify": """Please list the top 3 factual claims in the following text and rate their confidence.

Text: "The Eiffel Tower is in Paris and was completed in 1889."
""",
    "brainstorm": """Provide 6 diverse, short ideas (one per line) for addressing climate-related misinformation.""",
}

# Benchmark parameters
WARMUP_RUNS = 1
MEASURE_RUNS = 3
TIMEOUT_DEFAULT = 15

def safe_post(url, json_payload, headers=None, timeout=TIMEOUT_DEFAULT):
    if DRY_RUN:
        # Simulate network call result for safety.
        return {
            "simulated": True,
            "status_code": None,
            "text": "<DRY_RUN simulated response>",
            "elapsed_ms": 0
        }
    try:
        start = time.time()
        resp = requests.post(url, json=json_payload, headers=headers or {})
        elapsed = (time.time() - start) * 1000.0
        return {
            "simulated": False,
            "status_code": resp.status_code,
            "text": resp.text[:2000],
            "elapsed_ms": elapsed
        }
    except Exception as e:
        return {
            "simulated": False,
            "error": str(e)
        }

def run_benchmark_for_agent(agent):
    """
    agent: dict with keys name, provider, model (may be HF model name or full endpoint),
           role, timeout.
    For HTTP endpoints we will try to prepare a minimal request body; for provider names without
    endpoints, we will only record metadata (no network calls) unless user populates agent['endpoint'].
    """
    result = {
        "agent": agent,
        "runs": []
    }
    endpoint = agent.get("endpoint") or agent.get("url") or None
    # If agent describes a huggingface model (model string), attempt HF inference endpoint pattern
    if endpoint is None and isinstance(agent.get("model"), str) and "/" in agent.get("model"):
        endpoint = f"https://api-inference.huggingface.co/models/{agent['model']}"
    # Choose prompt based on role
    prompt = PROMPTS.get("verify") if "verif" in agent.get("role","").lower() else PROMPTS.get("brainstorm")
    payload = {"inputs": prompt}
    headers = {}
    # If DRY_RUN, we won't use headers
    # Warmup
    for i in range(WARMUP_RUNS):
        run_res = safe_post(endpoint, payload, headers=headers, timeout=agent.get("timeout", TIMEOUT_DEFAULT)) if endpoint else {"simulated": True, "note": "no_endpoint"}
        result["runs"].append({"type":"warmup","result": run_res})
    # Measured runs
    for i in range(MEASURE_RUNS):
        run_res = safe_post(endpoint, payload, headers=headers, timeout=agent.get("timeout", TIMEOUT_DEFAULT)) if endpoint else {"simulated": True, "note": "no_endpoint"}
        result["runs"].append({"type":"measure","result": run_res})
    # Basic aggregated stats
    elapsed = [r["result"].get("elapsed_ms") for r in result["runs"] if isinstance(r["result"].get("elapsed_ms"), (int, float))]
    result["stats"] = {
        "measured_runs": len(elapsed),
        "avg_latency_ms": sum(elapsed)/len(elapsed) if elapsed else None,
        "min_latency_ms": min(elapsed) if elapsed else None,
        "max_latency_ms": max(elapsed) if elapsed else None
    }
    return result

def run_all():
    report = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "dry_run": DRY_RUN,
        "agents_count": len(AGENTS),
        "agent_reports": []
    }
    for a in AGENTS:
        r = run_benchmark_for_agent(a)
        report["agent_reports"].append(r)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = REPORTS_DIR / f"benchmarks_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Benchmark report written to:", out_path)
    return out_path

if __name__ == "__main__":
    print("Agent benchmark runner (safe). DRY_RUN =", DRY_RUN)
    print("Agents detected:", [a.get("name") for a in AGENTS])
    print("To run real network benchmarks: edit this file, set DRY_RUN=False and ensure endpoints/keys are set.")
