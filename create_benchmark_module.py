"""

SINGLE-STEP: Phase 1 Step 1 — Safe Benchmarking Creator (Cursor-safe)

-------------------------------------------------------------------



Purpose:

- Create a safe, non-invasive benchmarking module for "Performance Benchmarking & Optimization".

- This script is a *single atomic action* to be executed by Cursor (or run inside your repo).

- It WILL NOT modify existing project files other than adding a new file under `benchmarks/`.

- It WILL NOT install packages, change configs, or scaffold other phases.

- It performs safety checks (git cleanliness, config presence) and aborts if it finds risk.



What it produces:

- `benchmarks/agent_benchmark.py` (idempotent: will not overwrite if file exists unless --overwrite)

- `benchmarks/example_prompts.yaml` (only created if missing)

- A short README `benchmarks/README.md` describing how to run the benchmark manually



Behavior:

- Reads `config.yaml` to detect configured agents (supports minimal config shape used in our plan).

- Creates a benchmark runner that:

    * loads config via config.Config (if available) or reads config.yaml directly

    * for each agent entry, performs 3 warmup requests (if endpoint present) and N timed runs

    * measures latency, success/failure, basic output length, HTTP status (if adapter is HTTP-based)

    * stores results to `reports/benchmarks_<timestamp>.json`

    * DOES NOT call any live endpoints by default when created — the generated runner has a flag

      `DRY_RUN = True` to avoid accidental execution. User must edit to `DRY_RUN=False` to run.

- Leaves all other code untouched.



INSTRUCTIONS FOR CURSOR:

- Save this file into the repo and execute it (Cursor should run it as a single job).

- If `--overwrite` CLI arg is passed it will replace existing benchmark file; otherwise it aborts to avoid accidental overwrite.



Deliverable: single file creator script (this file). When run, it will create the benchmark module(s) but not run them.



---------------------------

USAGE (run in repository root):

    python create_benchmark_module.py [--overwrite]



After this creates the files, review `benchmarks/agent_benchmark.py`, set DRY_RUN=False

and ensure API keys / endpoints are correct before executing the benchmark runner.



"""

import os
import sys
import json
import yaml
import textwrap
from pathlib import Path
from datetime import datetime


ROOT = Path.cwd()
CONFIG_YAML = ROOT / "config.yaml"
BENCH_DIR = ROOT / "benchmarks"
REPORTS_DIR = ROOT / "reports"
BENCH_FILE = BENCH_DIR / "agent_benchmark.py"
EX_PROMPTS = BENCH_DIR / "example_prompts.yaml"
README = BENCH_DIR / "README.md"


# Safety checks

def abort(msg: str):
    print("ABORT:", msg)
    sys.exit(1)


# 1) Ensure repo is safe to modify (no destructive ops)

def ensure_safe_to_write():
    # Prevent accidental run inside unfamiliar directories
    if not ROOT.exists():
        abort(f"Working directory {ROOT} does not exist.")
    # Check config.yaml presence
    if not CONFIG_YAML.exists():
        abort("config.yaml not found at project root. Please ensure config.yaml exists before proceeding.")
    # Check git cleanliness if repo present
    git_dir = ROOT / ".git"
    if git_dir.exists():
        # try running git status --porcelain
        try:
            import subprocess

            res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
            if res.stdout.strip():
                abort("Git working tree is not clean. Commit or stash changes before proceeding to avoid accidental conflicts.")
        except Exception:
            # If git not accessible, be conservative
            abort("Git exists but could not be queried. Please ensure working tree is clean.")
    # All good
    return True


# 2) Create target directories (non-destructive)

def mkdirs():
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# 3) Read config.yaml minimally to extract agents list (non-fatal fallback)

def read_agents_from_config():
    try:
        with open(CONFIG_YAML, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        abort(f"Failed to read config.yaml: {e}")

    agents = cfg.get("agents") or []
    # Normalize agent entries to minimal expected shape
    normalized = []
    for a in agents:
        if isinstance(a, dict):
            normalized.append(
                {
                    "name": a.get("name", a.get("model", "unnamed-agent")),
                    "provider": a.get("provider", "unknown"),
                    "model": a.get("model") or a.get("endpoint"),
                    "role": a.get("role", "agent"),
                    "timeout": a.get("timeout", 15),
                }
            )
    return normalized


# 4) Write example_prompts.yaml if missing

EX_PROMPTS_CONTENT = textwrap.dedent(
    """
    # example_prompts.yaml
    # Minimal examples to drive benchmarking.
    # You may extend prompts per-agent role.
    verify_prompt: |
      Please list the top 3 factual claims in the following text and rate their confidence.

      Text: "The Eiffel Tower is in Paris and was completed in 1889."

    brainstorm_prompt: |
      Provide 6 diverse, short ideas (one per line) for addressing climate-related misinformation.
    """
).strip()


# 5) Benchmark runner template (idempotent, DRY_RUN=True by default)

BENCHMARK_TEMPLATE = textwrap.dedent(
    '''\
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
    AGENTS = {AGENTS_JSON}

    # Example prompts (you can extend or replace)
    PROMPTS = {{
        "verify": """{VERIFY_PROMPT}""",
        "brainstorm": """{BRAINSTORM_PROMPT}""",
    }}

    # Benchmark parameters
    WARMUP_RUNS = 1
    MEASURE_RUNS = 3
    TIMEOUT_DEFAULT = 15

    def safe_post(url, json_payload, headers=None, timeout=TIMEOUT_DEFAULT):
        if DRY_RUN:
            # Simulate network call result for safety.
            return {{
                "simulated": True,
                "status_code": None,
                "text": "<DRY_RUN simulated response>",
                "elapsed_ms": 0
            }}
        try:
            start = time.time()
            resp = requests.post(url, json=json_payload, headers=headers or {{}})
            elapsed = (time.time() - start) * 1000.0
            return {{
                "simulated": False,
                "status_code": resp.status_code,
                "text": resp.text[:2000],
                "elapsed_ms": elapsed
            }}
        except Exception as e:
            return {{
                "simulated": False,
                "error": str(e)
            }}

    def run_benchmark_for_agent(agent):
        """
        agent: dict with keys name, provider, model (may be HF model name or full endpoint),
               role, timeout.
        For HTTP endpoints we will try to prepare a minimal request body; for provider names without
        endpoints, we will only record metadata (no network calls) unless user populates agent['endpoint'].
        """
        result = {{
            "agent": agent,
            "runs": []
        }}
        endpoint = agent.get("endpoint") or agent.get("url") or None
        # If agent describes a huggingface model (model string), attempt HF inference endpoint pattern
        if endpoint is None and isinstance(agent.get("model"), str) and "/" in agent.get("model"):
            endpoint = f"https://api-inference.huggingface.co/models/{{agent['model']}}"
        # Choose prompt based on role
        prompt = PROMPTS.get("verify") if "verif" in agent.get("role","").lower() else PROMPTS.get("brainstorm")
        payload = {{"inputs": prompt}}
        headers = {{}}
        # If DRY_RUN, we won't use headers
        # Warmup
        for i in range(WARMUP_RUNS):
            run_res = safe_post(endpoint, payload, headers=headers, timeout=agent.get("timeout", TIMEOUT_DEFAULT)) if endpoint else {{"simulated": True, "note": "no_endpoint"}}
            result["runs"].append({{"type":"warmup","result": run_res}})
        # Measured runs
        for i in range(MEASURE_RUNS):
            run_res = safe_post(endpoint, payload, headers=headers, timeout=agent.get("timeout", TIMEOUT_DEFAULT)) if endpoint else {{"simulated": True, "note": "no_endpoint"}}
            result["runs"].append({{"type":"measure","result": run_res}})
        # Basic aggregated stats
        elapsed = [r["result"].get("elapsed_ms") for r in result["runs"] if isinstance(r["result"].get("elapsed_ms"), (int, float))]
        result["stats"] = {{
            "measured_runs": len(elapsed),
            "avg_latency_ms": sum(elapsed)/len(elapsed) if elapsed else None,
            "min_latency_ms": min(elapsed) if elapsed else None,
            "max_latency_ms": max(elapsed) if elapsed else None
        }}
        return result

    def run_all():
        report = {{
            "created_at": datetime.utcnow().isoformat() + "Z",
            "dry_run": DRY_RUN,
            "agents_count": len(AGENTS),
            "agent_reports": []
        }}
        for a in AGENTS:
            r = run_benchmark_for_agent(a)
            report["agent_reports"].append(r)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_path = REPORTS_DIR / f"benchmarks_{{ts}}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("Benchmark report written to:", out_path)
        return out_path

    if __name__ == "__main__":
        print("Agent benchmark runner (safe). DRY_RUN =", DRY_RUN)
        print("Agents detected:", [a.get("name") for a in AGENTS])
        print("To run real network benchmarks: edit this file, set DRY_RUN=False and ensure endpoints/keys are set.")
    '''
)


def create_benchmark_module(agents, overwrite=False):
    # Safety: do not overwrite unless asked
    if BENCH_FILE.exists() and not overwrite:
        print(f"{BENCH_FILE} already exists. Use --overwrite to replace it. Aborting creation.")
        return False

    # Fill template fields
    agents_json = json.dumps(agents, indent=4)
    # Load example prompts
    try:
        with open(EX_PROMPTS, "r", encoding="utf-8") as f:
            ex_prompts = f.read()
    except FileNotFoundError:
        ex_prompts = EX_PROMPTS_CONTENT

    # Attempt to parse the example prompts YAML and extract verify/brainstorm prompts
    try:
        pmap = yaml.safe_load(ex_prompts) or {}
        verify_prompt = pmap.get("verify_prompt", "Summarize the text and list factual claims.")
        brainstorm_prompt = pmap.get("brainstorm_prompt", "Generate 6 short diverse ideas.")
    except Exception:
        verify_prompt = "Summarize the text and list factual claims."
        brainstorm_prompt = "Generate 6 short diverse ideas."

    filled = BENCHMARK_TEMPLATE.format(
        AGENTS_JSON=agents_json.replace("'", '"'),
        VERIFY_PROMPT=verify_prompt.replace('"""', '\\"""'),
        BRAINSTORM_PROMPT=brainstorm_prompt.replace('"""', '\\"""'),
    )

    # Write files
    BENCH_FILE.write_text(filled, encoding="utf-8")
    if not EX_PROMPTS.exists():
        EX_PROMPTS.write_text(EX_PROMPTS_CONTENT, encoding="utf-8")
    if not README.exists():
        README.write_text(
            textwrap.dedent(
                """\
            Benchmarks directory
            --------------------

            This folder contains an auto-generated safe benchmark runner:

            - agent_benchmark.py : safe runner. Edit DRY_RUN=False to actually run network calls.
            - example_prompts.yaml : prompts used by the runner.

            CAUTION: The generated runner defaults to DRY_RUN=True to avoid accidental live calls.
            Review agent endpoints and credentials before executing.
            """
            ),
            encoding="utf-8",
        )

    print(f"Created {BENCH_FILE} (overwrite={overwrite}) and supporting files.")
    return True


# CLI handling

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create safe benchmark module for multi_ai_system (single-step).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing benchmark module if present.")
    args = parser.parse_args()

    ensure_safe_to_write()
    mkdirs()
    agents = read_agents_from_config()
    # If no agents found, still create a template with an example agent metadata
    if not agents:
        agents = [
            {"name": "example-hf-agent", "provider": "huggingface", "model": "tiiuae/falcon-7b-instruct", "role": "creative", "timeout": 15}
        ]
        print("No agents found in config.yaml — creating benchmark template with an example agent. Edit the generated file to add real endpoints.")
    created = create_benchmark_module(agents, overwrite=args.overwrite)
    if created:
        print("Done. Inspect benchmarks/agent_benchmark.py and example_prompts.yaml. Edit DRY_RUN to run real benchmarks.")


if __name__ == "__main__":
    main()
