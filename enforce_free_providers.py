"""
enforce_free_providers.py

Purpose:
- Step 1 of "Option 3": ensure your config uses only free/allowed remote AI providers.
- Safe-by-default: DRY_RUN=True will only report what *would* change.
- When ready, pass --apply to write a new `config_free.yaml` (original left untouched).
- Pass --force to overwrite `config.yaml` (creates a backup `config.yaml.bak` first).

Usage (copy/paste & run from repo root):
    python enforce_free_providers.py            # dry run: only reports
    python enforce_free_providers.py --apply    # writes config_free.yaml with replacements
    python enforce_free_providers.py --apply --force  # overwrite config.yaml (backed up)

Notes:
- This script does NOT make network calls.
- It uses only stdlib + pyyaml (yaml). Install with `pip install pyyaml` if needed.
- Edit ALLOWED_PROVIDERS and RECOMMENDATIONS to tune which providers/models are considered "free" for your project.
"""

import argparse
import copy
import os
import sys
from pathlib import Path

try:
    import yaml
except Exception:
    print("Missing dependency 'pyyaml'. Install it with: pip install pyyaml")
    sys.exit(1)

ROOT = Path.cwd()
CONFIG_PATH = ROOT / "config.yaml"
OUTPUT_PATH = ROOT / "config_free.yaml"
BACKUP_PATH = ROOT / "config.yaml.bak"

# ---- Configure which providers we treat as "free / allowed" ----

# Adjust as you learn which providers you prefer.
ALLOWED_PROVIDERS = {
    "huggingface",   # HF Inference free endpoints (rate-limited, often works without key)
    "groq",          # Groq free tier / community endpoints
    "together",      # Together AI free credits / community models
    "openrouter",    # OpenRouter community endpoints (some free)
    "replicate",     # Replicate has free community models (may require token)
    "hf",            # sometimes provider is written 'hf'
    "generic",       # http generic adapter for public endpoints
    "web_search",    # search agent (DuckDuckGo / Wikipedia) - not LLM
}

# ---- Recommendations mapping for replacing non-free providers ----

# key = provider string you might find in config; value = dict with recommended provider/model
# Edit these suggestions to match your preferred free models.
RECOMMENDATIONS = {
    "openai": {"provider": "huggingface", "model": "tiiuae/falcon-40b-instruct"},  # example
    "gpt-4": {"provider": "together", "model": "mixtral-8x7b"},  # fallback suggestion
    "gpt-3.5": {"provider": "huggingface", "model": "bigscience/bloomz-7b1"},  # example
    "anthropic": {"provider": "huggingface", "model": "mistral-7b"},  # example
    "local": {"provider": "huggingface", "model": "tiiuae/falcon-7b-instruct"},
    # Add other common non-free names you expect
}

# ---- Helper functions ----

def load_config(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Create config.yaml first.")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}

def save_config(cfg: dict, path: Path):
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

def normalize_provider(p: str):
    if not p:
        return ""
    return p.strip().lower()

def analyze_agents(cfg: dict):
    agents = cfg.get("agents", [])
    if not isinstance(agents, list):
        return [], [{"error": "config.yaml 'agents' is not a list"}]
    issues = []
    for idx, a in enumerate(agents):
        if not isinstance(a, dict):
            issues.append({"index": idx, "agent": a, "issue": "agent entry not a dict"})
            continue
        provider = normalize_provider(a.get("provider") or a.get("type") or "")
        model = a.get("model") or a.get("endpoint") or ""
        allowed = provider in ALLOWED_PROVIDERS
        issues.append(
            {
                "index": idx,
                "name": a.get("name"),
                "provider": provider,
                "model": model,
                "allowed": allowed,
                "raw": a,
            }
        )
    return agents, issues

def suggest_replacement(provider: str, model: str):
    key = provider.lower()
    if key in RECOMMENDATIONS:
        return RECOMMENDATIONS[key]
    # If provider not in recommendations, attempt simple heuristic:
    if "openai" in key or "gpt" in key:
        return RECOMMENDATIONS.get("openai")
    return None

def build_free_config(orig_cfg: dict, issues: list):
    cfg = copy.deepcopy(orig_cfg)
    agents = cfg.get("agents", [])
    for it in issues:
        if not it.get("allowed"):
            idx = it["index"]
            orig = agents[idx]
            provider = normalize_provider(orig.get("provider") or orig.get("type") or "")
            suggestion = suggest_replacement(provider, orig.get("model", ""))
            if suggestion:
                # apply suggestion
                agents[idx]["provider"] = suggestion["provider"]
                if suggestion.get("model"):
                    agents[idx]["model"] = suggestion["model"]
                # optionally remove keys that look like paid config (api_key)
                agents[idx].pop("api_key", None)
                agents[idx].pop("key", None)
            else:
                # mark as generic (safe fallback)
                agents[idx]["provider"] = "generic"
                agents[idx].pop("api_key", None)
    cfg["agents"] = agents
    return cfg

# ---- Main CLI ----

def main():
    parser = argparse.ArgumentParser(description="Enforce free-only AI providers in config.yaml (safe-by-default).")
    parser.add_argument("--apply", action="store_true", help="Write suggested free-only config to config_free.yaml")
    parser.add_argument("--force", action="store_true", help="If --apply and --force, overwrite config.yaml after backup")
    parser.add_argument("--show-allowed", action="store_true", help="Print allowed providers and exit")
    args = parser.parse_args()

    if args.show_allowed:
        print("Allowed (free) providers:")
        for p in sorted(ALLOWED_PROVIDERS):
            print(" -", p)
        return

    try:
        cfg = load_config(CONFIG_PATH)
    except Exception as e:
        print("ERROR loading config:", e)
        sys.exit(2)

    agents, issues = analyze_agents(cfg)
    if not agents:
        print("No agents found in config.yaml (or agents list empty). Nothing to do.")
        # Still allow writing an empty config_free.yaml if --apply
        if args.apply:
            save_config(cfg, OUTPUT_PATH)
            print("Wrote empty config_free.yaml (no agents).")
        return

    # Report
    print(f"Found {len(agents)} agent(s) in config.yaml\n")

    non_allowed = [i for i in issues if not i.get("allowed")]
    if not non_allowed:
        print("All agents already use allowed free providers. No action required.")
        if args.apply:
            save_config(cfg, OUTPUT_PATH)
            print("Wrote config_free.yaml identical to config.yaml.")
        return

    print("Agents using non-allowed (non-free) providers:")
    for it in non_allowed:
        print(f" - index {it['index']}, name={it.get('name')}, provider={it.get('provider')}, model={it.get('model')}")
        suggestion = suggest_replacement(it.get("provider",""), it.get("model",""))
        if suggestion:
            print(f"    -> Suggested replacement: provider={suggestion['provider']}, model={suggestion.get('model')}")
        else:
            print("    -> No automated suggestion available. Will fallback to 'generic' if applied.")

    if not args.apply:
        print("\nDRY RUN complete. Re-run with --apply to create config_free.yaml with suggested changes.")
        return

    # Build and write new config
    new_cfg = build_free_config(cfg, issues)
    save_config(new_cfg, OUTPUT_PATH)
    print(f"\nWrote suggested free-only config to {OUTPUT_PATH}")

    if args.force:
        # backup existing config.yaml first
        if CONFIG_PATH.exists():
            print(f"Backing up existing config.yaml to {BACKUP_PATH}")
            CONFIG_PATH.replace(BACKUP_PATH)
        save_config(new_cfg, CONFIG_PATH)
        print("Overwrote config.yaml with free-only configuration.")

if __name__ == "__main__":
    main()
