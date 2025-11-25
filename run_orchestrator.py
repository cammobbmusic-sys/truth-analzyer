import argparse
import json
from config import Config
from orchestrator.pipelines.verify_pipeline import VerificationPipeline
from orchestrator.pipelines.triangulation_orchestrator import TriangulationOrchestrator
from orchestrator.pipelines.brainstorm_orchestrator import BrainstormOrchestrator
from orchestrator.pipelines.debate_orchestrator import DebateOrchestrator

def main(
    mode="verify",  # "verify", "brainstorm", "triangulate", or "debate"
    prompt_or_text="",
    max_agents=5,
    dry_run=True,
    run_verification=False,
    absolute_truth_mode=False,
    debate_rounds=3
):
    conf = Config()
    agent_configs = conf.models[:max_agents]

    report = None

    if mode == "verify":
        pipeline = VerificationPipeline(absolute_truth_mode=absolute_truth_mode)
        report = pipeline.run(
            text=prompt_or_text,
            agent_configs=agent_configs,
            dry_run=dry_run,
            absolute_truth_mode=absolute_truth_mode
        )

    elif mode == "triangulate":
        tri = TriangulationOrchestrator(max_agents=max_agents)
        report = tri.run(
            text=prompt_or_text,
            agent_configs=agent_configs,
            dry_run=dry_run
        )

    elif mode == "brainstorm":
        brainstorm = BrainstormOrchestrator(max_agents=max_agents, dry_run=dry_run)
        report = brainstorm.run(
            prompt=prompt_or_text,
            agent_configs=agent_configs,
            run_verification=run_verification
        )

    elif mode == "debate":
        debate = DebateOrchestrator(max_rounds=debate_rounds, absolute_truth_mode=absolute_truth_mode)
        report = debate.run(
            topic=prompt_or_text,
            agent_configs=agent_configs,
            dry_run=dry_run,
            absolute_truth_mode=absolute_truth_mode
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Multi-Agent Orchestrator Runner")
    parser.add_argument("--mode", type=str, default="verify", help="verify | triangulate | brainstorm | debate")
    parser.add_argument("--input", type=str, default="", help="Text or prompt for the workflow")
    parser.add_argument("--max_agents", type=int, default=5)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--run_verification", action="store_true", help="Enable verification for brainstorming")
    parser.add_argument("--absolute_truth", action="store_true", help="Enable absolute truth mode (requires unanimous agreement with 98% confidence)")
    parser.add_argument("--debate_rounds", type=int, default=3, help="Number of debate rounds (for debate mode)")
    args = parser.parse_args()

    main(
        mode=args.mode,
        prompt_or_text=args.input,
        max_agents=args.max_agents,
        dry_run=args.dry_run,
        run_verification=args.run_verification,
        absolute_truth_mode=args.absolute_truth,
        debate_rounds=args.debate_rounds
    )
