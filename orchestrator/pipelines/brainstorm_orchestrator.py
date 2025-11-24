
"""Brainstorm Orchestrator

Aggregates ideas from multiple agents, integrates verification optionally,
and provides structured output.
"""

import logging
from typing import List, Dict, Any, Optional

try:
    from agents.factory import create_agent
except Exception:
    create_agent = None

try:
    from orchestrator.pipelines.triangulation_orchestrator import TriangulationOrchestrator
except Exception:
    TriangulationOrchestrator = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BrainstormOrchestrator:
    def __init__(self, max_agents: int = 5, max_retries: int = 2, dry_run: bool = True):
        self.max_agents = max_agents
        self.max_retries = max_retries
        self.dry_run = dry_run
        self.triangulator = TriangulationOrchestrator() if TriangulationOrchestrator else None

    def _instantiate_agents(self, agent_configs: List[Dict[str, Any]]):
        agents = []
        if not agent_configs:
            return agents
        if create_agent is None:
            logger.warning('agents.factory.create_agent not importable; using placeholders.')
            return [None for _ in agent_configs[:self.max_agents]]

        for cfg in agent_configs[:self.max_agents]:
            try:
                # Handle ModelConfig objects by converting to dict
                if hasattr(cfg, '__dict__'):
                    # Convert ModelConfig to dict format expected by factory
                    config_dict = {
                        'name': getattr(cfg, 'name', 'unknown'),
                        'provider': getattr(cfg, 'model_name', 'generic').split('-')[0] if hasattr(cfg, 'model_name') else 'generic',
                        'model': getattr(cfg, 'model_name', 'unknown-model'),
                        'role': getattr(cfg, 'role', 'agent'),
                        'timeout': 15
                    }
                    agents.append(create_agent(config_dict))
                else:
                    # Assume it's already a dict
                    agents.append(create_agent(cfg))
            except Exception as e:
                logger.exception('Failed to create agent: %s', e)
                agents.append(None)
        return agents

    def run(self, prompt: str, agent_configs: Optional[List[Dict[str, Any]]] = None,
            agent_instances: Optional[List[Any]] = None, run_verification: bool = False) -> Dict[str, Any]:
        """
        Run multi-agent brainstorming session.
        If run_verification=True, each idea is verified with TriangulationOrchestrator.
        """
        agents = []
        if agent_instances:
            agents = agent_instances[:self.max_agents]
        elif agent_configs:
            agents = self._instantiate_agents(agent_configs)
        else:
            logger.warning('No agents provided; using placeholders.')
            return {'error': 'no_agents', 'ideas': [], 'dry_run': True}

        # Ensure at least 3 agents
        while len(agents) < 3:
            agents.append(None)

        ideas = []
        for idx, agent in enumerate(agents):
            retry_count = 0
            agent_idea = None
            while retry_count <= self.max_retries:
                if self.dry_run:
                    agent_idea = f'[SIMULATED IDEA {idx}] based on prompt: {prompt}'
                else:
                    try:
                        agent_idea = agent.generate(prompt) if agent else f'[SIMULATED IDEA {idx}]'
                    except Exception as e:
                        logger.exception('Agent failed: %s', e)
                        agent_idea = f'[SIMULATED IDEA {idx}]'

                if agent_idea or retry_count == self.max_retries:
                    break
                retry_count += 1
            ideas.append({'agent_index': idx, 'idea': agent_idea, 'retries': retry_count})

        # Optional verification
        if run_verification and self.triangulator:
            for idea in ideas:
                report = self.triangulator.run(text=idea['idea'], dry_run=self.dry_run)
                idea['verification'] = report.get('consensus', {}) if report else {}

        final_report = {
            'prompt': prompt,
            'agents_count': len(agents),
            'ideas': ideas,
            'dry_run': self.dry_run
        }

        return final_report
