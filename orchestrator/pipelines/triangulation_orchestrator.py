
"""Triangulation Orchestrator

Coordinates multiple agents (3–9), retries failing agents,
and computes final consensus for truth verification.
"""

import logging
from typing import List, Dict, Any, Optional

try:
    from orchestrator.pipelines.verify_pipeline import VerificationPipeline
except Exception:
    VerificationPipeline = None

try:
    from agents.factory import create_agent
except Exception:
    create_agent = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TriangulationOrchestrator:
    def __init__(self, similarity_threshold: float = 0.75, max_agents: int = 5, max_retries: int = 2):
        self.similarity_threshold = similarity_threshold
        self.max_agents = max_agents
        self.max_retries = max_retries
        self.verifier = VerificationPipeline(similarity_threshold=similarity_threshold) if VerificationPipeline else None

    def _instantiate_agents(self, agent_configs: List[Dict[str, Any]]):
        agents = []
        if not agent_configs:
            return agents
        if create_agent is None:
            logger.warning('agents.factory.create_agent not importable; using None placeholders.')
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

    def run(self, text: str, agent_configs: Optional[List[Dict[str, Any]]] = None,
            agent_instances: Optional[List[Any]] = None, dry_run: bool = True) -> Dict[str, Any]:
        """
        Run triangulation verification.
        """
        agents = []
        if agent_instances:
            agents = agent_instances[:self.max_agents]
        elif agent_configs:
            agents = self._instantiate_agents(agent_configs)
        else:
            logger.warning('No agents provided; using placeholders.')
            return {'error': 'no_agents', 'report': None, 'dry_run': True}

        # Ensure at least 3 agents
        while len(agents) < 3:
            agents.append(None)

        reports = []
        for idx, agent in enumerate(agents):
            retry_count = 0
            agent_report = None
            while retry_count <= self.max_retries:
                if self.verifier:
                    agent_report = self.verifier.run(
                        text=text,
                        agent_instances=[agent],
                        dry_run=dry_run
                    )
                else:
                    agent_report = {'text': '[SIMULATED]', 'error': None, 'meta': {'simulated': True}}
                confidence = agent_report.get('consensus', {}).get('confidence', 0)
                if confidence >= self.similarity_threshold or retry_count == self.max_retries:
                    break
                retry_count += 1
            agent_report['retries'] = retry_count
            reports.append(agent_report)

        # Compute aggregated consensus across all agents
        final_texts = [r.get('agents', [{}])[0].get('output', '[SIMULATED]') for r in reports]
        if self.verifier:
            pairwise_matrix = self.verifier._pairwise_similarity_matrix(final_texts)
            consensus = self.verifier._consensus_from_matrix(pairwise_matrix, self.similarity_threshold)
        else:
            pairwise_matrix = []
            consensus = {'verdict': 'simulated', 'supporting_pairs': [], 'confidence': 1.0}

        final_report = {
            'input': text,
            'agents_count': len(agents),
            'agent_reports': reports,
            'pairwise_similarity': pairwise_matrix,
            'consensus': consensus,
            'verdict': consensus.get('verdict', 'unknown'),
            'dry_run': dry_run
        }

        return final_report
