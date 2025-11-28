
"""
verify_pipeline.py

3-Agent Verification Pipeline

This module provides VerificationPipeline, which runs three agents (instances or configs),
compares their outputs with semantic embeddings, and produces a consensus result.

Key features:
- Safe-by-default: dry_run=True will not force live agent network calls.
- Accepts either:
    * agent_instances: list of instantiated agent objects with .generate(prompt, **kwargs) -> str
    * OR agent_configs: list of dicts (will be converted via agents.factory.create_agent)
- Uses utils.embeddings.embed_texts for semantic similarity (requires sentence-transformers)
- Outputs a JSON-serializable report containing:
    - agent outputs
    - pairwise similarity matrix
    - consensus cluster and verdict
    - per-agent confidence estimates

Example usage (in repo root / interactive):
    from orchestrator.pipelines.verify_pipeline import VerificationPipeline
    from config import conf
    from agents.factory import create_agent

    cfg = conf.get('agents') or []
    # pick 3 agents (or prepare 3 configs)
    agent_cfgs = cfg[:3]
    pipeline = VerificationPipeline()
    report = pipeline.run(text='The Eiffel Tower is in Paris and was completed in 1889.',
                          agent_configs=agent_cfgs,
                          dry_run=True)
    print(report)

Notes:
- To actually call remote providers, set dry_run=False and ensure agents (and their API keys) are configured.
- This implementation is intentionally conservative about live calls; use logging to inspect runtime.
"""

import json
import logging
from typing import List, Dict, Optional, Any

# Attempt to import project utilities; fall back to safe stubs if missing.
try:
    from agents.factory import create_agent
except Exception:
    create_agent = None  # type: ignore

try:
    from utils.embeddings import embed_texts, cosine_similarity
except Exception:
    # Minimal fallback implementation using locally available numpy if embedder not present.
    embed_texts = None
    def cosine_similarity(a, b):
        try:
            import numpy as _np
            a = _np.array(a, dtype=float)
            b = _np.array(b, dtype=float)
            if _np.linalg.norm(a) == 0 or _np.linalg.norm(b) == 0:
                return 0.0
            return float(_np.dot(a, b) / (_np.linalg.norm(a) * _np.linalg.norm(b)))
        except Exception:
            # fallback crude similarity
            return 1.0 if a == b else 0.0

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class VerificationPipeline:
    def __init__(self, similarity_threshold: float = 0.75, absolute_truth_mode: bool = False):
        """
        similarity_threshold: cosine similarity threshold to consider two agent outputs as 'agreeing'
        absolute_truth_mode: if True, requires unanimous agreement with very high confidence for truth claims
        """
        self.similarity_threshold = float(similarity_threshold)
        self.absolute_truth_mode = absolute_truth_mode

        # In absolute truth mode, use much stricter thresholds
        if absolute_truth_mode:
            self.similarity_threshold = 0.95  # Require near-identical responses

    def _instantiate_agents(self, agent_configs: List[Dict[str, Any]]):
        """
        Given a list of agent config dicts, instantiate agent objects using agents.factory.create_agent
        If create_agent is not available, returns None for each entry to avoid accidental calls.
        Handles both dict configs and ModelConfig objects.
        """
        agents = []
        if not agent_configs:
            return agents
        if create_agent is None:
            logger.warning('agents.factory.create_agent is not importable in this environment; returning simulated agents.')
            return [None for _ in agent_configs]

        for cfg in agent_configs:
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
                    a = create_agent(config_dict)
                else:
                    # Assume it's already a dict
                    a = create_agent(cfg)
                agents.append(a)
            except Exception as e:
                logger.exception('Failed to create agent from config: %s. Error: %s', cfg, e)
                agents.append(None)
        return agents

    def _run_agent(self, agent, prompt: str, timeout: Optional[int] = None, dry_run: bool = True) -> Dict[str, Any]:
        """
        Safely run a single agent.
        Returns a dict: {'text': str, 'error': str|None, 'meta': {...}}
        If dry_run=True and agent is callable, we still call generate() for adapters that simulate responses;
        If agent is None or dry_run=True and we want to avoid network, we return a simulated response.
        """
        if agent is None:
            # simulated fallback
            return {'text': f'[SIMULATED] no agent instantiated', 'error': None, 'meta': {'simulated': True}}

        # Prefer conservative behavior: if dry_run and agent likely makes network calls, simulate
        if dry_run:
            # If adapter exposes attribute 'simulate' or 'dry_run_safe', we could call it.
            try:
                simulated = getattr(agent, 'SIMULATED', None) or getattr(agent, 'DRY_RUN_SAFE', None)
            except Exception:
                simulated = None
            if simulated:
                try:
                    out = agent.generate(prompt, temperature=0.0, max_tokens=256)
                    return {'text': out, 'error': None, 'meta': {'simulated': True}}
                except Exception as e:
                    return {'text': f'[SIMULATED-FALLBACK] {str(e)}', 'error': str(e), 'meta': {'simulated': True}}
            # Generic simulation: return prompt summary stub
            return {'text': f'[SIMULATED RESPONSE from {getattr(agent, "name", "agent")}] Short analysis of: {prompt[:200]}', 'error': None, 'meta': {'simulated': True}}

        # Actual run (live)
        try:
            # many agents have generate(prompt, temperature, max_tokens)
            if hasattr(agent, 'generate'):
                out = agent.generate(prompt, temperature=0.0, max_tokens=512)
            elif hasattr(agent, 'run'):
                out = agent.run(prompt)
            else:
                out = str(agent)
            return {'text': (out or '').strip(), 'error': None, 'meta': {'simulated': False}}
        except Exception as e:
            logger.exception('Agent execution error: %s', e)
            return {'text': '', 'error': str(e), 'meta': {'simulated': False}}

    def _embed_texts_safe(self, texts: List[str]):
        """
        Use project's embed_texts if available, otherwise return deterministic simple embeddings.
        """
        if not texts:
            return []
        if embed_texts:
            try:
                embs = embed_texts(texts)
                return embs
            except Exception as e:
                logger.exception('Embedder failed: %s. Falling back to token-length embeddings.', e)
        # fallback: length-based pseudo-embeddings (not semantically meaningful but safe)
        out = []
        for t in texts:
            # simple vector: [len, unique_tokens]
            toks = len(set(t.split()))
            out.append([len(t), toks, float(len(t)) / (toks + 1)])
        return out

    def _pairwise_similarity_matrix(self, texts: List[str]):
        """
        Returns NxN matrix of cosine similarities between texts using embeddings.
        """
        embs = self._embed_texts_safe(texts)
        n = len(embs)
        mat = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                try:
                    sim = cosine_similarity(embs[i], embs[j])
                except Exception:
                    sim = 1.0 if texts[i] == texts[j] else 0.0
                mat[i][j] = float(sim)
        return mat

    def _consensus_from_matrix(self, mat: List[List[float]], threshold: float, absolute_truth_mode: bool = False):
        """
        Determine consensus clusters given pairwise similarity matrix and threshold.
        Simple algorithm for 3 agents: majority rule.
        Returns:
            - verdict: 'consensus' | 'disagree' | 'uncertain'
            - supporting_pairs: list of (i,j,sim) pairs that met threshold
            - confidence: average of supporting similarities normalized
        """
        n = len(mat)
        if n == 0:
            return {'verdict': 'no_agents', 'supporting_pairs': [], 'confidence': 0.0}
        # collect pairs that exceed threshold
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                if mat[i][j] >= threshold:
                    pairs.append((i, j, mat[i][j]))
        # majority logic for 3 agents:
        if n == 3:
            if absolute_truth_mode:
                # ABSOLUTE TRUTH MODE: Require unanimous agreement (all pairs agree) with very high confidence
                if len(pairs) == 3:  # All three pairs must agree (unanimous)
                    conf_vals = [p[2] for p in pairs]
                    confidence = float(sum(conf_vals) / len(conf_vals))
                    if confidence >= 0.98:  # Require 98% confidence for absolute truth
                        return {'verdict': 'absolute_truth', 'supporting_pairs': pairs, 'confidence': confidence}
                    else:
                        return {'verdict': 'high_confidence_agreement', 'supporting_pairs': pairs, 'confidence': confidence}
                else:
                    return {'verdict': 'insufficient_evidence', 'supporting_pairs': pairs, 'confidence': 0.0}
            else:
                # Standard mode
                agrees = [0] * 3
                for (i, j, s) in pairs:
                    agrees[i] += 1
                    agrees[j] += 1
                # check if any agent has agrees >=2 (agrees with both others)
                majority = any(a >= 2 for a in agrees)
                if majority:
                    # compute confidence as mean of pair sims that met threshold
                    conf_vals = [p[2] for p in pairs] or []
                    confidence = float(sum(conf_vals) / len(conf_vals)) if conf_vals else 0.0
                    return {'verdict': 'consensus', 'supporting_pairs': pairs, 'confidence': confidence}
                elif len(pairs) >= 1:
                    # at least one pair agrees but no majority (two agents agree but not transitive)
                    confidence = float(sum(p[2] for p in pairs) / len(pairs)) if pairs else 0.0
                    return {'verdict': 'partial_agreement', 'supporting_pairs': pairs, 'confidence': confidence}
                else:
                    return {'verdict': 'disagree', 'supporting_pairs': [], 'confidence': 0.0}
        else:
            if absolute_truth_mode:
                # ABSOLUTE TRUTH MODE: Require ALL pairs to agree with very high confidence
                total_pairs = n * (n - 1) / 2
                if len(pairs) == total_pairs:  # All pairs must agree (unanimous)
                    conf_vals = [p[2] for p in pairs]
                    confidence = float(sum(conf_vals) / len(conf_vals))
                    if confidence >= 0.98:  # Require 98% confidence for absolute truth
                        return {'verdict': 'absolute_truth', 'supporting_pairs': pairs, 'confidence': confidence}
                    else:
                        return {'verdict': 'high_confidence_agreement', 'supporting_pairs': pairs, 'confidence': confidence}
                else:
                    return {'verdict': 'insufficient_evidence', 'supporting_pairs': pairs, 'confidence': 0.0}
            else:
                # general heuristic: if more than half of pairs agree -> consensus
                total_pairs = n * (n - 1) / 2
                if total_pairs == 0:
                    return {'verdict': 'no_pairs', 'supporting_pairs': [], 'confidence': 0.0}
                if len(pairs) >= (total_pairs / 2):
                    confidence = float(sum(p[2] for p in pairs) / len(pairs)) if pairs else 0.0
                    return {'verdict': 'consensus', 'supporting_pairs': pairs, 'confidence': confidence}
                elif pairs:
                    return {'verdict': 'partial_agreement', 'supporting_pairs': pairs, 'confidence': float(sum(p[2] for p in pairs) / len(pairs))}
                else:
                    return {'verdict': 'disagree', 'supporting_pairs': [], 'confidence': 0.0}

    def run(self,
            text: str,
            agent_configs: Optional[List[Dict[str, Any]]] = None,
            agent_instances: Optional[List[Any]] = None,
            orchestrator = None,
            dry_run: bool = True,
            timeout: Optional[int] = None,
            absolute_truth_mode: bool = False) -> Dict[str, Any]:
        """
        Main entrypoint.
        - text: the input to verify
        - agent_configs: optional list of 3 agent config dicts (will be created by create_agent)
        - agent_instances: optional list of 3 already-instantiated agent objects
        - orchestrator: optional Orchestrator instance to run agents via orchestrator.run_all_agents(prompt)
        - dry_run: if True, avoid calling remote providers aggressively
        - timeout: optional per-agent timeout (not strictly enforced here)
        """

        # Prepare agent instances (prefer explicit instances)
        agents = []
        if agent_instances:
            agents = list(agent_instances)[:3]
        elif agent_configs:
            agents = self._instantiate_agents(agent_configs)[:3]
        elif orchestrator and hasattr(orchestrator, 'agents'):
            agents = list(orchestrator.agents)[:3]
        else:
            logger.warning('No agents provided. Returning simulated analysis.')
            return {'error': 'no_agents', 'report': None, 'dry_run': True}

        # Ensure exactly 3 slots (pad with None values if fewer)
        while len(agents) < 3:
            agents.append(None)

        # Run agents
        outputs = []
        for idx, a in enumerate(agents):
            res = self._run_agent(a, text, timeout=timeout, dry_run=dry_run)
            outputs.append(res)

        # Extract texts for similarity
        texts = [o.get('text','') or '' for o in outputs]

        # Compute pairwise similarity matrix
        sim_matrix = self._pairwise_similarity_matrix(texts)

        # Derive consensus
        consensus = self._consensus_from_matrix(sim_matrix, self.similarity_threshold, self.absolute_truth_mode)

        # Build structured report
        report = {
            'input': text,
            'agents_count': len(agents),
            'agents': [
                {
                    'index': i,
                    'name': getattr(agents[i], 'name', f'agent_{i}'),
                    'provider': getattr(agents[i], 'provider', None),
                    'model': getattr(agents[i], 'model', None),
                    'output': outputs[i].get('text'),
                    'error': outputs[i].get('error'),
                    'meta': outputs[i].get('meta', {}),
                } for i in range(len(agents))
            ],
            'pairwise_similarity': sim_matrix,
            'consensus': consensus,
            'dry_run': bool(dry_run)
        }

        # Optionally compute a simple human-friendly verdict
        verdict_map = {
            'consensus': 'accepted_by_majority',
            'partial_agreement': 'partial_agreement_detected',
            'disagree': 'no_agreement_detected',
            'no_agents': 'no_agents_provided',
            'absolute_truth': 'absolute_truth_confirmed',
            'high_confidence_agreement': 'high_confidence_agreement',
            'insufficient_evidence': 'insufficient_evidence_for_truth'
        }
        report['verdict'] = verdict_map.get(consensus.get('verdict'), 'unknown')
        return report
