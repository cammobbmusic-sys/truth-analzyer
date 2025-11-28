
"""Triangulation Orchestrator

Coordinates multiple agents (3–9), retries failing agents,
and computes final consensus for truth verification.
"""

import logging
import time
from typing import List, Dict, Any, Optional

try:
    from orchestrator.pipelines.verify_pipeline import VerificationPipeline
except Exception:
    VerificationPipeline = None

try:
    from agents.factory import create_agent
except Exception:
    create_agent = None

# Performance optimizer and evaluation integration
try:
    from optimizer import optimizer
except Exception:
    optimizer = None

try:
    from evaluation_framework import evaluator, EvaluationResult, EvaluationGenre
    from evaluation_tasks import task_library
except Exception:
    evaluator = None
    EvaluationResult = None
    EvaluationGenre = None
    task_library = None

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
            logger.warning('agents.factory.create_agent not importable; using simulated agents.')
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
            logger.warning('No agents provided; using simulated fallback.')
            return {'error': 'no_agents', 'report': None, 'dry_run': True}

        # Ensure at least 3 agents
        while len(agents) < 3:
            agents.append(None)

        # Run comprehensive verification with all agents
        retry_count = 0
        report = None

        while retry_count <= self.max_retries:
            if self.verifier:
                report = self.verifier.run(
                    text=text,
                    agent_instances=agents,
                    dry_run=dry_run
                )
            else:
                report = {'error': 'no_verifier', 'dry_run': dry_run}

            confidence = report.get('consensus', {}).get('confidence', 0)
            if confidence >= self.similarity_threshold or retry_count == self.max_retries:
                break
            retry_count += 1

        report['retries'] = retry_count

        # Get optimization insights
        optimization_insights = {}
        if optimizer:
            # Determine query type for recommendations
            query_lower = text.lower()
            if any(word in query_lower for word in ['what', 'who', 'when', 'where', 'how many']):
                query_type = 'factual'
            elif any(word in query_lower for word in ['explain', 'analyze', 'compare']):
                query_type = 'analytical'
            elif any(word in query_lower for word in ['create', 'generate', 'design']):
                query_type = 'creative'
            else:
                query_type = 'general'

            # Get best agent recommendations
            best_speed, speed_score = optimizer.get_best_agent(query_type, 'speed')
            best_accuracy, accuracy_score = optimizer.get_best_agent(query_type, 'accuracy')
            best_cost, cost_score = optimizer.get_best_agent(query_type, 'cost')

            optimization_insights = {
                'query_type': query_type,
                'recommendations': {
                    'best_for_speed': {'agent': best_speed, 'score': speed_score},
                    'best_for_accuracy': {'agent': best_accuracy, 'score': accuracy_score},
                    'best_for_cost': {'agent': best_cost, 'score': cost_score}
                },
                'total_queries_tracked': sum(m.total_queries for m in optimizer.metrics.values())
            }

        # Add optimization insights to the report
        report['optimization_insights'] = optimization_insights

        return report

        return final_report

    def _classify_query_genre(self, text: str) -> EvaluationGenre:
        """Classify query into evaluation genre based on content analysis."""
        text_lower = text.lower()

        # Factual questions
        if any(word in text_lower for word in ['what is', 'who is', 'when did', 'where is', 'how many', 'capital of']):
            return EvaluationGenre.FACTUAL_QA

        # Mathematical
        elif any(word in text_lower for word in ['calculate', 'solve', 'equation', 'square root', 'derivative', '+', '-', '×', '÷']):
            return EvaluationGenre.MATHEMATICAL

        # Reasoning
        elif any(word in text_lower for word in ['therefore', 'conclude', 'follows', 'logically', 'reasoning', 'if all', 'some are']):
            return EvaluationGenre.REASONING

        # Creative writing
        elif any(word in text_lower for word in ['write a', 'compose', 'create a', 'haiku', 'story', 'poem', 'describe']):
            return EvaluationGenre.CREATIVE_WRITING

        # Code generation
        elif any(word in text_lower for word in ['write a function', 'code for', 'implement', 'program', 'script']):
            return EvaluationGenre.CODE_GENERATION

        # Analysis
        elif any(word in text_lower for word in ['analyze', 'compare', 'evaluate', 'assess', 'impact of']):
            return EvaluationGenre.ANALYSIS

        # Conversation
        elif any(word in text_lower for word in ['hello', 'how are you', 'feeling', 'advice', 'help me']):
            return EvaluationGenre.CONVERSATION

        # Summarization
        elif any(word in text_lower for word in ['summarize', 'summary of', 'in brief', 'key points']):
            return EvaluationGenre.SUMMARIZATION

        # Classification
        elif any(word in text_lower for word in ['classify', 'category', 'type of', 'kind of']):
            return EvaluationGenre.CLASSIFICATION

        # Translation (basic detection)
        elif any(word in text_lower for word in ['translate', 'translation', 'in spanish', 'in french', 'to german']):
            return EvaluationGenre.TRANSLATION

        # Default to analysis for complex queries
        else:
            return EvaluationGenre.ANALYSIS

    def _calculate_evaluation_metrics(self, evaluation_result: 'EvaluationResult',
                                    agent_report: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics for the result."""
        metrics = {}

        # Basic performance metrics
        metrics['response_time'] = evaluation_result.processing_time

        # Success metric
        metrics['success'] = 1.0 if evaluation_result.success else 0.0

        # Consensus-based accuracy (from agent report)
        consensus = agent_report.get('consensus', {})
        metrics['accuracy'] = consensus.get('confidence', 0.5)

        # Response quality metrics (heuristic-based for now)
        output = evaluation_result.actual_output
        input_text = evaluation_result.input_text

        # Relevance score (basic heuristic)
        if output and len(output.strip()) > 10:
            # Check if output seems relevant to input
            input_words = set(input_text.lower().split())
            output_words = set(output.lower().split())
            overlap = len(input_words.intersection(output_words))
            metrics['relevance'] = min(1.0, overlap / max(len(input_words), 1))
        else:
            metrics['relevance'] = 0.0

        # Creativity score (for creative tasks)
        if evaluation_result.genre in [EvaluationGenre.CREATIVE_WRITING, EvaluationGenre.CONVERSATION]:
            # Basic creativity heuristics
            unique_words = len(set(output.lower().split())) if output else 0
            total_words = len(output.split()) if output else 0
            metrics['creativity'] = min(1.0, unique_words / max(total_words, 1))
        else:
            metrics['creativity'] = 0.5  # Neutral for non-creative tasks

        # Cost efficiency (simulated - would need real cost data)
        base_cost = 0.001 if 'openrouter' in evaluation_result.model_name.lower() else 0.0005
        metrics['cost_efficiency'] = metrics['accuracy'] / max(base_cost, 0.0001)

        # Genre-specific metrics
        if evaluation_result.genre == EvaluationGenre.MATHEMATICAL:
            # Check for numerical answers
            import re
            numbers = re.findall(r'\d+', output)
            metrics['numerical_answer'] = 1.0 if numbers else 0.0

        elif evaluation_result.genre == EvaluationGenre.CODE_GENERATION:
            # Basic code quality checks
            code_indicators = ['def ', 'class ', 'import ', 'function', 'return ']
            code_score = sum(1 for indicator in code_indicators if indicator in output)
            metrics['code_quality'] = min(1.0, code_score / 3)

        elif evaluation_result.genre == EvaluationGenre.REASONING:
            # Reasoning quality (basic heuristics)
            reasoning_words = ['because', 'therefore', 'thus', 'consequently', 'follows']
            reasoning_score = sum(1 for word in reasoning_words if word in output.lower())
            metrics['logical_reasoning'] = min(1.0, reasoning_score / 2)

        return metrics

    def execute(self, *args, **kwargs):
        """Execute the triangulation orchestrator."""
        pass

    def coordinate(self, *args, **kwargs):
        """Coordinate the triangulation orchestrator."""
        pass
