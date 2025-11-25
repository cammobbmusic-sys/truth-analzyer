"""
Simple Performance Optimizer for Hosted AI APIs

Tracks performance metrics and optimizes routing decisions for hosted API models.
Unlike Agent Lightning (designed for training), this optimizes how we use fixed models.
"""

import json
import os
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import statistics


@dataclass
class PerformanceMetrics:
    """Performance metrics for an agent/API combination."""
    agent_name: str
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    average_response_time: float = 0.0
    average_confidence: float = 0.0
    total_cost: float = 0.0
    last_used: Optional[datetime] = None
    response_times: List[float] = None
    confidence_scores: List[float] = None

    def __post_init__(self):
        if self.response_times is None:
            self.response_times = []
        if self.confidence_scores is None:
            self.confidence_scores = []

    def update_metrics(self, response_time: float, confidence: float, success: bool, cost: float = 0.0):
        """Update metrics with new query data."""
        self.total_queries += 1
        if success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1

        self.response_times.append(response_time)
        self.confidence_scores.append(confidence)
        self.total_cost += cost
        self.last_used = datetime.now()

        # Update averages (keep last 100 measurements for rolling average)
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
        if len(self.confidence_scores) > 100:
            self.confidence_scores = self.confidence_scores[-100:]

        self.average_response_time = statistics.mean(self.response_times) if self.response_times else 0.0
        self.average_confidence = statistics.mean(self.confidence_scores) if self.confidence_scores else 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_queries / self.total_queries if self.total_queries > 0 else 0.0

    @property
    def cost_per_query(self) -> float:
        """Calculate average cost per query."""
        return self.total_cost / self.total_queries if self.total_queries > 0 else 0.0


class SimpleOptimizer:
    """
    Simple performance optimizer for hosted AI APIs.

    Tracks metrics and makes intelligent routing decisions:
    - Which agent performs best for different query types
    - Cost optimization
    - Performance monitoring
    - Intelligent fallbacks
    """

    def __init__(self, metrics_file: str = "optimizer_metrics.json"):
        self.metrics_file = metrics_file
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.query_patterns: Dict[str, Dict[str, float]] = {}  # query_type -> {agent: score}
        self.load_metrics()

    def load_metrics(self):
        """Load metrics from disk."""
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)

                # Convert loaded data back to PerformanceMetrics objects
                for agent_name, metrics_data in data.get('metrics', {}).items():
                    # Handle datetime deserialization
                    last_used = None
                    if metrics_data.get('last_used'):
                        try:
                            last_used = datetime.fromisoformat(metrics_data['last_used'])
                        except:
                            pass

                    self.metrics[agent_name] = PerformanceMetrics(
                        agent_name=agent_name,
                        total_queries=metrics_data.get('total_queries', 0),
                        successful_queries=metrics_data.get('successful_queries', 0),
                        failed_queries=metrics_data.get('failed_queries', 0),
                        average_response_time=metrics_data.get('average_response_time', 0.0),
                        average_confidence=metrics_data.get('average_confidence', 0.0),
                        total_cost=metrics_data.get('total_cost', 0.0),
                        last_used=last_used,
                        response_times=metrics_data.get('response_times', []),
                        confidence_scores=metrics_data.get('confidence_scores', [])
                    )

                self.query_patterns = data.get('query_patterns', {})

            except Exception as e:
                print(f"Warning: Could not load metrics file: {e}")
                # Initialize empty if loading fails

    def save_metrics(self):
        """Save metrics to disk."""
        try:
            # Convert metrics to serializable format
            metrics_data = {}
            for agent_name, metrics in self.metrics.items():
                metrics_dict = asdict(metrics)
                # Convert datetime to ISO string
                if metrics.last_used:
                    metrics_dict['last_used'] = metrics.last_used.isoformat()
                metrics_data[agent_name] = metrics_dict

            data = {
                'metrics': metrics_data,
                'query_patterns': self.query_patterns,
                'last_updated': datetime.now().isoformat()
            }

            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not save metrics file: {e}")

    def get_or_create_metrics(self, agent_name: str) -> PerformanceMetrics:
        """Get metrics for an agent, creating if doesn't exist."""
        if agent_name not in self.metrics:
            self.metrics[agent_name] = PerformanceMetrics(agent_name=agent_name)
        return self.metrics[agent_name]

    def track_performance(self, agent_name: str, response_time: float,
                         confidence: float, success: bool, query_type: str = "general",
                         cost: float = 0.0):
        """
        Track performance metrics for an agent.

        Args:
            agent_name: Name of the agent
            response_time: Time taken for response (seconds)
            confidence: Confidence score (0.0-1.0)
            success: Whether the query was successful
            query_type: Category of query (e.g., "factual", "creative", "analysis")
            cost: Cost of the API call
        """
        metrics = self.get_or_create_metrics(agent_name)
        metrics.update_metrics(response_time, confidence, success, cost)

        # Update query pattern learning
        if query_type not in self.query_patterns:
            self.query_patterns[query_type] = {}

        # Simple scoring: higher confidence + faster response + success = better score
        base_score = confidence * 0.6 + (1.0 - min(response_time / 10.0, 1.0)) * 0.3 + (1.0 if success else 0.0) * 0.1
        self.query_patterns[query_type][agent_name] = base_score

        # Auto-save every 10 queries
        if sum(m.total_queries for m in self.metrics.values()) % 10 == 0:
            self.save_metrics()

    def get_best_agent(self, query_type: str = "general",
                      prioritize: str = "balanced") -> Tuple[str, float]:
        """
        Get the best agent for a query type based on learned patterns.

        Args:
            query_type: Type of query
            prioritize: "speed", "accuracy", "cost", or "balanced"

        Returns:
            Tuple of (agent_name, confidence_score)
        """
        if query_type in self.query_patterns and self.query_patterns[query_type]:
            # Use learned patterns
            agent_scores = self.query_patterns[query_type]
        else:
            # Fall back to general metrics
            agent_scores = {}
            for agent_name, metrics in self.metrics.items():
                if metrics.total_queries > 0:
                    # Calculate overall score
                    success_rate = metrics.success_rate
                    avg_confidence = metrics.average_confidence
                    avg_time = metrics.average_confidence
                    cost_per_query = metrics.cost_per_query

                    if prioritize == "speed":
                        score = (1.0 - min(avg_time / 5.0, 1.0)) * 0.7 + success_rate * 0.3
                    elif prioritize == "accuracy":
                        score = avg_confidence * 0.8 + success_rate * 0.2
                    elif prioritize == "cost":
                        score = (1.0 - min(cost_per_query / 0.01, 1.0)) * 0.6 + success_rate * 0.4
                    else:  # balanced
                        score = (avg_confidence * 0.4 +
                               (1.0 - min(avg_time / 5.0, 1.0)) * 0.3 +
                               success_rate * 0.2 +
                               (1.0 - min(cost_per_query / 0.005, 1.0)) * 0.1)

                    agent_scores[agent_name] = score

        if not agent_scores:
            return "groq-llama", 0.5  # Default fallback

        # Return agent with highest score
        best_agent = max(agent_scores.items(), key=lambda x: x[1])
        return best_agent

    def get_performance_report(self) -> Dict[str, Any]:
        """Get a comprehensive performance report."""
        report = {
            'total_queries': sum(m.total_queries for m in self.metrics.values()),
            'agent_performance': {},
            'query_patterns_learned': len(self.query_patterns),
            'optimization_ready': len(self.metrics) >= 2 and any(m.total_queries >= 5 for m in self.metrics.values())
        }

        for agent_name, metrics in self.metrics.items():
            report['agent_performance'][agent_name] = {
                'total_queries': metrics.total_queries,
                'success_rate': metrics.success_rate,
                'average_response_time': metrics.average_response_time,
                'average_confidence': metrics.average_confidence,
                'total_cost': metrics.total_cost,
                'cost_per_query': metrics.cost_per_query,
                'last_used': metrics.last_used.isoformat() if metrics.last_used else None
            }

        return report

    def reset_metrics(self, agent_name: Optional[str] = None):
        """Reset metrics for an agent or all agents."""
        if agent_name:
            if agent_name in self.metrics:
                self.metrics[agent_name] = PerformanceMetrics(agent_name=agent_name)
        else:
            self.metrics = {}

        # Remove related query patterns
        for query_type in list(self.query_patterns.keys()):
            if agent_name:
                self.query_patterns[query_type].pop(agent_name, None)
                if not self.query_patterns[query_type]:
                    del self.query_patterns[query_type]
            else:
                self.query_patterns[query_type] = {}

        self.save_metrics()


# Global optimizer instance
optimizer = SimpleOptimizer()
