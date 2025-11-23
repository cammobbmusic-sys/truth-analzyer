"""
Metrics - Evaluation metrics computation.
Provides quantitative evaluation of AI model responses and system performance.
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import statistics


class MetricsCalculator:
    """Calculator for various evaluation metrics."""

    def __init__(self):
        self.metrics_history = []

    def calculate_response_quality_metrics(
        self,
        response: str,
        expected_length: Optional[int] = None,
        query_complexity: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate quality metrics for a single response.

        Args:
            response: The AI response to evaluate
            expected_length: Expected response length (optional)
            query_complexity: Complexity score of the query (optional)

        Returns:
            Dictionary of quality metrics
        """
        metrics = {
            "length": len(response),
            "word_count": len(response.split()),
            "sentence_count": len(response.split('.')),
            "has_structure": self._has_structure(response),
            "clarity_score": self._calculate_clarity_score(response),
            "completeness_score": self._calculate_completeness_score(response),
            "timestamp": time.time()
        }

        if expected_length:
            metrics["length_deviation"] = abs(len(response) - expected_length) / expected_length

        if query_complexity:
            metrics["complexity_match"] = self._calculate_complexity_match(response, query_complexity)

        return metrics

    def _has_structure(self, response: str) -> bool:
        """Check if response has structural elements."""
        structure_indicators = [
            '1.', '2.', '3.',      # Numbered lists
            '•', '*', '-',         # Bullet points
            'First', 'Second', 'Next',  # Sequential indicators
            'However', 'Therefore', 'In conclusion'  # Logical connectors
        ]

        return any(indicator in response for indicator in structure_indicators)

    def _calculate_clarity_score(self, response: str) -> float:
        """Calculate clarity score based on language patterns."""
        score = 0.5  # Base score

        # Check for clear language
        clear_indicators = ['clearly', 'obviously', 'specifically', 'precisely']
        unclear_indicators = ['maybe', 'perhaps', 'might', 'could', 'possibly']

        clear_count = sum(1 for word in clear_indicators if word in response.lower())
        unclear_count = sum(1 for word in unclear_indicators if word in response.lower())

        # Adjust score based on clarity indicators
        score += clear_count * 0.1
        score -= unclear_count * 0.1

        # Check sentence length (prefer moderate length)
        sentences = response.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

        if 10 <= avg_sentence_length <= 25:
            score += 0.2
        elif avg_sentence_length > 40:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _calculate_completeness_score(self, response: str) -> float:
        """Calculate completeness score."""
        score = 0.5  # Base score

        # Check for comprehensive elements
        completeness_indicators = [
            'because', 'due to', 'therefore', 'thus',  # Explanations
            'example', 'instance', 'case',  # Examples
            'however', 'but', 'although',  # Nuance
            'summary', 'conclusion', 'overall'  # Summarization
        ]

        found_indicators = sum(1 for indicator in completeness_indicators
                             if indicator in response.lower())

        score += found_indicators * 0.05

        # Length bonus (longer responses tend to be more complete)
        word_count = len(response.split())
        if word_count > 100:
            score += 0.2
        elif word_count > 50:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _calculate_complexity_match(self, response: str, query_complexity: float) -> float:
        """Calculate how well response complexity matches query complexity."""
        response_complexity = self._calculate_response_complexity(response)

        # Ideal match when response complexity is close to query complexity
        complexity_match = 1.0 - abs(response_complexity - query_complexity)

        return max(0.0, complexity_match)

    def _calculate_response_complexity(self, response: str) -> float:
        """Calculate complexity score of response."""
        words = response.split()
        sentences = response.split('.')

        if not words:
            return 0.0

        # Factors contributing to complexity
        avg_word_length = sum(len(word) for word in words) / len(words)
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        vocabulary_richness = len(set(words)) / len(words)

        # Combine factors (weighted average)
        complexity = (
            avg_word_length * 0.3 +
            min(avg_sentence_length / 20, 1.0) * 0.3 +  # Normalize sentence length
            vocabulary_richness * 0.4
        )

        return min(1.0, complexity)

    def calculate_consensus_metrics(
        self,
        responses: List[str],
        consensus_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate consensus and agreement metrics across multiple responses.

        Args:
            responses: List of AI responses
            consensus_answer: Agreed-upon answer (optional)

        Returns:
            Consensus metrics
        """
        if len(responses) < 2:
            return {"agreement_score": 1.0, "diversity_score": 0.0}

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                sim = self._calculate_text_similarity(responses[i], responses[j])
                similarities.append(sim)

        agreement_score = sum(similarities) / len(similarities) if similarities else 0.0
        diversity_score = 1.0 - agreement_score

        metrics = {
            "agreement_score": agreement_score,
            "diversity_score": diversity_score,
            "num_responses": len(responses),
            "similarity_std": statistics.stdev(similarities) if len(similarities) > 1 else 0.0
        }

        if consensus_answer:
            # Calculate how well each response matches consensus
            consensus_matches = [
                self._calculate_text_similarity(response, consensus_answer)
                for response in responses
            ]
            metrics["consensus_alignment"] = sum(consensus_matches) / len(consensus_matches)
            metrics["consensus_variance"] = statistics.stdev(consensus_matches) if len(consensus_matches) > 1 else 0.0

        return metrics

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def calculate_performance_metrics(
        self,
        operation_times: List[float],
        success_rates: List[bool],
        response_qualities: List[float]
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics for system operations.

        Args:
            operation_times: List of operation durations
            success_rates: List of success/failure indicators
            response_qualities: List of response quality scores

        Returns:
            Performance metrics
        """
        metrics = {}

        # Time metrics
        if operation_times:
            metrics["avg_response_time"] = statistics.mean(operation_times)
            metrics["median_response_time"] = statistics.median(operation_times)
            metrics["min_response_time"] = min(operation_times)
            metrics["max_response_time"] = max(operation_times)
            if len(operation_times) > 1:
                metrics["response_time_std"] = statistics.stdev(operation_times)

        # Success metrics
        if success_rates:
            metrics["success_rate"] = sum(success_rates) / len(success_rates)
            metrics["total_operations"] = len(success_rates)

        # Quality metrics
        if response_qualities:
            metrics["avg_quality_score"] = statistics.mean(response_qualities)
            metrics["quality_std"] = statistics.stdev(response_qualities) if len(response_qualities) > 1 else 0.0
            metrics["quality_range"] = f"{min(response_qualities):.2f} - {max(response_qualities):.2f}"

        return metrics

    def calculate_model_comparison_metrics(
        self,
        model_responses: Dict[str, List[str]],
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate metrics for comparing different AI models.

        Args:
            model_responses: Dictionary mapping model names to lists of responses
            ground_truth: Ground truth answer for comparison (optional)

        Returns:
            Model comparison metrics
        """
        metrics = {}

        if not model_responses:
            return metrics

        # Basic statistics
        metrics["models_compared"] = list(model_responses.keys())
        metrics["total_responses"] = sum(len(responses) for responses in model_responses.values())

        # Calculate per-model metrics
        model_metrics = {}
        all_responses = []

        for model, responses in model_responses.items():
            model_metrics[model] = {
                "response_count": len(responses),
                "avg_length": statistics.mean([len(r) for r in responses]) if responses else 0,
                "avg_quality": statistics.mean([
                    self.calculate_response_quality_metrics(r)["completeness_score"]
                    for r in responses
                ]) if responses else 0
            }
            all_responses.extend(responses)

        metrics["model_metrics"] = model_metrics

        # Cross-model agreement
        if len(model_responses) > 1:
            metrics["cross_model_agreement"] = self.calculate_consensus_metrics(all_responses)

        # Ground truth comparison (if available)
        if ground_truth:
            ground_truth_similarities = {}
            for model, responses in model_responses.items():
                similarities = [
                    self._calculate_text_similarity(response, ground_truth)
                    for response in responses
                ]
                ground_truth_similarities[model] = {
                    "avg_similarity": statistics.mean(similarities) if similarities else 0,
                    "max_similarity": max(similarities) if similarities else 0
                }
            metrics["ground_truth_alignment"] = ground_truth_similarities

        return metrics

    def calculate_trend_metrics(
        self,
        historical_metrics: List[Dict[str, Any]],
        time_window: int = 10
    ) -> Dict[str, Any]:
        """
        Calculate trend metrics from historical data.

        Args:
            historical_metrics: List of historical metric dictionaries
            time_window: Number of recent data points to analyze

        Returns:
            Trend analysis metrics
        """
        if len(historical_metrics) < 2:
            return {"insufficient_data": True}

        # Use recent data
        recent_data = historical_metrics[-time_window:]

        trends = {}

        # Identify numeric metrics to track
        numeric_keys = ["agreement_score", "response_time", "quality_score", "success_rate"]

        for key in numeric_keys:
            values = [m.get(key) for m in recent_data if m.get(key) is not None]

            if len(values) >= 2:
                # Calculate trend (slope)
                try:
                    trend = self._calculate_trend(values)
                    trends[key] = {
                        "current_value": values[-1],
                        "trend": trend,
                        "trend_direction": "improving" if trend > 0.01 else "declining" if trend < -0.01 else "stable",
                        "data_points": len(values)
                    }
                except Exception:
                    continue

        return {
            "trends": trends,
            "analysis_window": time_window,
            "total_historical_points": len(historical_metrics)
        }

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend (slope) of values."""
        if len(values) < 2:
            return 0.0

        # Simple linear regression slope
        n = len(values)
        x = list(range(n))

        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(xi * yi for xi, yi in zip(x, values))
        sum_xx = sum(xi * xi for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)

        return slope

    def generate_performance_report(
        self,
        metrics_data: Dict[str, Any],
        include_trends: bool = True
    ) -> str:
        """
        Generate a human-readable performance report.

        Args:
            metrics_data: Metrics data to include in report
            include_trends: Whether to include trend analysis

        Returns:
            Formatted performance report
        """
        report_lines = ["MULTI-AI SYSTEM PERFORMANCE REPORT", "=" * 50]

        # System overview
        if "performance" in metrics_data:
            perf = metrics_data["performance"]
            report_lines.append("\nSYSTEM PERFORMANCE:")
            report_lines.append(".2f")
            report_lines.append(".1%")
            if "avg_quality_score" in perf:
                report_lines.append(".3f")

        # Model comparison
        if "model_comparison" in metrics_data:
            comp = metrics_data["model_comparison"]
            report_lines.append("\nMODEL COMPARISON:")
            for model, metrics in comp.get("model_metrics", {}).items():
                report_lines.append(f"  {model}:")
                report_lines.append(f"    Responses: {metrics['response_count']}")
                report_lines.append(".1f")
                report_lines.append(".3f")

        # Consensus analysis
        if "consensus" in metrics_data:
            cons = metrics_data["consensus"]
            report_lines.append("\nCONSENSUS ANALYSIS:")
            report_lines.append(".3f")
            report_lines.append(".3f")

        # Trends
        if include_trends and "trends" in metrics_data:
            trends = metrics_data["trends"]
            report_lines.append("\nTRENDS:")
            for metric, trend_data in trends.get("trends", {}).items():
                direction = trend_data["trend_direction"]
                current = trend_data["current_value"]
                report_lines.append(f"  {metric}: {direction} (current: {current:.3f})")

        return "\n".join(report_lines)

    def store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics for historical analysis."""
        metrics["timestamp"] = time.time()
        self.metrics_history.append(metrics)

        # Limit history size
        max_history = 10000
        if len(self.metrics_history) > max_history:
            self.metrics_history = self.metrics_history[-max_history:]

    def get_metrics_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent metrics history."""
        return self.metrics_history[-limit:]


def consensus_score(verified_count: int, total_count: int) -> float:
    """
    Calculate consensus score as ratio of verified items to total items.

    Args:
        verified_count: Number of verified/agreed items
        total_count: Total number of items

    Returns:
        Consensus score between 0 and 1
    """
    return verified_count / max(total_count, 1)


def diversity_score(clusters: list) -> int:
    """
    Calculate diversity score as the number of distinct clusters.

    Args:
        clusters: List of clusters

    Returns:
        Number of clusters (higher = more diverse)
    """
    return len(clusters)


def coherence_score(outputs: dict) -> float:
    """
    Calculate coherence score as average semantic similarity across outputs.

    Args:
        outputs: Dictionary mapping agent roles to their output text

    Returns:
        Average pairwise similarity score (0-1)
    """
    from .embeddings import pairwise_similarity
    return pairwise_similarity(list(outputs.values()))
