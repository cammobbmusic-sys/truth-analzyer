"""
Comprehensive AI Model Performance Evaluation Framework

A rigorous system for evaluating and comparing AI model performance across
genres, tasks, and metrics with statistical validation and automated insights.
"""

import json
import os
import time
import statistics
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class EvaluationGenre(Enum):
    """Task genres for evaluation."""
    FACTUAL_QA = "factual_qa"
    MATHEMATICAL = "mathematical"
    CREATIVE_WRITING = "creative_writing"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    ANALYSIS = "analysis"
    CONVERSATION = "conversation"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"


class MetricType(Enum):
    """Types of evaluation metrics."""
    QUANTITATIVE = "quantitative"
    QUALITATIVE = "qualitative"
    PERFORMANCE = "performance"
    COST = "cost"


@dataclass
class EvaluationMetric:
    """Definition of an evaluation metric."""
    name: str
    description: str
    metric_type: MetricType
    unit: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    higher_is_better: bool = True
    weight: float = 1.0

    def validate_value(self, value: Any) -> bool:
        """Validate if a value is acceptable for this metric."""
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True


@dataclass
class EvaluationResult:
    """Result of a single evaluation run."""
    evaluation_id: str
    model_name: str
    genre: EvaluationGenre
    task_description: str
    input_text: str
    expected_output: Optional[str]
    actual_output: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None

    def calculate_overall_score(self, metric_weights: Dict[str, float]) -> float:
        """Calculate weighted overall score."""
        total_weight = 0
        total_score = 0

        for metric_name, value in self.metrics.items():
            if metric_name in metric_weights and isinstance(value, (int, float)):
                weight = metric_weights[metric_name]
                # Normalize to 0-1 scale if needed
                normalized_value = min(max(value, 0), 1) if 0 <= value <= 1 else value
                total_score += normalized_value * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0


@dataclass
class StatisticalAnalysis:
    """Statistical analysis of evaluation results."""
    genre: EvaluationGenre
    model_name: str
    sample_size: int
    mean_score: float
    median_score: float
    std_deviation: float
    min_score: float
    max_score: float
    confidence_interval_95: Tuple[float, float]
    p_value_vs_baseline: Optional[float] = None
    effect_size: Optional[float] = None

    def is_significantly_better(self, other: 'StatisticalAnalysis',
                              alpha: float = 0.05) -> bool:
        """Check if this model is significantly better than another."""
        if self.p_value_vs_baseline is not None:
            return self.p_value_vs_baseline < alpha
        return False


class NotificationManager:
    """Handles automated notifications for performance insights."""

    def __init__(self, config_file: str = "notification_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self.notification_history = []

    def _load_config(self) -> Dict[str, Any]:
        """Load notification configuration."""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {
            "email_enabled": False,
            "email_server": "smtp.gmail.com",
            "email_port": 587,
            "email_user": "",
            "email_password": "",
            "email_recipients": [],
            "slack_webhook": "",
            "alert_thresholds": {
                "performance_drop": 0.1,  # 10% drop
                "significant_improvement": 0.15,  # 15% improvement
                "cost_anomaly": 2.0  # 2x cost increase
            }
        }

    def save_config(self):
        """Save notification configuration."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def send_notification(self, title: str, message: str, priority: str = "normal"):
        """Send notification via configured channels."""
        timestamp = datetime.now().isoformat()

        notification = {
            "timestamp": timestamp,
            "title": title,
            "message": message,
            "priority": priority
        }

        self.notification_history.append(notification)

        # Email notification
        if self.config.get("email_enabled", False):
            self._send_email_notification(title, message)

        # Slack notification
        slack_webhook = self.config.get("slack_webhook")
        if slack_webhook:
            self._send_slack_notification(title, message)

        # Log to file
        self._log_notification(notification)

        print(f"🔔 NOTIFICATION: {title}")
        print(f"   {message}")

    def _send_email_notification(self, title: str, message: str):
        """Send email notification."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config['email_user']
            msg['To'] = ', '.join(self.config['email_recipients'])
            msg['Subject'] = f"AI Evaluation Alert: {title}"

            body = f"""
AI Model Evaluation Alert

{title}

{message}

Timestamp: {datetime.now().isoformat()}

--
AI Evaluation Framework
            """
            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(self.config['email_server'], self.config['email_port'])
            server.starttls()
            server.login(self.config['email_user'], self.config['email_password'])
            text = msg.as_string()
            server.sendmail(self.config['email_user'], self.config['email_recipients'], text)
            server.quit()

        except Exception as e:
            print(f"Email notification failed: {e}")

    def _send_slack_notification(self, title: str, message: str):
        """Send Slack notification."""
        try:
            import requests
            payload = {
                "text": f"*AI Evaluation Alert*\n*{title}*\n{message}",
                "username": "AI Evaluator",
                "icon_emoji": ":robot_face:"
            }
            requests.post(self.config['slack_webhook'], json=payload)
        except Exception as e:
            print(f"Slack notification failed: {e}")

    def _log_notification(self, notification: Dict[str, Any]):
        """Log notification to file."""
        log_file = "notification_log.json"
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []

            logs.append(notification)

            # Keep only last 1000 notifications
            if len(logs) > 1000:
                logs = logs[-1000:]

            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)

        except Exception as e:
            print(f"Notification logging failed: {e}")

    def check_performance_anomaly(self, model_name: str, genre: EvaluationGenre,
                                current_score: float, historical_avg: float):
        """Check for performance anomalies and send alerts."""
        thresholds = self.config.get('alert_thresholds', {})

        if historical_avg > 0:
            change_percent = (current_score - historical_avg) / historical_avg

            if change_percent < -thresholds.get('performance_drop', 0.1):
                self.send_notification(
                    f"Performance Drop: {model_name}",
                    f"{model_name} performance in {genre.value} dropped by {abs(change_percent)*100:.1f}% "
                    f"(from {historical_avg:.3f} to {current_score:.3f})",
                    "high"
                )
            elif change_percent > thresholds.get('significant_improvement', 0.15):
                self.send_notification(
                    f"Performance Improvement: {model_name}",
                    f"{model_name} performance in {genre.value} improved by {change_percent*100:.1f}% "
                    f"(from {historical_avg:.3f} to {current_score:.3f})",
                    "normal"
                )


class ModelEvaluator:
    """
    Comprehensive AI model evaluation system.

    Handles data collection, metric calculation, statistical analysis,
    and automated insights for AI model performance evaluation.
    """

    def __init__(self, data_dir: str = "evaluation_data"):
        self.data_dir = data_dir
        self.notifications = NotificationManager()
        os.makedirs(data_dir, exist_ok=True)

        # Define standard metrics
        self.metrics = self._define_standard_metrics()

        # Data storage
        self.results: List[EvaluationResult] = []
        self.baseline_scores: Dict[str, Dict[EvaluationGenre, float]] = {}

        # Load existing data
        self._load_data()

    def _define_standard_metrics(self) -> Dict[str, EvaluationMetric]:
        """Define standard evaluation metrics."""
        return {
            "response_time": EvaluationMetric(
                name="Response Time",
                description="Time taken to generate response",
                metric_type=MetricType.PERFORMANCE,
                unit="seconds",
                higher_is_better=False,
                weight=0.2
            ),
            "accuracy": EvaluationMetric(
                name="Accuracy",
                description="Correctness of the response",
                metric_type=MetricType.QUANTITATIVE,
                unit="score",
                min_value=0.0,
                max_value=1.0,
                higher_is_better=True,
                weight=0.4
            ),
            "relevance": EvaluationMetric(
                name="Relevance",
                description="How relevant the response is to the query",
                metric_type=MetricType.QUALITATIVE,
                unit="score",
                min_value=0.0,
                max_value=1.0,
                higher_is_better=True,
                weight=0.2
            ),
            "creativity": EvaluationMetric(
                name="Creativity",
                description="Originality and creativity in response",
                metric_type=MetricType.QUALITATIVE,
                unit="score",
                min_value=0.0,
                max_value=1.0,
                higher_is_better=True,
                weight=0.15
            ),
            "cost_efficiency": EvaluationMetric(
                name="Cost Efficiency",
                description="Performance per unit cost",
                metric_type=MetricType.COST,
                unit="score_per_dollar",
                higher_is_better=True,
                weight=0.05
            )
        }

    def _load_data(self):
        """Load existing evaluation data."""
        results_file = os.path.join(self.data_dir, "evaluation_results.json")
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        # Convert timestamp string back to datetime
                        if 'timestamp' in item:
                            item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                        self.results.append(EvaluationResult(**item))
            except Exception as e:
                print(f"Warning: Could not load evaluation results: {e}")

        # Load baseline scores
        baseline_file = os.path.join(self.data_dir, "baseline_scores.json")
        if os.path.exists(baseline_file):
            try:
                with open(baseline_file, 'r') as f:
                    self.baseline_scores = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load baseline scores: {e}")

    def save_data(self):
        """Save evaluation data."""
        # Save results
        results_file = os.path.join(self.data_dir, "evaluation_results.json")
        try:
            results_data = []
            for result in self.results[-1000:]:  # Keep last 1000 results
                result_dict = asdict(result)
                result_dict['timestamp'] = result.timestamp.isoformat()
                results_data.append(result_dict)

            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not save evaluation results: {e}")

        # Save baseline scores
        baseline_file = os.path.join(self.data_dir, "baseline_scores.json")
        try:
            with open(baseline_file, 'w') as f:
                json.dump(self.baseline_scores, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save baseline scores: {e}")

    def evaluate_model(self, model_name: str, genre: EvaluationGenre,
                      task_description: str, input_text: str,
                      expected_output: Optional[str] = None,
                      custom_metrics: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """
        Evaluate a model on a specific task.

        This method should be overridden or extended with actual model calling logic.
        """
        # This is a placeholder - in real implementation, this would call the actual model
        evaluation_id = f"{model_name}_{genre.value}_{int(time.time())}"

        # Simulate evaluation (replace with actual model call)
        start_time = time.time()
        time.sleep(0.1)  # Simulate processing time
        processing_time = time.time() - start_time

        # Mock response (replace with actual model response)
        actual_output = f"Mock response for {task_description}"

        # Calculate metrics (replace with actual evaluation)
        metrics = {
            "response_time": processing_time,
            "accuracy": 0.85,  # Mock accuracy score
            "relevance": 0.9,   # Mock relevance score
            "creativity": 0.7,  # Mock creativity score
            "cost_efficiency": 0.8  # Mock cost efficiency
        }

        if custom_metrics:
            metrics.update(custom_metrics)

        result = EvaluationResult(
            evaluation_id=evaluation_id,
            model_name=model_name,
            genre=genre,
            task_description=task_description,
            input_text=input_text,
            expected_output=expected_output,
            actual_output=actual_output,
            metrics=metrics,
            processing_time=processing_time,
            success=True
        )

        self.results.append(result)
        self._check_for_anomalies(result)
        self.save_data()

        return result

    def _check_for_anomalies(self, result: EvaluationResult):
        """Check for performance anomalies and send notifications."""
        model_key = f"{result.model_name}_{result.genre.value}"

        # Calculate current score
        current_score = result.calculate_overall_score(
            {name: metric.weight for name, metric in self.metrics.items()}
        )

        # Get historical average
        historical_scores = [
            r.calculate_overall_score({name: metric.weight for name, metric in self.metrics.items()})
            for r in self.results[-20:]  # Last 20 evaluations
            if r.model_name == result.model_name and r.genre == result.genre
        ]

        if historical_scores:
            historical_avg = statistics.mean(historical_scores)
            self.notifications.check_performance_anomaly(
                result.model_name, result.genre, current_score, historical_avg
            )

    def get_statistical_analysis(self, model_name: str, genre: EvaluationGenre,
                               min_samples: int = 10) -> Optional[StatisticalAnalysis]:
        """Get statistical analysis for a model on a specific genre."""
        # Filter results
        relevant_results = [
            r for r in self.results
            if r.model_name == model_name and r.genre == genre and r.success
        ]

        if len(relevant_results) < min_samples:
            return None

        # Calculate scores
        scores = [
            r.calculate_overall_score({name: metric.weight for name, metric in self.metrics.items()})
            for r in relevant_results
        ]

        # Basic statistics
        mean_score = statistics.mean(scores)
        median_score = statistics.median(scores)
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0
        min_score = min(scores)
        max_score = max(scores)

        # Confidence interval (simplified)
        confidence_interval = (
            mean_score - 1.96 * std_dev / len(scores)**0.5,
            mean_score + 1.96 * std_dev / len(scores)**0.5
        )

        return StatisticalAnalysis(
            genre=genre,
            model_name=model_name,
            sample_size=len(scores),
            mean_score=mean_score,
            median_score=median_score,
            std_deviation=std_dev,
            min_score=min_score,
            max_score=max_score,
            confidence_interval_95=confidence_interval
        )

    def compare_models(self, genre: EvaluationGenre,
                      model_a: str, model_b: str) -> Dict[str, Any]:
        """Compare two models on a specific genre."""
        analysis_a = self.get_statistical_analysis(model_a, genre)
        analysis_b = self.get_statistical_analysis(model_b, genre)

        if not analysis_a or not analysis_b:
            return {
                "error": "Insufficient data for comparison",
                "model_a_samples": analysis_a.sample_size if analysis_a else 0,
                "model_b_samples": analysis_b.sample_size if analysis_b else 0
            }

        # Simple comparison (in real implementation, use statistical tests)
        winner = "tie"
        if analysis_a.mean_score > analysis_b.mean_score:
            winner = model_a
        elif analysis_b.mean_score > analysis_a.mean_score:
            winner = model_b

        return {
            "genre": genre.value,
            "comparison": {
                model_a: {
                    "mean_score": analysis_a.mean_score,
                    "confidence_interval": analysis_a.confidence_interval_95,
                    "sample_size": analysis_a.sample_size
                },
                model_b: {
                    "mean_score": analysis_b.mean_score,
                    "confidence_interval": analysis_b.confidence_interval_95,
                    "sample_size": analysis_b.sample_size
                }
            },
            "winner": winner,
            "score_difference": abs(analysis_a.mean_score - analysis_b.mean_score)
        }

    def generate_evaluation_report(self, model_name: Optional[str] = None,
                                 genre: Optional[EvaluationGenre] = None) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_evaluations": len(self.results),
            "models_evaluated": list(set(r.model_name for r in self.results)),
            "genres_evaluated": list(set(r.genre.value for r in self.results)),
            "performance_summary": {},
            "recommendations": []
        }

        # Filter results
        filtered_results = self.results
        if model_name:
            filtered_results = [r for r in filtered_results if r.model_name == model_name]
        if genre:
            filtered_results = [r for r in filtered_results if r.genre == genre]

        # Group by model and genre
        model_genre_stats = defaultdict(lambda: defaultdict(list))

        for result in filtered_results:
            if result.success:
                score = result.calculate_overall_score(
                    {name: metric.weight for name, metric in self.metrics.items()}
                )
                model_genre_stats[result.model_name][result.genre.value].append(score)

        # Calculate statistics
        for model, genres in model_genre_stats.items():
            report["performance_summary"][model] = {}
            for genre_name, scores in genres.items():
                if len(scores) >= 5:  # Minimum samples for meaningful stats
                    report["performance_summary"][model][genre_name] = {
                        "average_score": statistics.mean(scores),
                        "median_score": statistics.median(scores),
                        "sample_size": len(scores),
                        "consistency": 1.0 - (statistics.stdev(scores) if len(scores) > 1 else 0)
                    }

        # Generate recommendations
        if len(report["performance_summary"]) > 1:
            # Find best model per genre
            genre_best = {}
            for model, genres in report["performance_summary"].items():
                for genre, stats in genres.items():
                    if genre not in genre_best or stats["average_score"] > genre_best[genre]["score"]:
                        genre_best[genre] = {
                            "model": model,
                            "score": stats["average_score"]
                        }

            for genre, best in genre_best.items():
                report["recommendations"].append({
                    "type": "model_selection",
                    "genre": genre,
                    "recommended_model": best["model"],
                    "expected_score": best["score"]
                })

        return report

    def run_evaluation_suite(self, models: List[str], genres: List[EvaluationGenre],
                           tasks_per_genre: int = 5) -> Dict[str, Any]:
        """Run comprehensive evaluation suite."""
        results = {
            "evaluation_run_id": f"eval_{int(time.time())}",
            "start_time": datetime.now().isoformat(),
            "models_tested": models,
            "genres_tested": [g.value for g in genres],
            "tasks_per_genre": tasks_per_genre,
            "results": []
        }

        # Define sample tasks for each genre (in practice, this would be much more comprehensive)
        sample_tasks = {
            EvaluationGenre.FACTUAL_QA: [
                {"input": "What is the capital of France?", "expected": "Paris"},
                {"input": "Who wrote Romeo and Juliet?", "expected": "William Shakespeare"},
                {"input": "What is the largest planet in our solar system?", "expected": "Jupiter"}
            ],
            EvaluationGenre.MATHEMATICAL: [
                {"input": "What is 15 + 27?", "expected": "42"},
                {"input": "What is the square root of 144?", "expected": "12"}
            ],
            EvaluationGenre.CREATIVE_WRITING: [
                {"input": "Write a haiku about artificial intelligence", "expected": None},
                {"input": "Describe a sunset in three sentences", "expected": None}
            ]
        }

        for genre in genres:
            tasks = sample_tasks.get(genre, [{"input": f"Sample {genre.value} task", "expected": None}])

            for model in models:
                for i, task in enumerate(tasks[:tasks_per_genre]):
                    result = self.evaluate_model(
                        model_name=model,
                        genre=genre,
                        task_description=f"{genre.value} task {i+1}",
                        input_text=task["input"],
                        expected_output=task.get("expected")
                    )
                    results["results"].append(asdict(result))

        results["end_time"] = datetime.now().isoformat()
        results["total_evaluations"] = len(results["results"])

        return results


# Global evaluator instance
evaluator = ModelEvaluator()
