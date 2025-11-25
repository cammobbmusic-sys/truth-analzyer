"""
AI Model Evaluation Dashboard

Comprehensive web interface for viewing evaluation results,
comparative analysis, and managing the evaluation framework.
"""

from flask import Flask, render_template, jsonify, request, send_file
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import io
import base64

from evaluation_framework import evaluator, EvaluationGenre, StatisticalAnalysis
from evaluation_tasks import task_library


class EvaluationDashboard:
    """Web dashboard for AI model evaluation results."""

    def __init__(self, evaluator_instance, task_library_instance):
        self.evaluator = evaluator_instance
        self.task_library = task_library_instance
        self.app = Flask(__name__, template_folder='ui/templates')
        self._setup_routes()

    def _setup_routes(self):
        """Set up Flask routes for the dashboard."""

        @self.app.route('/')
        def dashboard():
            """Main dashboard page."""
            return render_template('evaluation_dashboard.html')

        @self.app.route('/api/evaluation_summary')
        def evaluation_summary():
            """Get evaluation summary statistics."""
            try:
                report = self.evaluator.generate_evaluation_report()

                # Add additional statistics
                total_evaluations = len(self.evaluator.results)
                successful_evaluations = len([r for r in self.evaluator.results if r.success])
                success_rate = successful_evaluations / total_evaluations if total_evaluations > 0 else 0

                # Recent evaluations (last 24 hours)
                yesterday = datetime.now() - timedelta(days=1)
                recent_evaluations = len([
                    r for r in self.evaluator.results
                    if r.timestamp > yesterday
                ])

                summary = {
                    "total_evaluations": total_evaluations,
                    "successful_evaluations": successful_evaluations,
                    "success_rate": success_rate,
                    "recent_evaluations": recent_evaluations,
                    "models_evaluated": report.get("models_evaluated", []),
                    "genres_evaluated": report.get("genres_evaluated", []),
                    "performance_summary": report.get("performance_summary", {}),
                    "recommendations": report.get("recommendations", [])
                }

                return jsonify(summary)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/model_comparison/<model_a>/<model_b>')
        def model_comparison(model_a: str, model_b: str):
            """Compare two models across all genres."""
            try:
                comparisons = {}

                for genre in EvaluationGenre:
                    try:
                        comparison = self.evaluator.compare_models(genre, model_a, model_b)
                        if "error" not in comparison:
                            comparisons[genre.value] = comparison
                    except Exception as e:
                        print(f"Comparison failed for {genre.value}: {e}")
                        continue

                return jsonify({
                    "model_a": model_a,
                    "model_b": model_b,
                    "comparisons": comparisons,
                    "generated_at": datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/statistical_analysis/<model>/<genre>')
        def statistical_analysis(model: str, genre: str):
            """Get statistical analysis for a specific model and genre."""
            try:
                genre_enum = EvaluationGenre(genre)
                analysis = self.evaluator.get_statistical_analysis(model, genre_enum)

                if analysis:
                    return jsonify({
                        "model": model,
                        "genre": genre,
                        "analysis": {
                            "sample_size": analysis.sample_size,
                            "mean_score": analysis.mean_score,
                            "median_score": analysis.median_score,
                            "std_deviation": analysis.std_deviation,
                            "min_score": analysis.min_score,
                            "max_score": analysis.max_score,
                            "confidence_interval_95": analysis.confidence_interval_95
                        }
                    })
                else:
                    return jsonify({
                        "error": "Insufficient data for statistical analysis",
                        "model": model,
                        "genre": genre
                    }), 404
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/task_library_stats')
        def task_library_stats():
            """Get statistics about available evaluation tasks."""
            try:
                stats = self.task_library.get_genre_statistics()
                return jsonify(stats)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/recent_evaluations/<int:limit>')
        def recent_evaluations(limit: int):
            """Get recent evaluation results."""
            try:
                # Sort by timestamp (most recent first)
                sorted_results = sorted(
                    self.evaluator.results,
                    key=lambda x: x.timestamp,
                    reverse=True
                )

                recent = []
                for result in sorted_results[:limit]:
                    recent.append({
                        "evaluation_id": result.evaluation_id,
                        "model_name": result.model_name,
                        "genre": result.genre.value,
                        "task_description": result.task_description,
                        "success": result.success,
                        "processing_time": result.processing_time,
                        "overall_score": result.calculate_overall_score(
                            {name: metric.weight for name, metric in self.evaluator.metrics.items()}
                        ),
                        "timestamp": result.timestamp.isoformat(),
                        "metrics": result.metrics
                    })

                return jsonify({
                    "count": len(recent),
                    "evaluations": recent
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/notifications')
        def get_notifications():
            """Get recent notifications."""
            try:
                # Read notification log
                log_file = "notification_log.json"
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        notifications = json.load(f)
                else:
                    notifications = []

                # Return last 20 notifications
                recent_notifications = notifications[-20:]

                return jsonify({
                    "count": len(recent_notifications),
                    "notifications": recent_notifications
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/performance_chart/<model>')
        def performance_chart(model: str):
            """Generate performance chart for a model."""
            try:
                # Get recent evaluations for this model
                model_results = [
                    r for r in self.evaluator.results
                    if r.model_name == model and r.success
                ][-20:]  # Last 20 evaluations

                if not model_results:
                    return jsonify({"error": "No data available for this model"}), 404

                # Calculate scores over time
                scores = [
                    r.calculate_overall_score(
                        {name: metric.weight for name, metric in self.evaluator.metrics.items()}
                    )
                    for r in model_results
                ]

                timestamps = [r.timestamp.strftime('%m/%d %H:%M') for r in model_results]

                # Create simple chart data (in production, you'd use matplotlib/seaborn)
                chart_data = {
                    "labels": timestamps,
                    "scores": scores,
                    "model": model,
                    "average_score": sum(scores) / len(scores) if scores else 0
                }

                return jsonify(chart_data)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route('/api/export_results')
        def export_results():
            """Export evaluation results as JSON."""
            try:
                # Convert results to exportable format
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_evaluations": len(self.evaluator.results),
                    "results": []
                }

                for result in self.evaluator.results[-1000:]:  # Last 1000 results
                    export_data["results"].append({
                        "evaluation_id": result.evaluation_id,
                        "model_name": result.model_name,
                        "genre": result.genre.value,
                        "task_description": result.task_description,
                        "input_text": result.input_text,
                        "actual_output": result.actual_output,
                        "expected_output": result.expected_output,
                        "metrics": result.metrics,
                        "processing_time": result.processing_time,
                        "success": result.success,
                        "timestamp": result.timestamp.isoformat()
                    })

                # In a real implementation, you'd return the file
                # For now, return as JSON response
                return jsonify(export_data)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    def run(self, host: str = '0.0.0.0', port: int = 5001, debug: bool = True):
        """Run the evaluation dashboard."""
        print("🚀 Starting AI Evaluation Dashboard..."        print(f"📊 Access at: http://{host}:{port}")
        print("📈 Features: Real-time metrics, comparative analysis, performance tracking"
        self.app.run(host=host, port=port, debug=debug)


# Global dashboard instance
dashboard = EvaluationDashboard(evaluator, task_library)
