#!/usr/bin/env python3
"""
AI Model Evaluation CLI

Command-line interface for running comprehensive AI model evaluations,
comparative analysis, and managing the evaluation framework.
"""

import argparse
import json
import sys
from typing import List, Optional
from datetime import datetime

from evaluation_framework import evaluator, EvaluationGenre, StatisticalAnalysis
from evaluation_tasks import task_library
from evaluation_dashboard import dashboard


def run_evaluation_suite(models: List[str], genres: Optional[List[str]] = None,
                        tasks_per_genre: int = 5, output_file: Optional[str] = None):
    """Run a comprehensive evaluation suite."""
    print("🚀 Starting AI Model Evaluation Suite")
    print(f"📊 Models: {', '.join(models)}")
    print(f"🎯 Genres: {genres or 'ALL'}")
    print(f"📝 Tasks per genre: {tasks_per_genre}")
    print("-" * 60)

    # Convert genre strings to enums
    genre_enums = []
    if genres:
        for genre_str in genres:
            try:
                genre_enums.append(EvaluationGenre(genre_str))
            except ValueError:
                print(f"⚠️  Invalid genre: {genre_str}. Skipping.")
    else:
        genre_enums = list(EvaluationGenre)

    # Run evaluation
    start_time = datetime.now()
    results = evaluator.run_evaluation_suite(models, genre_enums, tasks_per_genre)
    end_time = datetime.now()

    # Print summary
    print("📊 EVALUATION COMPLETE")
    print(f"⏱️  Duration: {(end_time - start_time).total_seconds():.2f} seconds")
    print(f"📈 Total Evaluations: {results['total_evaluations']}")

    # Print results summary
    model_stats = {}
    for result in results['results']:
        model = result['model_name']
        if model not in model_stats:
            model_stats[model] = {'total': 0, 'successful': 0}
        model_stats[model]['total'] += 1
        if result.get('success', False):
            model_stats[model]['successful'] += 1

    print("📋 MODEL PERFORMANCE SUMMARY:")
    for model, stats in model_stats.items():
        success_rate = (stats['successful'] / stats['total']) * 100
        print(f"  {model}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%)")

    # Save results if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {output_file}")

    print("\n✅ Evaluation suite completed successfully!")
    return results


def compare_models_cli(model_a: str, model_b: str, genres: Optional[List[str]] = None):
    """Compare two models via command line."""
    print(f"⚖️  Comparing {model_a} vs {model_b}")
    print("-" * 60)

    genres_to_compare = []
    if genres:
        for genre_str in genres:
            try:
                genres_to_compare.append(EvaluationGenre(genre_str))
            except ValueError:
                print(f"⚠️  Invalid genre: {genre_str}. Skipping.")
    else:
        genres_to_compare = list(EvaluationGenre)

    print(f"🎯 Comparing across {len(genres_to_compare)} genres")

    comparisons = {}
    for genre in genres_to_compare:
        try:
            comparison = evaluator.compare_models(genre, model_a, model_b)
            if "error" not in comparison:
                comparisons[genre.value] = comparison
                winner = comparison['winner']
                diff = comparison['score_difference']
                print(f"  {genre.value}: {winner} wins by {diff:.3f}")
        except Exception as e:
            print(f"  {genre.value}: Error - {e}")

    # Overall summary
    if comparisons:
        genre_wins = {'tie': 0, model_a: 0, model_b: 0}
        for comparison in comparisons.values():
            winner = comparison['winner']
            genre_wins[winner] += 1

        print("🏆 OVERALL RESULTS:")
        print(f"  {model_a}: {genre_wins[model_a]} genre wins")
        print(f"  {model_b}: {genre_wins[model_b]} genre wins")
        print(f"  Ties: {genre_wins['tie']}")

        overall_winner = max(genre_wins.items(), key=lambda x: x[1])
        if overall_winner[0] != 'tie':
            print(f"  🏅 Overall Winner: {overall_winner[0]}")
        else:
            print("  🤝 Overall Result: Tie")
    else:
        print("❌ No valid comparisons could be performed")

    return comparisons


def show_evaluation_report(model: Optional[str] = None, genre: Optional[str] = None):
    """Display evaluation report."""
    print("📊 AI MODEL EVALUATION REPORT")
    print("=" * 60)

    genre_enum = None
    if genre:
        try:
            genre_enum = EvaluationGenre(genre)
        except ValueError:
            print(f"❌ Invalid genre: {genre}")
            return

    report = evaluator.generate_evaluation_report(model, genre_enum)

    print(f"📅 Generated: {report['generated_at'][:19]}")
    print(f"📈 Total Evaluations: {report['total_evaluations']}")
    print(f"🤖 Models Evaluated: {len(report['models_evaluated'])}")
    print(f"🎯 Genres Evaluated: {len(report['genres_evaluated'])}")

    if report['performance_summary']:
        print("📋 MODEL PERFORMANCE SUMMARY:")
        print(f"{'Model':<15} {'Genres':<8} {'Avg Score':<10} {'Samples':<8}")
        print("-" * 50)

        for model_name, genres in report['performance_summary'].items():
            total_genres = len(genres)
            avg_score = sum(g['average_score'] for g in genres.values()) / total_genres
            total_samples = sum(g['sample_size'] for g in genres.values())

            score_indicator = "🟢" if avg_score > 0.8 else "🟡" if avg_score > 0.6 else "🔴"
            print(f"{model_name:<15} {total_genres:<8} {score_indicator} {avg_score:.3f}     {total_samples:<8}")

    if report['recommendations']:
        print("💡 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  • Use {rec['recommended_model']} for {rec['genre']} tasks")
            print(f"    Expected score: {rec['expected_score']:.3f}")

    print("\n✅ Report generation complete!")


def show_task_library_stats():
    """Display task library statistics."""
    print("📚 EVALUATION TASK LIBRARY STATISTICS")
    print("=" * 60)

    stats = task_library.get_genre_statistics()

    print(f"{'Genre':<20} {'Total':<8} {'Easy':<8} {'Medium':<8} {'Hard':<8}")
    print("-" * 60)

    for genre, genre_stats in stats.items():
        genre_name = genre.replace('_', ' ').title()
        print(f"{genre_name:<20} {genre_stats['total']:<8} {genre_stats['easy']:<8} {genre_stats['medium']:<8} {genre_stats['hard']:<8}")

    total_tasks = sum(sum(g.values()) for g in stats.values())
    print("-" * 60)
    print(f"{'TOTAL':<20} {total_tasks}")

    print("✅ Task library analysis complete!")


def start_dashboard(host: str = '0.0.0.0', port: int = 5001):
    """Start the evaluation dashboard."""
    print("🚀 Starting AI Evaluation Dashboard...")
    print(f"📊 Access at: http://{host}:{port}")
    print("📈 Features: Real-time metrics, comparative analysis, performance tracking")
    try:
        dashboard.run(host=host, port=port, debug=False)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Model Evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full evaluation suite
  python evaluation_cli.py evaluate --models groq-llama openrouter-claude cohere-command-nightly

  # Compare specific models
  python evaluation_cli.py compare groq-llama openrouter-claude --genres factual_qa mathematical

  # Show evaluation report
  python evaluation_cli.py report --model groq-llama

  # Start dashboard
  python evaluation_cli.py dashboard

  # Show task library stats
  python evaluation_cli.py tasks
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run evaluation suite')
    eval_parser.add_argument('--models', nargs='+', required=True,
                            help='Models to evaluate')
    eval_parser.add_argument('--genres', nargs='*',
                            choices=[e.value for e in EvaluationGenre],
                            help='Genres to evaluate (default: all)')
    eval_parser.add_argument('--tasks-per-genre', type=int, default=5,
                            help='Tasks per genre (default: 5)')
    eval_parser.add_argument('--output', type=str,
                            help='Output file for results')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two models')
    compare_parser.add_argument('model_a', help='First model to compare')
    compare_parser.add_argument('model_b', help='Second model to compare')
    compare_parser.add_argument('--genres', nargs='*',
                               choices=[e.value for e in EvaluationGenre],
                               help='Genres to compare (default: all)')

    # Report command
    report_parser = subparsers.add_parser('report', help='Show evaluation report')
    report_parser.add_argument('--model', help='Filter by specific model')
    report_parser.add_argument('--genre', choices=[e.value for e in EvaluationGenre],
                              help='Filter by specific genre')

    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Start evaluation dashboard')
    dashboard_parser.add_argument('--host', default='0.0.0.0', help='Host address')
    dashboard_parser.add_argument('--port', type=int, default=5001, help='Port number')

    # Tasks command
    subparsers.add_parser('tasks', help='Show task library statistics')

    args = parser.parse_args()

    if args.command == 'evaluate':
        run_evaluation_suite(
            models=args.models,
            genres=args.genres,
            tasks_per_genre=args.tasks_per_genre,
            output_file=args.output
        )

    elif args.command == 'compare':
        compare_models_cli(args.model_a, args.model_b, args.genres)

    elif args.command == 'report':
        show_evaluation_report(args.model, args.genre)

    elif args.command == 'dashboard':
        start_dashboard(args.host, args.port)

    elif args.command == 'tasks':
        show_task_library_stats()

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
