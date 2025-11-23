"""
Utils module for the Multi-AI System.
Contains logging, embeddings, and metrics utilities.
"""

from .logger import Logger
from .embeddings import EmbeddingService, cosine_similarity, pairwise_similarity
from .metrics import MetricsCalculator, consensus_score, diversity_score, coherence_score
from .cross_validation import cross_validate, analyze_consensus, generate_consensus_report
from .idea_aggregation import (
    aggregate_ideas,
    rank_ideas_by_novelty,
    generate_idea_report,
    find_contradictory_ideas
)

__all__ = [
    "Logger",
    "EmbeddingService",
    "cosine_similarity",
    "pairwise_similarity",
    "MetricsCalculator",
    "consensus_score",
    "diversity_score",
    "coherence_score",
    "cross_validate",
    "analyze_consensus",
    "generate_consensus_report",
    "aggregate_ideas",
    "rank_ideas_by_novelty",
    "generate_idea_report",
    "find_contradictory_ideas"
]
