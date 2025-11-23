"""
Idea Aggregation - Combines and clusters diverse outputs from multiple agents.
Provides similarity-based clustering and novelty ranking.
"""

from typing import Dict, List, Tuple, Any
from collections import defaultdict
import re

from .embeddings import cosine_similarity
from .metrics import diversity_score


def aggregate_ideas(outputs: dict) -> List[Tuple[str, List[str]]]:
    """
    Combine diverse outputs into clusters and rank by novelty.

    Args:
        outputs: Dictionary mapping agent roles to their output text

    Returns:
        List of (cluster_key, ideas_list) tuples, sorted by cluster size
    """
    all_ideas = []

    # Extract ideas from outputs (split by newlines and clean up)
    for role, text in outputs.items():
        ideas = _extract_ideas_from_text(text)
        for idea in ideas:
            all_ideas.append((idea.strip(), role))

    if not all_ideas:
        return []

    # Cluster ideas by semantic similarity
    clusters = _cluster_ideas_semantic(all_ideas)

    # Convert to the expected format and sort by cluster size
    cluster_items = [(key, ideas) for key, ideas in clusters.items()]
    return sorted(cluster_items, key=lambda x: len(x[1]), reverse=True)


def _extract_ideas_from_text(text: str) -> List[str]:
    """Extract individual ideas from agent output text."""
    # Split by newlines and common separators
    separators = r'[•\-\*\d+\.\n]'
    ideas = re.split(separators, text)

    # Clean up and filter
    cleaned_ideas = []
    for idea in ideas:
        idea = idea.strip()
        # Filter out very short items and common non-idea text
        if len(idea) > 10 and not _is_noise_text(idea):
            cleaned_ideas.append(idea)

    return cleaned_ideas


def _is_noise_text(text: str) -> bool:
    """Check if text is likely noise rather than an actual idea."""
    noise_indicators = [
        'here are', 'the following', 'consider', 'think about',
        'one approach', 'another way', 'finally', 'summary',
        'conclusion', 'overall', 'therefore', 'thus'
    ]

    text_lower = text.lower()
    return any(indicator in text_lower for indicator in noise_indicators)


def _cluster_ideas_semantic(ideas_with_roles: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """
    Cluster ideas using semantic similarity.

    Args:
        ideas_with_roles: List of (idea_text, role) tuples

    Returns:
        Dictionary mapping cluster representatives to lists of ideas
    """
    if not ideas_with_roles:
        return {}

    clusters = {}

    for idea_text, role in ideas_with_roles:
        # Find best cluster match
        best_cluster = None
        best_similarity = 0.0

        for cluster_rep in clusters.keys():
            try:
                similarity = cosine_similarity(idea_text, cluster_rep)
                if similarity > best_similarity and similarity > 0.6:  # Similarity threshold
                    best_similarity = similarity
                    best_cluster = cluster_rep
            except Exception:
                continue

        if best_cluster:
            # Add to existing cluster
            clusters[best_cluster].append(idea_text)
        else:
            # Create new cluster
            clusters[idea_text] = [idea_text]

    return clusters


def rank_ideas_by_novelty(clusters: Dict[str, List[str]], base_ideas: List[str] = None) -> List[Dict[str, Any]]:
    """
    Rank clustered ideas by novelty and diversity.

    Args:
        clusters: Dictionary of clustered ideas
        base_ideas: Optional list of base ideas to compare against for novelty

    Returns:
        List of ranked idea clusters with novelty scores
    """
    ranked_ideas = []

    for cluster_rep, ideas in clusters.items():
        # Calculate novelty score (inverse of similarity to base ideas)
        novelty_score = _calculate_novelty_score(cluster_rep, base_ideas or [])

        # Calculate diversity within cluster
        diversity_score = _calculate_cluster_diversity(ideas)

        # Calculate overall quality score
        quality_score = (novelty_score * 0.6 + diversity_score * 0.4)

        ranked_ideas.append({
            "cluster_representative": cluster_rep,
            "ideas": ideas,
            "cluster_size": len(ideas),
            "novelty_score": novelty_score,
            "diversity_score": diversity_score,
            "quality_score": quality_score
        })

    # Sort by quality score (descending)
    ranked_ideas.sort(key=lambda x: x["quality_score"], reverse=True)

    return ranked_ideas


def _calculate_novelty_score(idea: str, base_ideas: List[str]) -> float:
    """Calculate novelty score relative to base ideas."""
    if not base_ideas:
        return 0.8  # Default high novelty if no base ideas

    max_similarity = 0.0
    for base_idea in base_ideas:
        try:
            similarity = cosine_similarity(idea, base_idea)
            max_similarity = max(max_similarity, similarity)
        except Exception:
            continue

    # Novelty is inverse of similarity (higher similarity = lower novelty)
    novelty = 1.0 - max_similarity

    return novelty


def _calculate_cluster_diversity(ideas: List[str]) -> float:
    """Calculate diversity within a cluster of ideas."""
    if len(ideas) <= 1:
        return 0.0

    total_similarity = 0.0
    comparison_count = 0

    # Calculate average pairwise similarity
    for i in range(len(ideas)):
        for j in range(i + 1, len(ideas)):
            try:
                similarity = cosine_similarity(ideas[i], ideas[j])
                total_similarity += similarity
                comparison_count += 1
            except Exception:
                continue

    if comparison_count == 0:
        return 0.0

    avg_similarity = total_similarity / comparison_count

    # Diversity is inverse of average similarity
    diversity = 1.0 - avg_similarity

    return diversity


def generate_idea_report(ranked_ideas: List[Dict[str, Any]], top_n: int = 5) -> str:
    """
    Generate a human-readable idea aggregation report.

    Args:
        ranked_ideas: List of ranked idea clusters
        top_n: Number of top clusters to include

    Returns:
        Formatted report string
    """
    overall_diversity = diversity_score(ranked_ideas)

    report_lines = [
        "IDEA AGGREGATION REPORT",
        "=" * 40,
        f"Total Clusters: {len(ranked_ideas)}",
        f"Overall Diversity Score: {overall_diversity}",
        f"Showing Top {min(top_n, len(ranked_ideas))} Clusters",
        "",
        "TOP IDEAS BY QUALITY SCORE:"
    ]

    for i, cluster in enumerate(ranked_ideas[:top_n], 1):
        report_lines.extend([
            f"",
            f"{i}. CLUSTER (Quality: {cluster['quality_score']:.2f})",
            f"   Novelty: {cluster['novelty_score']:.2f} | Diversity: {cluster['diversity_score']:.2f}",
            f"   Size: {cluster['cluster_size']} ideas",
            f"   Representative: {cluster['cluster_representative'][:80]}...",
        ])

        if len(cluster['ideas']) > 1:
            report_lines.append("   Related Ideas:")
            for idea in cluster['ideas'][1:3]:  # Show up to 2 additional ideas
                report_lines.append(f"   • {idea[:60]}...")

    if len(ranked_ideas) > top_n:
        report_lines.append(f"\n... and {len(ranked_ideas) - top_n} more clusters")

    return "\n".join(report_lines)


def find_contradictory_ideas(outputs: dict) -> List[Dict[str, Any]]:
    """
    Find potentially contradictory ideas across different agents.

    Args:
        outputs: Dictionary mapping agent roles to their output text

    Returns:
        List of contradictory idea pairs with context
    """
    contradictions = []
    roles = list(outputs.keys())

    # Extract ideas from each agent
    agent_ideas = {}
    for role, text in outputs.items():
        agent_ideas[role] = _extract_ideas_from_text(text)

    # Look for contradictions between agents
    contradiction_indicators = [
        ("benefits", "drawbacks"),
        ("advantages", "disadvantages"),
        ("positive", "negative"),
        ("good", "bad"),
        ("effective", "ineffective"),
        ("useful", "useless")
    ]

    for i in range(len(roles)):
        for j in range(i + 1, len(roles)):
            role1, role2 = roles[i], roles[j]
            ideas1, ideas2 = agent_ideas[role1], agent_ideas[role2]

            for idea1 in ideas1:
                for idea2 in ideas2:
                    if _ideas_are_contradictory(idea1, idea2, contradiction_indicators):
                        contradictions.append({
                            "agent1": role1,
                            "agent2": role2,
                            "idea1": idea1,
                            "idea2": idea2,
                            "contradiction_type": _identify_contradiction_type(idea1, idea2, contradiction_indicators)
                        })

    return contradictions


def _ideas_are_contradictory(idea1: str, idea2: str, indicators: List[Tuple[str, str]]) -> bool:
    """Check if two ideas express contradictions."""
    idea1_lower = idea1.lower()
    idea2_lower = idea2.lower()

    for pos, neg in indicators:
        if ((pos in idea1_lower and neg in idea2_lower) or
            (neg in idea1_lower and pos in idea2_lower)):
            return True

    return False


def _identify_contradiction_type(idea1: str, idea2: str, indicators: List[Tuple[str, str]]) -> str:
    """Identify the type of contradiction between ideas."""
    idea1_lower = idea1.lower()
    idea2_lower = idea2.lower()

    for pos, neg in indicators:
        if pos in idea1_lower and neg in idea2_lower:
            return f"{pos} vs {neg}"
        if neg in idea1_lower and pos in idea2_lower:
            return f"{neg} vs {pos}"

    return "general_contradiction"
