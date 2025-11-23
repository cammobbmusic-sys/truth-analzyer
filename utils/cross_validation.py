"""
Cross Validation - Compares outputs from multiple models using semantic similarity.
Provides consensus scoring and discrepancy detection.
"""

from typing import Dict, List, Tuple, Any
from .embeddings import cosine_similarity
from .metrics import consensus_score, coherence_score


def cross_validate(outputs: dict, threshold: float = 0.8) -> Tuple[List[str], List[Tuple[str, str, str, str]]]:
    """
    Compares outputs from multiple models and computes consensus scores.

    Returns verified statements and flagged discrepancies.

    Args:
        outputs: Dictionary mapping agent roles to their output text
        threshold: Similarity threshold for consensus (0-1)

    Returns:
        Tuple of (verified_statements, flagged_discrepancies)
        where flagged_discrepancies is list of (role1, role2, text1, text2) tuples
    """
    verified = []
    flagged = []
    roles = list(outputs.keys())

    for i in range(len(roles)):
        for j in range(i+1, len(roles)):
            role1, role2 = roles[i], roles[j]
            text1, text2 = outputs[role1], outputs[role2]

            try:
                sim = cosine_similarity(text1, text2)
                if sim >= threshold:
                    # Add the more detailed response to verified
                    verified.append(text1 if len(text1) >= len(text2) else text2)
                else:
                    flagged.append((role1, role2, text1, text2))
            except Exception as e:
                print(f"Warning: Could not calculate similarity between {role1} and {role2}: {e}")
                flagged.append((role1, role2, text1, text2))

    # Remove duplicates from verified
    verified = list(set(verified))

    return verified, flagged


def analyze_consensus(outputs: dict, threshold: float = 0.8) -> Dict[str, Any]:
    """
    Comprehensive consensus analysis for multiple agent outputs.

    Args:
        outputs: Dictionary mapping agent roles to their output text
        threshold: Similarity threshold for consensus

    Returns:
        Dictionary with consensus analysis results
    """
    verified, flagged = cross_validate(outputs, threshold)

    # Calculate agreement metrics
    total_pairs = len(list(outputs.keys())) * (len(outputs.keys()) - 1) // 2
    agreement_ratio = (total_pairs - len(flagged)) / total_pairs if total_pairs > 0 else 1.0

    # Use the new consensus_score function
    consensus_ratio = consensus_score(len(verified), len(outputs))

    # Calculate coherence across all outputs
    coherence = coherence_score(outputs)

    # Calculate confidence based on agreement and coherence
    confidence = min(1.0, (agreement_ratio * 0.6 + coherence * 0.4) * 1.2)

    return {
        "verified_statements": verified,
        "flagged_discrepancies": flagged,
        "agreement_score": agreement_ratio,
        "consensus_score": consensus_ratio,
        "coherence_score": coherence,
        "total_comparisons": total_pairs,
        "discrepancies_found": len(flagged),
        "confidence_score": confidence,
        "consensus_achieved": len(flagged) == 0 and len(verified) > 0
    }


def extract_common_statements(outputs: dict, min_agreement: int = 2) -> List[str]:
    """
    Extract statements that appear in multiple outputs (with some variation).

    Args:
        outputs: Dictionary mapping agent roles to their output text
        min_agreement: Minimum number of agents that must agree

    Returns:
        List of common statements found across multiple outputs
    """
    if len(outputs) < min_agreement:
        return []

    # Split outputs into sentences
    all_sentences = {}
    for role, text in outputs.items():
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        for sentence in sentences:
            if sentence not in all_sentences:
                all_sentences[sentence] = []
            all_sentences[sentence].append(role)

    # Find sentences that appear in multiple outputs
    common_statements = []
    for sentence, roles in all_sentences.items():
        if len(roles) >= min_agreement and len(sentence) > 20:  # Minimum length
            common_statements.append({
                "statement": sentence,
                "agreeing_roles": roles,
                "agreement_count": len(roles)
            })

    # Sort by agreement count (descending)
    common_statements.sort(key=lambda x: x["agreement_count"], reverse=True)

    return common_statements


def generate_consensus_report(outputs: dict, threshold: float = 0.8) -> str:
    """
    Generate a human-readable consensus report.

    Args:
        outputs: Dictionary mapping agent roles to their output text
        threshold: Similarity threshold for consensus

    Returns:
        Formatted consensus report
    """
    analysis = analyze_consensus(outputs, threshold)
    common_statements = extract_common_statements(outputs)

    report_lines = [
        "CROSS-VALIDATION CONSENSUS REPORT",
        "=" * 40,
        f"Total Agents: {len(outputs)}",
        f"Agreement Score: {analysis['agreement_score']:.2f}",
        f"Confidence Score: {analysis['confidence_score']:.2f}",
        f"Consensus Achieved: {analysis['consensus_achieved']}",
        f"Discrepancies Found: {analysis['discrepancies_found']}",
        "",
        "VERIFIED STATEMENTS:"
    ]

    if analysis['verified_statements']:
        for i, statement in enumerate(analysis['verified_statements'][:5], 1):  # Limit to 5
            report_lines.append(f"{i}. {statement[:100]}...")
    else:
        report_lines.append("No statements reached consensus threshold.")

    if common_statements:
        report_lines.extend([
            "",
            "COMMON STATEMENTS:"
        ])
        for stmt in common_statements[:3]:  # Limit to 3
            report_lines.extend([
                f"• {stmt['statement'][:80]}...",
                f"  Agreed by: {', '.join(stmt['agreeing_roles'])}",
                ""
            ])

    if analysis['flagged_discrepancies']:
        report_lines.extend([
            "",
            "FLAGGED DISCREPANCIES:"
        ])
        for role1, role2, text1, text2 in analysis['flagged_discrepancies'][:3]:  # Limit to 3
            report_lines.extend([
                f"• {role1} vs {role2}:",
                f"  {role1}: {text1[:50]}...",
                f"  {role2}: {text2[:50]}...",
                ""
            ])

    return "\n".join(report_lines)
