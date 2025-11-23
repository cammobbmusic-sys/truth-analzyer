"""
Cross Validator - Compares outputs from multiple AI models.
Implements consensus scoring and agreement analysis.
"""

import re
import json
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

from agents.model_agent import ModelResponse
from config import Config
from utils.embeddings import EmbeddingService


class CrossValidator:
    """Handles cross-validation of multiple AI model responses."""

    def __init__(self, config: Config):
        self.config = config
        self.embedding_service = EmbeddingService()

    def validate_responses(self, responses: List[ModelResponse]) -> Dict[str, Any]:
        """
        Validate and compare multiple model responses.

        Args:
            responses: List of model responses to compare

        Returns:
            Validation results with consensus analysis
        """
        if len(responses) < 2:
            return self._handle_single_response(responses[0] if responses else None)

        # Extract key information from responses
        extracted_info = [self._extract_response_info(resp) for resp in responses]

        # Calculate agreement metrics
        agreement_metrics = self._calculate_agreement_metrics(extracted_info)

        # Find consensus answer
        consensus_result = self._find_consensus_answer(extracted_info, agreement_metrics)

        # Calculate confidence scores
        confidence_analysis = self._analyze_confidence_distribution(responses)

        return {
            "consensus_achieved": consensus_result["achieved"],
            "consensus_answer": consensus_result["answer"],
            "agreement_score": agreement_metrics["overall_agreement"],
            "confidence_analysis": confidence_analysis,
            "individual_scores": agreement_metrics["individual_scores"],
            "disagreements": agreement_metrics["disagreements"],
            "validation_metadata": {
                "num_responses": len(responses),
                "models_used": [resp.model_name for resp in responses],
                "threshold_used": self.config.verification.consensus_threshold
            }
        }

    def _handle_single_response(self, response: Optional[ModelResponse]) -> Dict[str, Any]:
        """Handle validation when only one response is available."""
        if not response:
            return {
                "consensus_achieved": False,
                "consensus_answer": None,
                "agreement_score": 0.0,
                "error": "No responses to validate"
            }

        return {
            "consensus_achieved": True,
            "consensus_answer": {
                "content": response.content,
                "confidence": response.confidence,
                "source": "single_model"
            },
            "agreement_score": 1.0,  # Single response always agrees with itself
            "confidence_analysis": {
                "average_confidence": response.confidence,
                "confidence_std": 0.0,
                "high_confidence_count": 1 if response.confidence > 0.8 else 0
            }
        }

    def _extract_response_info(self, response: ModelResponse) -> Dict[str, Any]:
        """
        Extract key information from a model response for comparison.

        Args:
            response: Model response to analyze

        Returns:
            Extracted information dictionary
        """
        content = response.content

        # Extract direct answer
        direct_answer = self._extract_direct_answer(content)

        # Extract confidence if not already provided
        confidence = response.confidence
        if confidence == 0.8:  # Default value, try to extract from content
            confidence = self._extract_confidence_from_text(content)

        # Extract key points and evidence
        key_points = self._extract_key_points(content)

        # Extract assumptions and limitations
        assumptions = self._extract_assumptions(content)

        return {
            "model_name": response.model_name,
            "direct_answer": direct_answer,
            "confidence": confidence,
            "key_points": key_points,
            "assumptions": assumptions,
            "full_content": content,
            "metadata": response.metadata
        }

    def _extract_direct_answer(self, content: str) -> Optional[str]:
        """Extract the direct answer from response content."""
        # Look for common answer patterns
        patterns = [
            r"Answer:\s*(.+?)(?:\n|$)",
            r"Final Answer:\s*(.+?)(?:\n|$)",
            r"Conclusion:\s*(.+?)(?:\n|$)",
            r"Therefore,\s*(.+?)(?:\n|$)",
            r"So,\s*(.+?)(?:\n|$)"
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).strip()
                # Clean up the answer
                answer = re.sub(r'[.\s]+$', '', answer)  # Remove trailing dots/spaces
                return answer

        # If no clear answer found, try to get the first substantial sentence
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and not sentence.lower().startswith(('i think', 'perhaps', 'maybe')):
                return sentence

        return None

    def _extract_confidence_from_text(self, content: str) -> float:
        """Extract confidence score from text content."""
        # Look for explicit confidence mentions
        confidence_patterns = [
            r"confidence[:\s]+(\d*\.?\d+)",
            r"certainty[:\s]+(\d*\.?\d+)",
            r"surety[:\s]+(\d*\.?\d+)",
            r"(\d*\.?\d+)%\s+sure",
            r"(\d*\.?\d+)%\s+certain"
        ]

        for pattern in confidence_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    conf = float(match.group(1))
                    # Convert percentage to decimal if needed
                    if conf > 1:
                        conf /= 100.0
                    return min(conf, 1.0)
                except ValueError:
                    continue

        # Estimate confidence based on language patterns
        confidence_indicators = {
            0.9: ['definitely', 'absolutely', 'certainly', 'clearly'],
            0.8: ['very likely', 'highly confident', 'strong evidence'],
            0.7: ['likely', 'probably', 'good chance'],
            0.6: ['possibly', 'maybe', 'somewhat'],
            0.5: ['unclear', 'uncertain', 'not sure'],
            0.3: ['unlikely', 'doubtful', 'questionable']
        }

        content_lower = content.lower()
        for conf_score, indicators in confidence_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                return conf_score

        return 0.7  # Default moderate confidence

    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points and evidence from content."""
        points = []

        # Split into sentences
        sentences = re.split(r'[.!?]+', content)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15 and len(sentence) < 200:  # Reasonable length
                # Check if it looks like a key point
                if self._is_key_point(sentence):
                    points.append(sentence)

        return points[:10]  # Limit to top 10 points

    def _is_key_point(self, sentence: str) -> bool:
        """Determine if a sentence appears to be a key point."""
        # Skip obvious non-points
        skip_indicators = [
            'i think', 'perhaps', 'maybe', 'in my opinion',
            'let me', 'first', 'second', 'next', 'finally',
            'however', 'therefore', 'thus', 'so', 'because'
        ]

        sentence_lower = sentence.lower()
        if any(indicator in sentence_lower for indicator in skip_indicators):
            return False

        # Look for substantive content
        word_count = len(sentence.split())
        if word_count < 3:
            return False

        # Prefer sentences with specific information
        specific_indicators = [
            'because', 'due to', 'according to', 'evidence shows',
            'research indicates', 'data suggests', 'studies show'
        ]

        return any(indicator in sentence_lower for indicator in specific_indicators)

    def _extract_assumptions(self, content: str) -> List[str]:
        """Extract assumptions and limitations mentioned."""
        assumptions = []

        # Look for assumption indicators
        assumption_patterns = [
            r"assum(?:e|ing|es|ed)[:\s]*(.+?)(?:\n|$)",
            r"assumption[:\s]*(.+?)(?:\n|$)",
            r"limitation[:\s]*(.+?)(?:\n|$)",
            r"however[:\s]*(.+?)(?:\n|$)",
            r"but[:\s]*(.+?)(?:\n|$)"
        ]

        for pattern in assumption_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
            assumptions.extend(matches)

        return list(set(assumptions))  # Remove duplicates

    def _calculate_agreement_metrics(self, extracted_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate agreement metrics between responses."""
        if len(extracted_info) < 2:
            return {"overall_agreement": 1.0, "individual_scores": [], "disagreements": []}

        # Calculate pairwise agreements
        pairwise_scores = []
        disagreements = []

        for i in range(len(extracted_info)):
            for j in range(i + 1, len(extracted_info)):
                score, disagreement = self._calculate_pairwise_agreement(
                    extracted_info[i], extracted_info[j]
                )
                pairwise_scores.append(score)
                if disagreement:
                    disagreements.append(disagreement)

        # Calculate overall agreement
        overall_agreement = sum(pairwise_scores) / len(pairwise_scores) if pairwise_scores else 0.0

        # Calculate individual model agreement scores
        individual_scores = []
        for i, info in enumerate(extracted_info):
            model_scores = [
                self._calculate_pairwise_agreement(info, other_info)[0]
                for j, other_info in enumerate(extracted_info)
                if i != j
            ]
            avg_score = sum(model_scores) / len(model_scores) if model_scores else 1.0
            individual_scores.append({
                "model": info["model_name"],
                "agreement_score": avg_score
            })

        return {
            "overall_agreement": overall_agreement,
            "individual_scores": individual_scores,
            "disagreements": disagreements[:5]  # Limit to top 5 disagreements
        }

    def _calculate_pairwise_agreement(
        self,
        info1: Dict[str, Any],
        info2: Dict[str, Any]
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """Calculate agreement score between two responses."""
        scores = []

        # Compare direct answers
        if info1["direct_answer"] and info2["direct_answer"]:
            answer_similarity = self._calculate_answer_similarity(
                info1["direct_answer"], info2["direct_answer"]
            )
            scores.append(answer_similarity)
        else:
            scores.append(0.0)  # No direct answers to compare

        # Compare key points overlap
        if info1["key_points"] and info2["key_points"]:
            points_overlap = self._calculate_points_overlap(
                info1["key_points"], info2["key_points"]
            )
            scores.append(points_overlap)

        # Compare confidence levels
        confidence_diff = abs(info1["confidence"] - info2["confidence"])
        confidence_agreement = 1.0 - (confidence_diff / 2.0)  # Normalize difference
        scores.append(confidence_agreement)

        # Calculate weighted average
        if scores:
            agreement_score = sum(scores) / len(scores)
        else:
            agreement_score = 0.0

        # Identify major disagreements
        disagreement = None
        if agreement_score < 0.5:
            disagreement = {
                "models": [info1["model_name"], info2["model_name"]],
                "issue": "significant_disagreement",
                "score": agreement_score,
                "answers": [info1["direct_answer"], info2["direct_answer"]]
            }

        return agreement_score, disagreement

    def _calculate_answer_similarity(self, answer1: str, answer2: str) -> float:
        """Calculate similarity between two answers."""
        # Simple text similarity
        text1 = answer1.lower().strip()
        text2 = answer2.lower().strip()

        # Exact match
        if text1 == text2:
            return 1.0

        # Check for semantic similarity using embeddings if available
        try:
            return self.embedding_service.calculate_similarity(text1, text2)
        except Exception:
            # Fallback to simple string similarity
            return self._simple_text_similarity(text1, text2)

    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        # Jaccard similarity of words
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _calculate_points_overlap(self, points1: List[str], points2: List[str]) -> float:
        """Calculate overlap between key points lists."""
        if not points1 or not points2:
            return 0.0

        # Calculate pairwise similarities
        similarities = []
        for p1 in points1:
            for p2 in points2:
                sim = self._simple_text_similarity(p1, p2)
                similarities.append(sim)

        # Return average similarity
        return sum(similarities) / len(similarities) if similarities else 0.0

    def _find_consensus_answer(
        self,
        extracted_info: List[Dict[str, Any]],
        agreement_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Find consensus answer from multiple responses."""
        # Check if consensus threshold is met
        overall_agreement = agreement_metrics["overall_agreement"]
        threshold = self.config.verification.consensus_threshold

        if overall_agreement >= threshold:
            # Find most common answer
            answers = [info["direct_answer"] for info in extracted_info if info["direct_answer"]]
            if answers:
                consensus_answer = self._find_most_common_answer(answers)

                # Calculate average confidence
                confidences = [info["confidence"] for info in extracted_info]
                avg_confidence = sum(confidences) / len(confidences)

                return {
                    "achieved": True,
                    "answer": {
                        "content": consensus_answer,
                        "confidence": avg_confidence,
                        "supporting_models": len(answers),
                        "method": "majority_vote"
                    }
                }

        # No consensus achieved
        return {
            "achieved": False,
            "answer": None,
            "reason": f"Agreement score {overall_agreement:.2f} below threshold {threshold}"
        }

    def _find_most_common_answer(self, answers: List[str]) -> str:
        """Find the most common answer among multiple responses."""
        # Group similar answers
        answer_groups = defaultdict(list)

        for answer in answers:
            # Find best match in existing groups
            best_match = None
            best_similarity = 0.0

            for group_rep in answer_groups.keys():
                similarity = self._simple_text_similarity(answer, group_rep)
                if similarity > best_similarity and similarity > 0.7:  # Similarity threshold
                    best_similarity = similarity
                    best_match = group_rep

            if best_match:
                answer_groups[best_match].append(answer)
            else:
                answer_groups[answer].append(answer)

        # Find group with most answers
        best_group = max(answer_groups.items(), key=lambda x: len(x[1]))
        return best_group[0]

    def _analyze_confidence_distribution(self, responses: List[ModelResponse]) -> Dict[str, Any]:
        """Analyze the distribution of confidence scores."""
        confidences = [resp.confidence for resp in responses]

        if not confidences:
            return {"average_confidence": 0.0, "confidence_std": 0.0, "high_confidence_count": 0}

        avg_confidence = sum(confidences) / len(confidences)

        # Calculate standard deviation
        variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
        confidence_std = variance ** 0.5

        # Count high confidence responses
        high_confidence_count = sum(1 for c in confidences if c > 0.8)

        return {
            "average_confidence": avg_confidence,
            "confidence_std": confidence_std,
            "high_confidence_count": high_confidence_count,
            "confidence_range": f"{min(confidences):.2f} - {max(confidences):.2f}"
        }
