"""
Conflict Resolver - Merges conflicting outputs and assigns confidence scores.
Handles cases where models disagree and provides unified responses.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from agents.model_agent import ModelResponse
from config import Config
from utils.embeddings import EmbeddingService


class ConflictResolver:
    """Resolves conflicts between multiple AI model responses."""

    def __init__(self, config: Config):
        self.config = config
        self.embedding_service = EmbeddingService()

    def resolve_conflicts(
        self,
        responses: List[ModelResponse],
        validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve conflicts between multiple model responses.

        Args:
            responses: List of conflicting model responses
            validation_results: Results from cross-validation

        Returns:
            Resolved answer with confidence assessment
        """
        if len(responses) == 1:
            return self._format_single_response(responses[0])

        # Analyze the nature of conflicts
        conflict_analysis = self._analyze_conflicts(responses, validation_results)

        # Choose resolution method based on conflict type
        method = self.config.verification.conflict_resolution_method

        if method == "weighted_average":
            resolved_answer = self._resolve_weighted_average(responses, conflict_analysis)
        elif method == "majority_vote":
            resolved_answer = self._resolve_majority_vote(responses, conflict_analysis)
        elif method == "expert_consensus":
            resolved_answer = self._resolve_expert_consensus(responses, conflict_analysis)
        else:
            resolved_answer = self._resolve_weighted_average(responses, conflict_analysis)

        # Add conflict context to the resolved answer
        resolved_answer["conflict_analysis"] = conflict_analysis

        return resolved_answer

    def _analyze_conflicts(
        self,
        responses: List[ModelResponse],
        validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the nature and severity of conflicts."""
        # Extract key information
        answers = []
        confidences = []
        key_points = []

        for resp in responses:
            # Extract direct answer
            direct_answer = self._extract_direct_answer(resp.content)
            answers.append(direct_answer)

            confidences.append(resp.confidence)

            # Extract key points
            points = self._extract_key_points(resp.content)
            key_points.append(points)

        # Categorize conflict types
        conflict_types = self._categorize_conflicts(answers, key_points)

        # Assess conflict severity
        severity_score = self._calculate_conflict_severity(
            answers, key_points, confidences, validation_results
        )

        return {
            "conflict_types": conflict_types,
            "severity_score": severity_score,
            "num_responses": len(responses),
            "confidence_distribution": self._analyze_confidence_distribution(confidences),
            "answer_diversity": self._calculate_answer_diversity(answers),
            "key_agreements": self._find_key_agreements(key_points)
        }

    def _categorize_conflicts(
        self,
        answers: List[Optional[str]],
        key_points: List[List[str]]
    ) -> List[str]:
        """Categorize the types of conflicts present."""
        conflict_types = []

        # Check for direct answer conflicts
        valid_answers = [a for a in answers if a is not None]
        if len(set(valid_answers)) > 1:
            conflict_types.append("direct_answer_disagreement")

        # Check for partial agreements
        if len(valid_answers) > 1 and len(set(valid_answers)) < len(valid_answers):
            conflict_types.append("partial_answer_agreement")

        # Check for missing answers
        missing_count = sum(1 for a in answers if a is None)
        if missing_count > 0:
            conflict_types.append("incomplete_responses")

        # Check for key points conflicts
        if self._have_conflicting_key_points(key_points):
            conflict_types.append("evidence_conflict")

        # Check for complementary information
        if self._have_complementary_key_points(key_points):
            conflict_types.append("complementary_information")

        return conflict_types

    def _have_conflicting_key_points(self, key_points_lists: List[List[str]]) -> bool:
        """Check if key points from different models conflict."""
        # Simple heuristic: if we have very different key points
        all_points = []
        for points in key_points_lists:
            all_points.extend(points)

        if len(all_points) < 4:  # Too few points to analyze
            return False

        # Check for contradictory statements (very basic)
        contradictions = 0
        for i, points1 in enumerate(key_points_lists):
            for j, points2 in enumerate(key_points_lists):
                if i != j:
                    contradictions += self._count_contradictions(points1, points2)

        return contradictions > 2  # Arbitrary threshold

    def _count_contradictions(self, points1: List[str], points2: List[str]) -> int:
        """Count contradictory statements between two point lists."""
        # Very basic contradiction detection
        contradictions = 0

        contradiction_pairs = [
            ("yes", "no"),
            ("true", "false"),
            ("correct", "incorrect"),
            ("good", "bad"),
            ("positive", "negative"),
            ("beneficial", "harmful")
        ]

        for p1 in points1:
            for p2 in points2:
                p1_lower = p1.lower()
                p2_lower = p2.lower()
                for pos, neg in contradiction_pairs:
                    if (pos in p1_lower and neg in p2_lower) or (neg in p1_lower and pos in p2_lower):
                        contradictions += 1

        return contradictions

    def _have_complementary_key_points(self, key_points_lists: List[List[str]]) -> bool:
        """Check if key points provide complementary information."""
        # Check if different models provide different but related information
        all_unique_points = set()
        for points in key_points_lists:
            all_unique_points.update(points)

        total_individual_points = sum(len(points) for points in key_points_lists)

        # If we have significantly more unique points than average per model,
        # it suggests complementary information
        if len(key_points_lists) > 1:
            avg_points_per_model = total_individual_points / len(key_points_lists)
            if len(all_unique_points) > avg_points_per_model * 1.5:
                return True

        return False

    def _calculate_conflict_severity(
        self,
        answers: List[Optional[str]],
        key_points: List[List[str]],
        confidences: List[float],
        validation_results: Dict[str, Any]
    ) -> float:
        """Calculate the severity of conflicts (0-1 scale)."""
        severity_factors = []

        # Factor 1: Answer agreement (from validation results)
        agreement_score = validation_results.get("agreement_score", 0.5)
        severity_factors.append(1.0 - agreement_score)  # Higher disagreement = higher severity

        # Factor 2: Confidence variance
        if len(confidences) > 1:
            avg_confidence = sum(confidences) / len(confidences)
            variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
            confidence_severity = min(variance * 2, 1.0)  # Scale variance to 0-1
            severity_factors.append(confidence_severity)

        # Factor 3: Missing information
        missing_answers = sum(1 for a in answers if a is None)
        missing_severity = missing_answers / len(answers)
        severity_factors.append(missing_severity)

        # Factor 4: Evidence conflicts
        if self._have_conflicting_key_points(key_points):
            severity_factors.append(0.8)  # High severity for evidence conflicts
        else:
            severity_factors.append(0.2)  # Lower severity if no evidence conflicts

        # Calculate weighted average
        if severity_factors:
            return sum(severity_factors) / len(severity_factors)
        return 0.5  # Default moderate severity

    def _analyze_confidence_distribution(self, confidences: List[float]) -> Dict[str, Any]:
        """Analyze the distribution of confidence scores."""
        if not confidences:
            return {"average": 0.0, "variance": 0.0, "range": "0.0-0.0"}

        average = sum(confidences) / len(confidences)
        variance = sum((c - average) ** 2 for c in confidences) / len(confidences)

        return {
            "average": average,
            "variance": variance,
            "range": ".2f",
            "high_confidence_ratio": sum(1 for c in confidences if c > 0.8) / len(confidences)
        }

    def _calculate_answer_diversity(self, answers: List[Optional[str]]) -> float:
        """Calculate diversity of answers (0-1 scale, higher = more diverse)."""
        valid_answers = [a for a in answers if a is not None]

        if len(valid_answers) <= 1:
            return 0.0

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(valid_answers)):
            for j in range(i + 1, len(valid_answers)):
                sim = self._calculate_answer_similarity(valid_answers[i], valid_answers[j])
                similarities.append(sim)

        if not similarities:
            return 0.0

        avg_similarity = sum(similarities) / len(similarities)
        return 1.0 - avg_similarity  # Higher diversity = lower average similarity

    def _calculate_answer_similarity(self, answer1: str, answer2: str) -> float:
        """Calculate similarity between two answers."""
        try:
            return self.embedding_service.calculate_similarity(answer1, answer2)
        except Exception:
            # Fallback to simple text similarity
            return self._simple_text_similarity(answer1, answer2)

    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _find_key_agreements(self, key_points_lists: List[List[str]]) -> List[str]:
        """Find key points that multiple models agree on."""
        # Count occurrences of similar points
        point_counts = defaultdict(int)

        # Normalize and count points
        for points in key_points_lists:
            seen_in_model = set()  # Avoid double-counting within a model
            for point in points:
                normalized = point.lower().strip()
                if normalized not in seen_in_model and len(normalized) > 10:
                    point_counts[normalized] += 1
                    seen_in_model.add(normalized)

        # Find points mentioned by multiple models
        agreements = [
            point for point, count in point_counts.items()
            if count >= 2  # Mentioned by at least 2 models
        ]

        return agreements[:5]  # Return top 5 agreements

    def _resolve_weighted_average(
        self,
        responses: List[ModelResponse],
        conflict_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve conflicts using weighted average approach."""
        # Weight responses by confidence
        weighted_responses = []
        total_weight = 0

        for resp in responses:
            weight = resp.confidence
            weighted_responses.append((resp, weight))
            total_weight += weight

        # Create weighted answer
        if total_weight > 0:
            # Combine direct answers
            direct_answers = []
            for resp, weight in weighted_responses:
                answer = self._extract_direct_answer(resp.content)
                if answer:
                    direct_answers.append((answer, weight))

            if direct_answers:
                # Use the highest weighted answer
                direct_answers.sort(key=lambda x: x[1], reverse=True)
                primary_answer = direct_answers[0][0]
            else:
                primary_answer = "Unable to determine a clear answer from the responses."

            # Calculate combined confidence
            avg_confidence = sum(resp.confidence for resp in responses) / len(responses)

            # Adjust confidence based on conflict severity
            severity_penalty = conflict_analysis["severity_score"] * 0.3
            final_confidence = max(0.1, avg_confidence - severity_penalty)

            # Combine key points from all responses
            all_key_points = []
            for resp in responses:
                points = self._extract_key_points(resp.content)
                all_key_points.extend(points)

            # Remove duplicates and select top points
            unique_points = list(set(all_key_points))
            top_points = unique_points[:8]  # Limit to 8 points

            return {
                "content": primary_answer,
                "confidence": final_confidence,
                "method": "weighted_average",
                "supporting_points": top_points,
                "num_models_contributing": len(responses),
                "confidence_adjustment": -severity_penalty
            }

        # Fallback if no valid weights
        return self._resolve_majority_vote(responses, conflict_analysis)

    def _resolve_majority_vote(
        self,
        responses: List[ModelResponse],
        conflict_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve conflicts using majority vote approach."""
        # Count votes for each answer
        answer_votes = defaultdict(list)  # answer -> list of (response, confidence)

        for resp in responses:
            answer = self._extract_direct_answer(resp.content)
            if answer:
                answer_votes[answer].append((resp, resp.confidence))

        if not answer_votes:
            return {
                "content": "No clear answers provided by any model.",
                "confidence": 0.1,
                "method": "majority_vote",
                "supporting_points": [],
                "num_models_contributing": 0
            }

        # Find majority answer
        majority_answer = max(answer_votes.items(), key=lambda x: len(x[1]))

        # Calculate confidence based on vote strength
        total_votes = sum(len(votes) for votes in answer_votes.values())
        majority_ratio = len(majority_answer[1]) / total_votes

        # Average confidence of majority voters
        majority_confidences = [resp.confidence for resp, _ in majority_answer[1]]
        avg_confidence = sum(majority_confidences) / len(majority_confidences)

        # Adjust confidence based on consensus strength
        consensus_bonus = (majority_ratio - 0.5) * 0.4  # Bonus for strong majorities
        final_confidence = min(0.95, avg_confidence + consensus_bonus)

        # Collect supporting points from majority
        supporting_points = []
        for resp, _ in majority_answer[1]:
            points = self._extract_key_points(resp.content)
            supporting_points.extend(points)

        unique_points = list(set(supporting_points))[:6]  # Limit points

        return {
            "content": majority_answer[0],
            "confidence": final_confidence,
            "method": "majority_vote",
            "supporting_points": unique_points,
            "num_models_contributing": len(majority_answer[1]),
            "vote_ratio": majority_ratio
        }

    def _resolve_expert_consensus(
        self,
        responses: List[ModelResponse],
        conflict_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve conflicts using expert consensus approach."""
        # Prioritize responses with higher confidence and more detailed answers
        scored_responses = []

        for resp in responses:
            # Score based on multiple factors
            confidence_score = resp.confidence
            detail_score = min(1.0, len(resp.content.split()) / 200)  # Length indicator
            structure_score = self._calculate_structure_score(resp.content)

            total_score = (confidence_score * 0.5 + detail_score * 0.3 + structure_score * 0.2)

            scored_responses.append((resp, total_score))

        # Sort by score
        scored_responses.sort(key=lambda x: x[1], reverse=True)

        # Use top response as primary, but incorporate insights from others
        primary_resp = scored_responses[0][0]

        primary_answer = self._extract_direct_answer(primary_resp.content)
        primary_points = self._extract_key_points(primary_resp.content)

        # Add complementary points from other high-scoring responses
        additional_points = []
        for resp, score in scored_responses[1:4]:  # Top 3 additional responses
            if score > 0.6:  # Only include reasonably good responses
                points = self._extract_key_points(resp.content)
                # Add points not already covered
                for point in points:
                    if not self._point_already_covered(point, primary_points + additional_points):
                        additional_points.append(point)

        final_points = primary_points + additional_points[:4]  # Limit additional points

        # Adjust confidence based on agreement with other models
        agreement_bonus = self._calculate_agreement_bonus(primary_resp, scored_responses[1:])
        final_confidence = min(0.9, primary_resp.confidence + agreement_bonus)

        return {
            "content": primary_answer or "Complex question requiring expert analysis.",
            "confidence": final_confidence,
            "method": "expert_consensus",
            "supporting_points": final_points,
            "num_models_contributing": len(scored_responses),
            "primary_model": primary_resp.model_name
        }

    def _calculate_structure_score(self, content: str) -> float:
        """Calculate how well-structured a response is."""
        score = 0.0

        # Check for numbered lists
        if re.search(r'\d+\.', content):
            score += 0.3

        # Check for bullet points
        if '•' in content or '*' in content[:100]:
            score += 0.2

        # Check for section headers
        if re.search(r'[A-Z][^.!?]*:', content):
            score += 0.2

        # Check for conclusion indicators
        if any(word in content.lower() for word in ['therefore', 'thus', 'conclusion', 'summary']):
            score += 0.3

        return min(1.0, score)

    def _point_already_covered(self, point: str, existing_points: List[str]) -> bool:
        """Check if a point is already covered by existing points."""
        point_lower = point.lower()
        for existing in existing_points:
            if self._simple_text_similarity(point_lower, existing.lower()) > 0.8:
                return True
        return False

    def _calculate_agreement_bonus(self, primary_resp: ModelResponse, other_responses: List[Tuple[ModelResponse, float]]) -> float:
        """Calculate confidence bonus based on agreement with other responses."""
        if not other_responses:
            return 0.0

        agreements = 0
        for resp, _ in other_responses:
            if self._responses_agree(primary_resp, resp):
                agreements += 1

        agreement_ratio = agreements / len(other_responses)
        return agreement_ratio * 0.2  # Max bonus of 0.2

    def _responses_agree(self, resp1: ModelResponse, resp2: ModelResponse) -> bool:
        """Check if two responses agree."""
        answer1 = self._extract_direct_answer(resp1.content)
        answer2 = self._extract_direct_answer(resp2.content)

        if answer1 and answer2:
            similarity = self._calculate_answer_similarity(answer1, answer2)
            return similarity > 0.7

        return False

    def _extract_direct_answer(self, content: str) -> Optional[str]:
        """Extract direct answer from content."""
        # Similar to cross_validate implementation
        patterns = [
            r"Answer:\s*(.+?)(?:\n|$)",
            r"Final Answer:\s*(.+?)(?:\n|$)",
            r"Conclusion:\s*(.+?)(?:\n|$)"
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()

        return None

    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content."""
        # Similar to cross_validate implementation
        points = []
        sentences = re.split(r'[.!?]+', content)

        for sentence in sentences:
            sentence = sentence.strip()
            if 15 < len(sentence) < 200 and not sentence.lower().startswith(('i think', 'perhaps')):
                points.append(sentence)

        return points[:5]  # Limit points

    def _format_single_response(self, response: ModelResponse) -> Dict[str, Any]:
        """Format a single response as resolved answer."""
        return {
            "content": response.content,
            "confidence": response.confidence,
            "method": "single_response",
            "supporting_points": self._extract_key_points(response.content),
            "num_models_contributing": 1,
            "model_used": response.model_name
        }
