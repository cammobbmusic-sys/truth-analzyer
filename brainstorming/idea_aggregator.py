"""
Idea Aggregator - Iterative idea generation and clustering.
Manages brainstorming sessions and organizes generated ideas.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

from config import Config
from utils.embeddings import EmbeddingService


@dataclass
class Idea:
    """Represents a single generated idea."""
    title: str
    description: str
    benefits: List[str]
    challenges: List[str]
    implementation_steps: List[str]
    innovation_level: str  # "low", "medium", "high"
    source_model: str
    confidence_score: float = 0.0
    cluster_id: Optional[int] = None


@dataclass
class IdeaCluster:
    """Represents a cluster of related ideas."""
    cluster_id: int
    central_theme: str
    ideas: List[Idea]
    common_benefits: List[str]
    common_challenges: List[str]
    diversity_score: float
    average_innovation: float


class IdeaAggregator:
    """Aggregates and organizes ideas from multiple AI models during brainstorming."""

    def __init__(self, config: Config):
        self.config = config
        self.embedding_service = EmbeddingService()

    def extract_ideas(self, responses: List[Any]) -> List[Idea]:
        """
        Extract ideas from multiple model responses.

        Args:
            responses: List of model responses containing ideas

        Returns:
            List of extracted and structured ideas
        """
        all_ideas = []

        for response in responses:
            try:
                ideas = self._parse_ideas_from_response(response)
                all_ideas.extend(ideas)
            except Exception as e:
                print(f"Warning: Failed to parse ideas from response: {e}")
                continue

        # Remove duplicates and near-duplicates
        unique_ideas = self._deduplicate_ideas(all_ideas)

        # Calculate confidence scores
        for idea in unique_ideas:
            idea.confidence_score = self._calculate_idea_confidence(idea)

        return unique_ideas

    def _parse_ideas_from_response(self, response: Any) -> List[Idea]:
        """Parse ideas from a single model response."""
        content = response.content
        model_name = response.model_name

        ideas = []

        # Split content into idea sections
        idea_sections = self._split_into_idea_sections(content)

        for section in idea_sections:
            try:
                idea = self._parse_single_idea(section, model_name)
                if idea:
                    ideas.append(idea)
            except Exception as e:
                print(f"Warning: Failed to parse idea section: {e}")
                continue

        return ideas

    def _split_into_idea_sections(self, content: str) -> List[str]:
        """Split response content into individual idea sections."""
        # Look for numbered lists, bullet points, or idea headers
        sections = []

        # Try numbered lists first (1., 2., etc.)
        numbered_pattern = r'(\d+\.[\s\S]*?)(?=\d+\.|$)'
        matches = re.findall(numbered_pattern, content, re.MULTILINE)

        if matches:
            sections = [match.strip() for match in matches if len(match.strip()) > 20]
        else:
            # Try bullet points
            bullet_pattern = r'(•[\s\S]*?)(?=•|$)'
            matches = re.findall(bullet_pattern, content)

            if matches:
                sections = [match.strip() for match in matches if len(match.strip()) > 20]
            else:
                # Try to split by idea titles or headers
                lines = content.split('\n')
                current_section = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Check if this looks like an idea title
                    if self._is_idea_title(line):
                        if current_section:
                            sections.append('\n'.join(current_section))
                        current_section = [line]
                    else:
                        current_section.append(line)

                if current_section:
                    sections.append('\n'.join(current_section))

        # If no clear sections found, treat whole content as one idea
        if not sections and len(content.strip()) > 50:
            sections = [content.strip()]

        return sections

    def _is_idea_title(self, line: str) -> bool:
        """Check if a line looks like an idea title."""
        # Criteria for idea titles:
        # - Short (less than 100 chars)
        # - Starts with capital letter
        # - Contains action words
        # - Not ending with punctuation that suggests continuation

        if len(line) > 100 or len(line) < 5:
            return False

        if not line[0].isupper():
            return False

        # Check for action words
        action_words = ['use', 'create', 'develop', 'implement', 'build', 'design', 'establish']
        has_action = any(word in line.lower() for word in action_words)

        # Avoid lines that end with colons (likely section headers)
        if line.endswith(':'):
            return False

        return has_action

    def _parse_single_idea(self, section: str, model_name: str) -> Optional[Idea]:
        """Parse a single idea from a section of text."""
        lines = section.split('\n')
        lines = [line.strip() for line in lines if line.strip()]

        if not lines:
            return None

        # Extract title (first line or first meaningful line)
        title = self._extract_title(lines)

        # Extract description
        description = self._extract_description(lines)

        # Extract benefits
        benefits = self._extract_benefits(lines)

        # Extract challenges
        challenges = self._extract_challenges(lines)

        # Extract implementation steps
        implementation_steps = self._extract_implementation(lines)

        # Extract innovation level
        innovation_level = self._extract_innovation_level(lines)

        # Validate that we have minimum required information
        if not title or not description:
            return None

        return Idea(
            title=title,
            description=description,
            benefits=benefits,
            challenges=challenges,
            implementation_steps=implementation_steps,
            innovation_level=innovation_level,
            source_model=model_name
        )

    def _extract_title(self, lines: List[str]) -> str:
        """Extract idea title from lines."""
        # First line is usually the title
        if lines:
            title = lines[0].strip()
            # Clean up common prefixes
            title = re.sub(r'^\d+\.\s*', '', title)  # Remove numbering
            title = re.sub(r'^•\s*', '', title)  # Remove bullets
            return title

        return "Untitled Idea"

    def _extract_description(self, lines: List[str]) -> str:
        """Extract idea description."""
        # Look for description patterns
        description_lines = []

        for line in lines[1:]:  # Skip title
            # Stop at structured sections
            if any(indicator in line.lower() for indicator in [
                'benefits:', 'advantages:', 'challenges:', 'problems:',
                'implementation:', 'steps:', 'innovation:'
            ]):
                break

            # Skip very short lines that might be headers
            if len(line) < 10:
                continue

            description_lines.append(line)

        if description_lines:
            return ' '.join(description_lines)

        # Fallback: use second line or combine first few lines
        if len(lines) > 1:
            return lines[1] if len(lines) == 2 else ' '.join(lines[1:3])

        return "No description available."

    def _extract_benefits(self, lines: List[str]) -> List[str]:
        """Extract benefits from idea description."""
        return self._extract_list_section(lines, ['benefits', 'advantages', 'pros', 'positive'])

    def _extract_challenges(self, lines: List[str]) -> List[str]:
        """Extract challenges from idea description."""
        return self._extract_list_section(lines, ['challenges', 'problems', 'cons', 'difficulties', 'risks'])

    def _extract_implementation(self, lines: List[str]) -> List[str]:
        """Extract implementation steps."""
        return self._extract_list_section(lines, ['implementation', 'steps', 'how to', 'process'])

    def _extract_list_section(self, lines: List[str], keywords: List[str]) -> List[str]:
        """Extract a list section based on keywords."""
        items = []
        in_section = False

        for line in lines:
            line_lower = line.lower()

            # Check if we're entering the section
            if any(keyword in line_lower for keyword in keywords):
                in_section = True
                continue

            # Check if we're leaving the section (next major section)
            if in_section and any(indicator in line_lower for indicator in [
                'benefits:', 'challenges:', 'implementation:', 'innovation:'
            ] if indicator not in [f"{k}:" for k in keywords]):
                break

            if in_section:
                # Extract list items
                if line.strip().startswith(('- ', '• ', '* ', '1. ', '2. ', '3. ')):
                    # Remove list markers
                    clean_line = re.sub(r'^[-•*]\s*', '', line.strip())
                    clean_line = re.sub(r'^\d+\.\s*', '', clean_line)
                    if len(clean_line) > 5:
                        items.append(clean_line)

        return items

    def _extract_innovation_level(self, lines: List[str]) -> str:
        """Extract innovation level from the idea."""
        content = ' '.join(lines).lower()

        # Look for explicit mentions
        if any(word in content for word in ['revolutionary', 'breakthrough', 'innovative']):
            return 'high'
        elif any(word in content for word in ['novel', 'creative', 'new approach']):
            return 'medium'
        elif any(word in content for word in ['improvement', 'enhancement', 'modification']):
            return 'medium'
        else:
            return 'low'

    def _deduplicate_ideas(self, ideas: List[Idea]) -> List[Idea]:
        """Remove duplicate and near-duplicate ideas."""
        if not ideas:
            return []

        unique_ideas = []
        seen_titles = set()
        seen_descriptions = set()

        for idea in ideas:
            # Check title uniqueness
            title_key = idea.title.lower().strip()
            if title_key in seen_titles:
                continue

            # Check description similarity
            desc_key = idea.description.lower().strip()[:100]  # First 100 chars
            is_duplicate = False

            for seen_desc in seen_descriptions:
                try:
                    similarity = self.embedding_service.calculate_similarity(desc_key, seen_desc)
                    if similarity > 0.85:  # Very similar
                        is_duplicate = True
                        break
                except Exception:
                    # Fallback to simple string similarity
                    if self._simple_similarity(desc_key, seen_desc) > 0.8:
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique_ideas.append(idea)
                seen_titles.add(title_key)
                seen_descriptions.add(desc_key)

        return unique_ideas

    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _calculate_idea_confidence(self, idea: Idea) -> float:
        """Calculate confidence score for an idea."""
        score = 0.5  # Base score

        # Factor 1: Completeness
        completeness_factors = [
            bool(idea.title),
            bool(idea.description and len(idea.description) > 20),
            bool(idea.benefits),
            bool(idea.challenges),
            bool(idea.implementation_steps)
        ]
        completeness_score = sum(completeness_factors) / len(completeness_factors)
        score += completeness_score * 0.2

        # Factor 2: Innovation level
        innovation_scores = {'low': 0.1, 'medium': 0.15, 'high': 0.2}
        score += innovation_scores.get(idea.innovation_level, 0.1)

        # Factor 3: Detail level
        total_details = len(idea.benefits) + len(idea.challenges) + len(idea.implementation_steps)
        detail_score = min(total_details / 10, 1.0) * 0.2  # Max 0.2 for very detailed ideas
        score += detail_score

        # Factor 4: Description quality
        if idea.description:
            desc_length_score = min(len(idea.description) / 200, 1.0) * 0.1
            score += desc_length_score

        return min(1.0, score)

    def aggregate_and_cluster_ideas(self, ideas: List[Idea]) -> Dict[str, Any]:
        """
        Aggregate ideas and cluster them by similarity.

        Args:
            ideas: List of ideas to cluster

        Returns:
            Clustering results
        """
        if not ideas:
            return {
                "clusters": [],
                "top_ideas": [],
                "diversity_score": 0.0,
                "total_ideas": 0
            }

        # Perform clustering
        clusters = self._cluster_ideas(ideas)

        # Select top ideas
        top_ideas = self._select_top_ideas(ideas)

        # Calculate diversity score
        diversity_score = self._calculate_diversity_score(clusters)

        return {
            "clusters": clusters,
            "top_ideas": top_ideas,
            "diversity_score": diversity_score,
            "total_ideas": len(ideas),
            "num_clusters": len(clusters)
        }

    def _cluster_ideas(self, ideas: List[Idea]) -> List[IdeaCluster]:
        """Cluster ideas based on semantic similarity."""
        if len(ideas) <= 1:
            return [IdeaCluster(
                cluster_id=0,
                central_theme=ideas[0].title if ideas else "Single Idea",
                ideas=ideas,
                common_benefits=[],
                common_challenges=[],
                diversity_score=0.0,
                average_innovation=0.0
            )]

        # Calculate similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(ideas)

        # Perform clustering (simple agglomerative approach)
        clusters = self._perform_clustering(ideas, similarity_matrix)

        # Convert to IdeaCluster objects
        idea_clusters = []
        for i, cluster_ideas in enumerate(clusters):
            if cluster_ideas:  # Skip empty clusters
                cluster = self._create_cluster_object(i, cluster_ideas)
                idea_clusters.append(cluster)

        return idea_clusters

    def _calculate_similarity_matrix(self, ideas: List[Idea]) -> List[List[float]]:
        """Calculate pairwise similarity matrix for ideas."""
        n = len(ideas)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                similarity = self._calculate_idea_similarity(ideas[i], ideas[j])
                matrix[i][j] = similarity
                matrix[j][i] = similarity

        return matrix

    def _calculate_idea_similarity(self, idea1: Idea, idea2: Idea) -> float:
        """Calculate similarity between two ideas."""
        # Combine title and description for comparison
        text1 = f"{idea1.title} {idea1.description}"
        text2 = f"{idea2.title} {idea2.description}"

        try:
            return self.embedding_service.calculate_similarity(text1, text2)
        except Exception:
            # Fallback to simple similarity
            return self._simple_similarity(text1, text2)

    def _perform_clustering(self, ideas: List[Idea], similarity_matrix: List[List[float]]) -> List[List[Idea]]:
        """Perform simple agglomerative clustering."""
        # Start with each idea in its own cluster
        clusters = [[idea] for idea in ideas]

        # Continue merging until we have a reasonable number of clusters
        target_clusters = max(3, len(ideas) // 4)  # Aim for 3-4 ideas per cluster

        while len(clusters) > target_clusters and len(clusters) > 1:
            # Find most similar pair of clusters
            max_similarity = -1
            merge_i, merge_j = -1, -1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    similarity = self._calculate_cluster_similarity(
                        clusters[i], clusters[j], similarity_matrix, ideas.index
                    )
                    if similarity > max_similarity:
                        max_similarity = similarity
                        merge_i, merge_j = i, j

            if max_similarity < 0.3:  # Stop if clusters are too dissimilar
                break

            # Merge the clusters
            clusters[merge_i].extend(clusters[merge_j])
            del clusters[merge_j]

        return clusters

    def _calculate_cluster_similarity(
        self,
        cluster1: List[Idea],
        cluster2: List[Idea],
        similarity_matrix: List[List[float]],
        idea_index
    ) -> float:
        """Calculate similarity between two clusters."""
        similarities = []

        for idea1 in cluster1:
            for idea2 in cluster2:
                i = idea_index(idea1)
                j = idea_index(idea2)
                similarities.append(similarity_matrix[i][j])

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _create_cluster_object(self, cluster_id: int, ideas: List[Idea]) -> IdeaCluster:
        """Create an IdeaCluster object from a list of ideas."""
        # Determine central theme
        central_theme = self._determine_central_theme(ideas)

        # Find common benefits and challenges
        common_benefits = self._find_common_elements([idea.benefits for idea in ideas])
        common_challenges = self._find_common_elements([idea.challenges for idea in ideas])

        # Calculate diversity within cluster
        diversity_score = self._calculate_cluster_diversity(ideas)

        # Calculate average innovation
        innovation_mapping = {'low': 1, 'medium': 2, 'high': 3}
        innovation_scores = [innovation_mapping.get(idea.innovation_level, 1) for idea in ideas]
        average_innovation = sum(innovation_scores) / len(innovation_scores) if innovation_scores else 1

        return IdeaCluster(
            cluster_id=cluster_id,
            central_theme=central_theme,
            ideas=ideas,
            common_benefits=common_benefits,
            common_challenges=common_challenges,
            diversity_score=diversity_score,
            average_innovation=average_innovation
        )

    def _determine_central_theme(self, ideas: List[Idea]) -> str:
        """Determine the central theme of a cluster of ideas."""
        if not ideas:
            return "Empty Cluster"

        # Use the title of the first idea as a starting point
        titles = [idea.title for idea in ideas]

        # If all titles are very similar, use that
        if len(set(title.lower() for title in titles)) == 1:
            return titles[0]

        # Otherwise, try to find common keywords
        all_words = []
        for title in titles:
            words = title.lower().split()
            all_words.extend(words)

        word_counts = defaultdict(int)
        for word in all_words:
            if len(word) > 3:  # Skip short words
                word_counts[word] += 1

        # Find most common words
        common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        top_words = [word for word, count in common_words[:3] if count > 1]

        if top_words:
            return " ".join(top_words).title()

        # Fallback to first idea's title
        return ideas[0].title

    def _find_common_elements(self, element_lists: List[List[str]]) -> List[str]:
        """Find elements that appear in multiple lists."""
        element_counts = defaultdict(int)

        for element_list in element_lists:
            seen_in_list = set()  # Avoid double-counting within a list
            for element in element_list:
                normalized = element.lower().strip()
                if normalized not in seen_in_list:
                    element_counts[normalized] += 1
                    seen_in_list.add(normalized)

        # Return elements that appear in at least 2 lists
        common = [element for element, count in element_counts.items() if count >= 2]
        return common[:5]  # Limit to 5

    def _calculate_cluster_diversity(self, ideas: List[Idea]) -> float:
        """Calculate diversity score within a cluster."""
        if len(ideas) <= 1:
            return 0.0

        # Calculate average pairwise similarity
        similarities = []
        for i in range(len(ideas)):
            for j in range(i + 1, len(ideas)):
                sim = self._calculate_idea_similarity(ideas[i], ideas[j])
                similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        # Diversity is inverse of similarity
        return 1.0 - avg_similarity

    def _select_top_ideas(self, ideas: List[Idea], top_n: int = 10) -> List[Dict[str, Any]]:
        """Select top ideas based on various criteria."""
        # Sort ideas by confidence score
        sorted_ideas = sorted(ideas, key=lambda x: x.confidence_score, reverse=True)

        top_ideas = []
        for idea in sorted_ideas[:top_n]:
            top_ideas.append({
                "title": idea.title,
                "description": idea.description,
                "benefits": idea.benefits,
                "challenges": idea.challenges,
                "innovation_level": idea.innovation_level,
                "confidence_score": idea.confidence_score,
                "source_model": idea.source_model
            })

        return top_ideas

    def _calculate_diversity_score(self, clusters: List[IdeaCluster]) -> float:
        """Calculate overall diversity score across all clusters."""
        if not clusters:
            return 0.0

        # Average diversity across clusters
        cluster_diversities = [cluster.diversity_score for cluster in clusters]
        avg_cluster_diversity = sum(cluster_diversities) / len(cluster_diversities)

        # Factor in number of clusters (more clusters = more diverse)
        cluster_factor = min(len(clusters) / 5, 1.0)  # Max benefit at 5 clusters

        return (avg_cluster_diversity + cluster_factor) / 2

    def generate_refinement_prompt(
        self,
        original_topic: str,
        existing_ideas: List[Idea],
        iteration: int
    ) -> str:
        """
        Generate a refinement prompt for the next brainstorming iteration.

        Args:
            original_topic: The original topic
            existing_ideas: Ideas generated so far
            iteration: Current iteration number

        Returns:
            Refined prompt for next iteration
        """
        # Analyze existing ideas
        themes_covered = set()
        innovation_levels = []

        for idea in existing_ideas:
            # Extract themes from titles
            title_words = idea.title.lower().split()
            themes_covered.update(title_words)
            innovation_levels.append(idea.innovation_level)

        # Determine focus for next iteration
        if iteration == 1:
            focus = "Focus on more unconventional and creative approaches."
        elif iteration == 2:
            if innovation_levels.count('high') < len(existing_ideas) * 0.3:
                focus = "Emphasize breakthrough innovations and novel solutions."
            else:
                focus = "Explore practical implementation details and scalability."
        else:
            focus = "Consider hybrid approaches combining different ideas."

        # Build prompt
        prompt = f"""
Generate {self.config.brainstorming.num_ideas_per_iteration} additional ideas for: {original_topic}

Existing themes covered: {', '.join(list(themes_covered)[:10])}
{focus}

Requirements:
- Build upon or complement existing ideas
- Explore different angles not yet covered
- Maintain the same format as previous responses
- Focus on feasibility and impact

"""

        return prompt.strip()
