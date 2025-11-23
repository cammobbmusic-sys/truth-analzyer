"""
Embeddings - Semantic similarity and verification support.
Provides text embedding services for similarity calculations.
"""

import os
import hashlib
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np


class EmbeddingService:
    """Service for calculating text embeddings and similarities."""

    def __init__(self, cache_dir: str = ".embeddings_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Simple in-memory cache for frequently used embeddings
        self.memory_cache = {}
        self.max_memory_cache_size = 1000

        # Try to initialize sentence transformers if available
        self.use_transformers = self._try_import_transformers()

        if not self.use_transformers:
            print("Warning: Sentence transformers not available. Using fallback similarity calculation.")

    def _try_import_transformers(self) -> bool:
        """Try to import sentence transformers library."""
        try:
            import sentence_transformers
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            return True
        except ImportError:
            try:
                # Try alternative import path
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                return True
            except ImportError:
                return False

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        if self.use_transformers:
            return self._calculate_transformer_similarity(text1, text2)
        else:
            return self._calculate_fallback_similarity(text1, text2)

    def _calculate_transformer_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using sentence transformers."""
        try:
            # Get embeddings
            emb1 = self._get_embedding(text1)
            emb2 = self._get_embedding(text2)

            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

            # Ensure result is between 0 and 1
            return max(0.0, min(1.0, (similarity + 1) / 2))

        except Exception as e:
            print(f"Error calculating transformer similarity: {e}")
            return self._calculate_fallback_similarity(text1, text2)

    def _calculate_fallback_similarity(self, text1: str, text2: str) -> float:
        """Fallback similarity calculation using basic text analysis."""
        # Normalize texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()

        if text1 == text2:
            return 1.0

        # Jaccard similarity of words
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        jaccard_sim = len(intersection) / len(union)

        # Boost similarity for partial matches
        if len(intersection) > 0:
            # Consider word order and context
            jaccard_sim = min(1.0, jaccard_sim * 1.2)

        return jaccard_sim

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text, with caching."""
        # Create cache key
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # Check memory cache first
        if text_hash in self.memory_cache:
            return self.memory_cache[text_hash]

        # Check file cache
        cache_file = self.cache_dir / f"{text_hash}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                self._add_to_memory_cache(text_hash, embedding)
                return embedding
            except Exception:
                pass  # Cache miss, compute embedding

        # Compute embedding
        embedding = self.model.encode(text, convert_to_numpy=True)

        # Cache the result
        self._add_to_memory_cache(text_hash, embedding)

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            print(f"Warning: Could not cache embedding: {e}")

        return embedding

    def _add_to_memory_cache(self, key: str, embedding: np.ndarray):
        """Add embedding to memory cache."""
        self.memory_cache[key] = embedding

        # Clean up cache if too large
        if len(self.memory_cache) > self.max_memory_cache_size:
            # Remove oldest entries (simple FIFO)
            items_to_remove = len(self.memory_cache) - self.max_memory_cache_size + 100
            keys_to_remove = list(self.memory_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.memory_cache[key]

    def find_similar_texts(
        self,
        query: str,
        texts: List[str],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Find texts most similar to a query.

        Args:
            query: Query text
            texts: List of texts to search
            top_k: Number of top results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (text, similarity_score) tuples
        """
        similarities = []

        for text in texts:
            similarity = self.calculate_similarity(query, text)
            if similarity >= threshold:
                similarities.append((text, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def cluster_texts(
        self,
        texts: List[str],
        num_clusters: Optional[int] = None,
        threshold: float = 0.7
    ) -> List[List[str]]:
        """
        Cluster texts based on semantic similarity.

        Args:
            texts: List of texts to cluster
            num_clusters: Target number of clusters (optional)
            threshold: Similarity threshold for clustering

        Returns:
            List of text clusters
        """
        if not texts:
            return []

        if len(texts) == 1:
            return [texts]

        # Calculate similarity matrix
        n = len(texts)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                sim = self.calculate_similarity(texts[i], texts[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        # Perform clustering
        clusters = []
        used = set()

        for i in range(n):
            if i in used:
                continue

            # Start new cluster
            cluster = [texts[i]]
            used.add(i)

            # Find similar texts
            for j in range(n):
                if j not in used and similarity_matrix[i, j] >= threshold:
                    cluster.append(texts[j])
                    used.add(j)

            clusters.append(cluster)

        # If target number of clusters specified, merge smallest clusters
        if num_clusters and len(clusters) > num_clusters:
            clusters = self._merge_clusters(clusters, num_clusters, similarity_matrix, texts)

        return clusters

    def _merge_clusters(
        self,
        clusters: List[List[str]],
        target_num: int,
        similarity_matrix: np.ndarray,
        texts: List[str]
    ) -> List[List[str]]:
        """Merge clusters until target number is reached."""
        text_to_index = {text: i for i, text in enumerate(texts)}

        while len(clusters) > target_num:
            # Find the two closest clusters
            min_distance = float('inf')
            merge_i, merge_j = -1, -1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    distance = self._cluster_distance(clusters[i], clusters[j], similarity_matrix, text_to_index)
                    if distance < min_distance:
                        min_distance = distance
                        merge_i, merge_j = i, j

            if merge_i == -1:
                break  # Cannot merge further

            # Merge clusters
            clusters[merge_i].extend(clusters[merge_j])
            del clusters[merge_j]

        return clusters

    def _cluster_distance(
        self,
        cluster1: List[str],
        cluster2: List[str],
        similarity_matrix: np.ndarray,
        text_to_index: Dict[str, int]
    ) -> float:
        """Calculate distance between two clusters."""
        similarities = []

        for text1 in cluster1:
            for text2 in cluster2:
                i = text_to_index[text1]
                j = text_to_index[text2]
                similarities.append(similarity_matrix[i, j])

        if similarities:
            return 1.0 - (sum(similarities) / len(similarities))  # Convert similarity to distance
        return 1.0  # Maximum distance

    def calculate_text_complexity(self, text: str) -> Dict[str, float]:
        """
        Calculate various complexity metrics for text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary of complexity metrics
        """
        words = text.split()
        sentences = text.split('.')

        metrics = {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "vocabulary_richness": len(set(words)) / len(words) if words else 0
        }

        return metrics

    def extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Extract important keywords from text using TF-IDF like approach.

        Args:
            text: Text to extract keywords from
            top_k: Number of top keywords to return

        Returns:
            List of (keyword, score) tuples
        """
        words = text.lower().split()
        word_counts = {}

        # Count word frequencies
        for word in words:
            word = word.strip('.,!?;:')
            if len(word) > 2:  # Skip very short words
                word_counts[word] = word_counts.get(word, 0) + 1

        # Calculate scores (simple TF approach)
        total_words = len(words)
        keywords = []

        for word, count in word_counts.items():
            # Simple scoring: TF * (1 / log(word_length + 1)) to prefer meaningful words
            tf = count / total_words
            length_penalty = 1.0 / (len(word) ** 0.5)  # Penalize very long/short words
            score = tf * length_penalty
            keywords.append((word, score))

        # Sort by score
        keywords.sort(key=lambda x: x[1], reverse=True)

        return keywords[:top_k]

    def clear_cache(self):
        """Clear the embedding cache."""
        import shutil
        try:
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            self.memory_cache.clear()
            print("Embedding cache cleared.")
        except Exception as e:
            print(f"Error clearing cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.pkl"))

        return {
            "cache_directory": str(self.cache_dir),
            "cached_embeddings": len(cache_files),
            "memory_cache_size": len(self.memory_cache),
            "max_memory_cache_size": self.max_memory_cache_size,
            "transformers_available": self.use_transformers
        }


def cosine_similarity(text1: str, text2: str) -> float:
    """
    Convenience function for calculating cosine similarity between two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0 and 1
    """
    service = EmbeddingService()
    return service.calculate_similarity(text1, text2)


def pairwise_similarity(texts: List[str]) -> float:
    """
    Calculate average pairwise similarity across a list of texts.

    Args:
        texts: List of text strings

    Returns:
        Average similarity score (0-1), or 0 if insufficient texts
    """
    if len(texts) < 2:
        return 0.0

    total_similarity = 0.0
    pair_count = 0

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            try:
                similarity = cosine_similarity(texts[i], texts[j])
                total_similarity += similarity
                pair_count += 1
            except Exception:
                continue

    return total_similarity / max(pair_count, 1)
