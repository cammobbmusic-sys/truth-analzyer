"""
Evaluation Task Library

Comprehensive collection of standardized evaluation tasks across different genres
and difficulty levels for consistent AI model benchmarking.
"""

from evaluation_framework import EvaluationGenre
from typing import Dict, List, Any, Optional


class EvaluationTaskLibrary:
    """
    Library of standardized evaluation tasks for consistent AI model benchmarking.

    Provides tasks across multiple genres with varying difficulty levels,
    expected outputs, and evaluation criteria.
    """

    def __init__(self):
        self.tasks = self._load_standard_tasks()

    def _load_standard_tasks(self) -> Dict[EvaluationGenre, List[Dict[str, Any]]]:
        """Load comprehensive task library."""
        return {
            EvaluationGenre.FACTUAL_QA: [
                # Easy factual questions
                {
                    "id": "factual_001",
                    "difficulty": "easy",
                    "input": "What is the capital of France?",
                    "expected_output": "Paris",
                    "evaluation_criteria": ["exact_match", "semantic_similarity"],
                    "metadata": {"category": "geography", "requires_knowledge": True}
                },
                {
                    "id": "factual_002",
                    "difficulty": "easy",
                    "input": "Who wrote the play Romeo and Juliet?",
                    "expected_output": "William Shakespeare",
                    "evaluation_criteria": ["exact_match", "semantic_similarity"],
                    "metadata": {"category": "literature", "requires_knowledge": True}
                },
                # Medium factual questions
                {
                    "id": "factual_003",
                    "difficulty": "medium",
                    "input": "What is the chemical formula for table salt?",
                    "expected_output": "NaCl",
                    "evaluation_criteria": ["exact_match", "chemical_formula_validation"],
                    "metadata": {"category": "chemistry", "requires_knowledge": True}
                },
                {
                    "id": "factual_004",
                    "difficulty": "medium",
                    "input": "In what year did World War II end?",
                    "expected_output": "1945",
                    "evaluation_criteria": ["exact_match", "temporal_accuracy"],
                    "metadata": {"category": "history", "requires_knowledge": True}
                },
                # Hard factual questions
                {
                    "id": "factual_005",
                    "difficulty": "hard",
                    "input": "What is the name of the deepest known point in Earth's oceans?",
                    "expected_output": "Challenger Deep",
                    "evaluation_criteria": ["exact_match", "semantic_similarity"],
                    "metadata": {"category": "geography", "requires_knowledge": True}
                }
            ],

            EvaluationGenre.MATHEMATICAL: [
                # Basic arithmetic
                {
                    "id": "math_001",
                    "difficulty": "easy",
                    "input": "What is 15 + 27?",
                    "expected_output": "42",
                    "evaluation_criteria": ["exact_match", "numerical_accuracy"],
                    "metadata": {"category": "arithmetic", "requires_calculation": True}
                },
                {
                    "id": "math_002",
                    "difficulty": "easy",
                    "input": "What is 8 × 12?",
                    "expected_output": "96",
                    "evaluation_criteria": ["exact_match", "numerical_accuracy"],
                    "metadata": {"category": "arithmetic", "requires_calculation": True}
                },
                # Intermediate math
                {
                    "id": "math_003",
                    "difficulty": "medium",
                    "input": "What is the square root of 144?",
                    "expected_output": "12",
                    "evaluation_criteria": ["exact_match", "numerical_accuracy"],
                    "metadata": {"category": "algebra", "requires_calculation": True}
                },
                {
                    "id": "math_004",
                    "difficulty": "medium",
                    "input": "Solve for x: 2x + 5 = 17",
                    "expected_output": "6",
                    "evaluation_criteria": ["exact_match", "step_by_step_reasoning"],
                    "metadata": {"category": "algebra", "requires_reasoning": True}
                },
                # Advanced math
                {
                    "id": "math_005",
                    "difficulty": "hard",
                    "input": "What is the derivative of f(x) = x³ + 2x² - 5x + 1?",
                    "expected_output": "3x² + 4x - 5",
                    "evaluation_criteria": ["exact_match", "calculus_rules"],
                    "metadata": {"category": "calculus", "requires_calculation": True}
                }
            ],

            EvaluationGenre.REASONING: [
                # Logical reasoning
                {
                    "id": "reasoning_001",
                    "difficulty": "easy",
                    "input": "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?",
                    "expected_output": "No, we cannot conclude this. The premises don't establish a connection between roses and fading quickly.",
                    "evaluation_criteria": ["logical_correctness", "explanation_clarity"],
                    "metadata": {"category": "logic", "requires_reasoning": True}
                },
                {
                    "id": "reasoning_002",
                    "difficulty": "medium",
                    "input": "If all lawyers are politicians, and some politicians are honest, does it follow that some lawyers are honest?",
                    "expected_output": "Yes, this follows logically. If some politicians are honest and all lawyers are politicians, then at least some lawyers must be honest.",
                    "evaluation_criteria": ["logical_correctness", "syllogistic_reasoning"],
                    "metadata": {"category": "logic", "requires_reasoning": True}
                },
                # Analytical reasoning
                {
                    "id": "reasoning_003",
                    "difficulty": "hard",
                    "input": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                    "expected_output": "$0.05 (not $0.10 as many incorrectly assume)",
                    "evaluation_criteria": ["mathematical_reasoning", "avoids_common_fallacy"],
                    "metadata": {"category": "cognitive_bias", "requires_critical_thinking": True}
                }
            ],

            EvaluationGenre.CREATIVE_WRITING: [
                {
                    "id": "creative_001",
                    "difficulty": "easy",
                    "input": "Write a haiku about artificial intelligence",
                    "expected_output": None,  # Subjective evaluation
                    "evaluation_criteria": ["haiku_structure", "thematic_relevance", "poetic_quality"],
                    "metadata": {"category": "poetry", "requires_creativity": True}
                },
                {
                    "id": "creative_002",
                    "difficulty": "medium",
                    "input": "Write a short story beginning with: 'The old clock struck thirteen...'",
                    "expected_output": None,
                    "evaluation_criteria": ["narrative_coherence", "creative_opening", "story_development"],
                    "metadata": {"category": "fiction", "requires_creativity": True}
                },
                {
                    "id": "creative_003",
                    "difficulty": "hard",
                    "input": "Compose a sonnet about the future of human-AI collaboration",
                    "expected_output": None,
                    "evaluation_criteria": ["sonnet_structure", "metaphor_usage", "thematic_depth"],
                    "metadata": {"category": "poetry", "requires_creativity": True}
                }
            ],

            EvaluationGenre.CODE_GENERATION: [
                {
                    "id": "code_001",
                    "difficulty": "easy",
                    "input": "Write a Python function to calculate the factorial of a number",
                    "expected_output": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n-1)",
                    "evaluation_criteria": ["syntax_correctness", "logic_correctness", "efficiency"],
                    "metadata": {"language": "python", "category": "recursion"}
                },
                {
                    "id": "code_002",
                    "difficulty": "medium",
                    "input": "Write a Python function to check if a string is a palindrome",
                    "expected_output": "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]",
                    "evaluation_criteria": ["correctness", "edge_case_handling", "efficiency"],
                    "metadata": {"language": "python", "category": "string_manipulation"}
                },
                {
                    "id": "code_003",
                    "difficulty": "hard",
                    "input": "Implement a binary search tree with insert and search operations in Python",
                    "expected_output": None,  # Complex implementation
                    "evaluation_criteria": ["data_structure_correctness", "algorithm_efficiency", "code_quality"],
                    "metadata": {"language": "python", "category": "data_structures"}
                }
            ],

            EvaluationGenre.ANALYSIS: [
                {
                    "id": "analysis_001",
                    "difficulty": "easy",
                    "input": "Analyze the main themes in the sentence: 'The quick brown fox jumps over the lazy dog.'",
                    "expected_output": "The sentence demonstrates linguistic completeness and contains all letters of the alphabet.",
                    "evaluation_criteria": ["comprehensiveness", "accuracy", "insight_depth"],
                    "metadata": {"category": "linguistic_analysis", "requires_analytical_thinking": True}
                },
                {
                    "id": "analysis_002",
                    "difficulty": "medium",
                    "input": "Analyze the potential environmental impact of electric vehicles compared to gasoline vehicles.",
                    "expected_output": None,  # Complex analysis
                    "evaluation_criteria": ["comprehensiveness", "balanced_viewpoint", "evidence_based_reasoning"],
                    "metadata": {"category": "environmental_analysis", "requires_knowledge": True}
                },
                {
                    "id": "analysis_003",
                    "difficulty": "hard",
                    "input": "Analyze the economic implications of universal basic income in a post-AI automation society.",
                    "expected_output": None,
                    "evaluation_criteria": ["economic_reasoning", "societal_impact_analysis", "feasibility_assessment"],
                    "metadata": {"category": "economic_analysis", "requires_interdisciplinary_knowledge": True}
                }
            ],

            EvaluationGenre.CONVERSATION: [
                {
                    "id": "conv_001",
                    "difficulty": "easy",
                    "input": "Hello! How are you today?",
                    "expected_output": None,  # Natural conversation
                    "evaluation_criteria": ["naturalness", "context_awareness", "engagement"],
                    "metadata": {"category": "casual_conversation", "requires_social_understanding": True}
                },
                {
                    "id": "conv_002",
                    "difficulty": "medium",
                    "input": "I'm feeling anxious about an upcoming presentation. Any advice?",
                    "expected_output": None,
                    "evaluation_criteria": ["empathy", "practical_advice", "supportiveness"],
                    "metadata": {"category": "emotional_support", "requires_emotional_intelligence": True}
                }
            ],

            EvaluationGenre.SUMMARIZATION: [
                {
                    "id": "summary_001",
                    "difficulty": "easy",
                    "input": "Summarize this text in 2-3 sentences: 'The Industrial Revolution was a period of major industrialization that took place from the late 1700s to early 1800s. It originated in Great Britain and spread to other parts of the world. During this time, many agricultural societies became industrial ones, with factories and mass production becoming common.'",
                    "expected_output": None,
                    "evaluation_criteria": ["conciseness", "key_points_coverage", "coherence"],
                    "metadata": {"category": "historical_summary", "requires_comprehension": True}
                },
                {
                    "id": "summary_002",
                    "difficulty": "medium",
                    "input": "Summarize the main arguments in favor of renewable energy adoption.",
                    "expected_output": None,
                    "evaluation_criteria": ["argument_comprehensiveness", "logical_structure", "balanced_presentation"],
                    "metadata": {"category": "argumentative_summary", "requires_analytical_skills": True}
                }
            ],

            EvaluationGenre.CLASSIFICATION: [
                {
                    "id": "classify_001",
                    "difficulty": "easy",
                    "input": "Classify this animal: 'A large mammal with a long neck and brown spots.'",
                    "expected_output": "Giraffe",
                    "evaluation_criteria": ["correctness", "classification_reasoning"],
                    "metadata": {"category": "biological_classification", "requires_knowledge": True}
                },
                {
                    "id": "classify_002",
                    "difficulty": "medium",
                    "input": "Classify the type of literary device: 'The wind whispered through the trees.'",
                    "expected_output": "Personification",
                    "evaluation_criteria": ["literary_knowledge", "explanation_accuracy"],
                    "metadata": {"category": "literary_analysis", "requires_education": True}
                }
            ]
        }

    def get_tasks_by_genre(self, genre: EvaluationGenre,
                          difficulty: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get tasks for a specific genre and difficulty level."""
        tasks = self.tasks.get(genre, [])

        if difficulty:
            tasks = [task for task in tasks if task.get("difficulty") == difficulty]

        return tasks

    def get_task_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific task by ID."""
        for genre_tasks in self.tasks.values():
            for task in genre_tasks:
                if task["id"] == task_id:
                    return task
        return None

    def get_random_tasks(self, genre: EvaluationGenre, count: int = 5,
                        difficulty: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get random tasks for evaluation."""
        import random
        tasks = self.get_tasks_by_genre(genre, difficulty)

        if len(tasks) <= count:
            return tasks

        return random.sample(tasks, count)

    def add_custom_task(self, genre: EvaluationGenre, task: Dict[str, Any]):
        """Add a custom task to the library."""
        if genre not in self.tasks:
            self.tasks[genre] = []

        # Validate task structure
        required_fields = ["id", "difficulty", "input", "evaluation_criteria"]
        for field in required_fields:
            if field not in task:
                raise ValueError(f"Task missing required field: {field}")

        self.tasks[genre].append(task)

    def get_genre_statistics(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about available tasks."""
        stats = {}

        for genre, tasks in self.tasks.items():
            genre_stats = {
                "total": len(tasks),
                "easy": len([t for t in tasks if t["difficulty"] == "easy"]),
                "medium": len([t for t in tasks if t["difficulty"] == "medium"]),
                "hard": len([t for t in tasks if t["difficulty"] == "hard"])
            }
            stats[genre.value] = genre_stats

        return stats


# Global task library instance
task_library = EvaluationTaskLibrary()
