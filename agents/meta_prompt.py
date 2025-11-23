"""
Meta-Prompt - Dynamically constructs prompts using template files.
"""

import os
from typing import Optional


class MetaPrompt:
    """Handles dynamic prompt construction using template files."""

    def __init__(self, templates_path: str = "data/prompts/"):
        """
        Initialize MetaPrompt with templates directory.

        Args:
            templates_path: Path to the directory containing prompt templates
        """
        self.templates_path = templates_path
        # Ensure templates directory exists
        os.makedirs(templates_path, exist_ok=True)

    def load_template(self, template_name: str) -> str:
        """
        Load a template from file.

        Args:
            template_name: Name of the template file (without .txt extension)

        Returns:
            Template content as string

        Raises:
            FileNotFoundError: If template file doesn't exist
        """
        template_path = os.path.join(self.templates_path, f"{template_name}.txt")

        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Template '{template_name}' not found at {template_path}")

    def generate_prompt(self, task: str, context: str, model_role: str) -> str:
        """
        Dynamically constructs prompts for a given model role and task.

        Args:
            task: The task type (e.g., 'verification', 'brainstorming')
            context: The specific context or query
            model_role: The role of the model (e.g., 'expert', 'creative')

        Returns:
            Constructed prompt with placeholders replaced
        """
        try:
            template = self.load_template(task)
        except FileNotFoundError:
            # Fallback to a basic template if specific template doesn't exist
            template = self._get_fallback_template(task)

        # Replace placeholders
        prompt = template.replace("{context}", context).replace("{role}", model_role)

        return prompt.strip()

    def _get_fallback_template(self, task: str) -> str:
        """
        Provide fallback templates when specific template files don't exist.

        Args:
            task: The task type

        Returns:
            Basic template content
        """
        templates = {
            "verification": """
You are a {role} analyzing the following question.

Question: {context}

Please provide:
1. Your direct answer
2. Explanation with evidence
3. Confidence level (0-1)
4. Key assumptions made
""",
            "brainstorming": """
You are a {role} generating creative ideas.

Topic: {context}

Generate 5 diverse ideas with:
• Brief description
• Potential benefits
• Implementation considerations
""",
            "analysis": """
You are a {role} performing analysis.

Topic: {context}

Provide a comprehensive analysis including:
• Key findings
• Implications
• Recommendations
"""
        }

        return templates.get(task, f"""
You are a {{role}} working on: {{context}}

Please provide your analysis and insights.
""")

    def save_template(self, template_name: str, content: str):
        """
        Save a template to file.

        Args:
            template_name: Name of the template (without .txt extension)
            content: Template content to save
        """
        template_path = os.path.join(self.templates_path, f"{template_name}.txt")

        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(content)

    def list_templates(self) -> list:
        """
        List all available templates.

        Returns:
            List of template names (without .txt extension)
        """
        if not os.path.exists(self.templates_path):
            return []

        templates = []
        for filename in os.listdir(self.templates_path):
            if filename.endswith('.txt'):
                templates.append(filename[:-4])  # Remove .txt extension

        return sorted(templates)