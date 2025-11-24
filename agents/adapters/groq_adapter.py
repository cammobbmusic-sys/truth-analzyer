import os
import requests
from agents.base import ModelAgent


class GroqAdapter(ModelAgent):
    """
    Groq API Agent Adapter

    Uses the GROQ_API_KEY from environment variables.
    This avoids storing any secrets in code or commits.

    Setup:
    1. Get API key from https://console.groq.com/keys
    2. Set environment variable: export GROQ_API_KEY=your_key_here
    3. Or create .env file with: GROQ_API_KEY=your_key_here

    Supported models: llama-3.1-8b-instant, llama-3.3-70b-versatile, etc.
    See https://console.groq.com/docs/models for full list.
    """

    def __init__(self, name: str, provider: str, model: str, role: str = "agent", timeout: int = 15):
        super().__init__(name, provider, model, role, timeout)

        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing GROQ_API_KEY environment variable. "
                "Please create a `.env` file with: GROQ_API_KEY=your_key_here"
            )

        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 256) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout)

            if response.status_code != 200:
                raise RuntimeError(f"Groq error: {response.status_code} {response.text}")

            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        except requests.RequestException as e:
            raise RuntimeError(f"Network error calling Groq API: {e}")
