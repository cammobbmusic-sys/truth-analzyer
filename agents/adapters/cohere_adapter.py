import os
import requests
from agents.base import ModelAgent


class CohereAdapter(ModelAgent):
    """
    Cohere AI Agent Adapter

    Uses Cohere's Chat API for text generation (Generate API was deprecated).
    Set COHERE_API_KEY environment variable.

    Models available: command-r-plus, command-r, command, command-light, etc.
    """

    def __init__(self, name: str, provider: str, model: str, role: str = "agent", timeout: int = 15):
        super().__init__(name, provider, model, role, timeout)

        self.api_key = os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing COHERE_API_KEY environment variable. "
                "Please create a `.env` file with: COHERE_API_KEY=your_key_here"
            )

        self.api_url = "https://api.cohere.com/v1/chat"

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 256) -> str:
        """
        Generate response using Cohere Chat API.
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "message": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("text", "").strip()

            elif response.status_code == 401:
                return "[COHERE AUTH ERROR] Invalid API key"

            elif response.status_code == 429:
                return "[COHERE RATE LIMITED] Too many requests"

            elif response.status_code == 400:
                error_data = response.json()
                error_msg = error_data.get("message", "Bad request")
                return f"[COHERE API ERROR] {error_msg}"

            else:
                return f"[COHERE HTTP {response.status_code}] {response.text[:100]}"

        except requests.RequestException as e:
            return f"[COHERE NETWORK ERROR] {str(e)}"

        except Exception as e:
            return f"[COHERE UNEXPECTED ERROR] {str(e)}"
