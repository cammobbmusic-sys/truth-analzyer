import os
import requests
from agents.base import ModelAgent


class OpenRouterAdapter(ModelAgent):

    """

    OpenRouter API Agent Adapter

    OpenRouter provides access to multiple AI models through a unified API.

    Set OPENROUTER_API_KEY environment variable.

    Visit: https://openrouter.ai/ for API key and supported models.

    """

    def __init__(self, name: str, provider: str, model: str, role: str = "agent", timeout: int = 15):
        super().__init__(name, provider, model, role, timeout)

        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing OPENROUTER_API_KEY environment variable. "
                "Get one from: https://openrouter.ai/keys"
            )

        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 256) -> str:
        """

        Generate response using OpenRouter API.

        Supports various models like gpt-4, claude, gemini, etc.

        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost:5000",  # Required by OpenRouter
            "X-Title": "Multi-Agent Truth Analyzer"    # Optional identifier
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1,
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

                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        return choice["message"]["content"].strip()

                return f"[OPENROUTER UNEXPECTED RESPONSE] {str(data)}"

            else:
                error_data = response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                error_msg = error_data.get('error', {}).get('message', response.text)
                return f"[OPENROUTER API ERROR {response.status_code}] {error_msg}"

        except requests.RequestException as e:
            return f"[OPENROUTER NETWORK ERROR] {str(e)}"

        except Exception as e:
            return f"[OPENROUTER UNEXPECTED ERROR] {str(e)}"
