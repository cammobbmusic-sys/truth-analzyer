

import os
import requests
from agents.base import ModelAgent


class HuggingFaceAdapter(ModelAgent):

    """

    HuggingFace Inference API Agent Adapter

    Uses HF Inference API for free/community models.

    Set HF_TOKEN environment variable for authenticated requests (optional).

    """

    def __init__(self, name: str, provider: str, model: str, role: str = "agent", timeout: int = 15):
        super().__init__(name, provider, model, role, timeout)

        self.api_key = os.getenv("HF_TOKEN")  # Optional: for higher rate limits
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model}"

        # Headers for API request
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 256) -> str:
        """

        Generate response using HuggingFace Inference API.

        Note: HF Inference API has rate limits and may queue requests.

        """

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": min(max_tokens, 512),  # HF limit
                "temperature": temperature,
                "do_sample": temperature > 0.1,
                "return_full_text": False
            },
            "options": {
                "wait_for_model": True,
                "use_cache": True
            }
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()

                # Handle different response formats
                if isinstance(data, list) and len(data) > 0:
                    # Standard text generation format
                    if "generated_text" in data[0]:
                        return data[0]["generated_text"].strip()
                    # Conversational format
                    elif "conversation" in data[0]:
                        return data[0]["conversation"]["generated_responses"][-1].strip()

                # Fallback
                return str(data).strip()

            elif response.status_code == 410:
                # API deprecated
                return f"[HF API DEPRECATED] HuggingFace Inference API is no longer available. Consider using Together AI or OpenRouter for {self.model}."

            elif response.status_code == 503:
                # Model loading - HF returns this when model needs to load
                return f"[HF MODEL LOADING] {self.model} is loading on HuggingFace. Please retry in a few moments."

            elif response.status_code == 429:
                # Rate limited
                return f"[HF RATE LIMITED] Too many requests to HuggingFace API. Please wait before retrying."

            else:
                error_msg = response.text[:200] if response.text else "Unknown error"
                return f"[HF API ERROR {response.status_code}] {error_msg}"

        except requests.RequestException as e:
            return f"[HF NETWORK ERROR] {str(e)}"

        except Exception as e:
            return f"[HF UNEXPECTED ERROR] {str(e)}"

