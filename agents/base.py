

from abc import ABC, abstractmethod



class ModelAgent(ABC):

    '''

    Abstract base class for all AI agents.

    '''



    def __init__(self, name: str, provider: str, model: str, role: str = "agent", timeout: int = 15):

        self.name = name

        self.provider = provider

        self.model = model

        self.role = role

        self.timeout = timeout



    @abstractmethod

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 256) -> str:

        '''

        Generate a response given a prompt.

        Must be implemented by all concrete adapters.

        '''

        pass

    def run(self, *args, **kwargs):
        """Run the agent."""
        pass

    def analyze(self, *args, **kwargs):
        """Analyze data or input."""
        pass

    def process(self, *args, **kwargs):
        """Process input and return results."""
        pass

