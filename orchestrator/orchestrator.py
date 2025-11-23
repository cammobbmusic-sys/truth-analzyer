

from agents.factory import create_agent



class Orchestrator:

    '''

    Orchestrates multiple agents and pipelines.

    '''



    def __init__(self, agent_configs: list):

        # List of agent configs (dicts)

        self.agent_configs = agent_configs

        # Instantiate agents dynamically

        self.agents = [create_agent(cfg) for cfg in agent_configs]



    def run_all_agents(self, prompt: str):

        '''

        Run all agents with the given prompt (simulated responses for safety)

        '''

        results = {}

        for agent in self.agents:

            results[agent.name] = agent.generate(prompt)

        return results

