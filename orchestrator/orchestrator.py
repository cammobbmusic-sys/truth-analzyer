

from agents.factory import create_agent



class Orchestrator:

    '''

    Orchestrates multiple agents and pipelines.

    '''



    def __init__(self, agent_configs: list = None):

        self.agent_configs = agent_configs or []

        self.agents = []

        for cfg in self.agent_configs:

            if cfg is not None:

                self.agents.append(create_agent(cfg))



    def run_all_agents(self, prompt: str):

        '''

        Run all agents with the given prompt (simulated responses for safety)

        '''

        results = {}

        for agent in self.agents:

            results[agent.name] = agent.generate(prompt)

        return results

