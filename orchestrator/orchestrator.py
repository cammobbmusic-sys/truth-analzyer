

from agents.factory import create_agent



class Orchestrator:

    '''

    Orchestrates multiple agents and pipelines.

    '''



    def __init__(self, agent_configs: list = None, dry_run=True):

        self.agent_configs = agent_configs or []

        self.agents = []

        if not dry_run:

            for cfg in self.agent_configs:

                if cfg is not None:

                    self.agents.append(create_agent(cfg))



    def run_all_agents(self, prompt: str):

        '''

        Run all agents with the given prompt (simulated responses for safety)

        Returns empty dict if no agents are instantiated (dry_run mode)

        '''

        results = {}

        if not self.agents:

            return results  # Dry run mode - no agents instantiated

        for agent in self.agents:

            results[agent.name] = agent.generate(prompt)

        return results

