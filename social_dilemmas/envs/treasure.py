import numpy as np

from social_dilemmas.envs.agent import TreasureAgent  # TREASURE_VIEW_SIZE
from social_dilemmas.constants import TREASURE_MAP
from social_dilemmas.envs.map_env import MapEnv, ACTIONS


class TreasureEnv(MapEnv):

    def __init__(self, ascii_map=TREASURE_MAP, num_agents=2, render=False, **kwargs):
        assert num_agents <= 2, "This env can only be played with 2 agents or less"
        self.view_size = kwargs.get('view_size')
        super().__init__(ascii_map, num_agents, render, **kwargs)
        self.treasure_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == 'A':
                    self.treasure_points.append([row, col])

    @property
    def action_space(self):
        agents = list(self.agents.values())
        return agents[0].action_space

    @property
    def observation_space(self):
        agents = list(self.agents.values())
        return agents[0].observation_space

    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            spawn_point = self.spawn_points[i]
            rotation = self.spawn_rotation(fixed=True)
            grid = map_with_agents
            if self.view_size is None:
                agent = TreasureAgent(agent_id, spawn_point, rotation, grid)
            else:
                agent = TreasureAgent(agent_id, spawn_point, rotation, grid, view_len=self.view_size)
            self.agents[agent_id] = agent

    def custom_reset(self):
        """Initialize the treasure"""
        self.spawn_treasure()

    def custom_map_update(self):
        "Spawn a new treasure when needed and update the other agent's reward"
        if self.agents['agent-0'].consumed:
            self.agents['agent-0'].consumed = False
            if len(self.agents) > 1:
                self.agents['agent-1'].reward_this_turn += 1.
            self.spawn_treasure()

    def spawn_treasure(self):
        shuffled_list = np.random.permutation(self.treasure_points)
        for row, col in shuffled_list:
            if (self.agents['agent-0'].pos != [row, col]).any():
                self.update_map([(row, col, 'A')])
                break


