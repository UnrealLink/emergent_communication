import numpy as np

from social_dilemmas.envs.agent import FinderAgent  # FINDER_VIEW_SIZE
from social_dilemmas.constants import FINDER_MAP
from social_dilemmas.envs.map_env import MapEnv, ACTIONS

class FinderEnv(MapEnv):

    def __init__(self, ascii_map=FINDER_MAP, num_agents=1, render=False, share_reward=True, **kwargs):
        self.view_size = kwargs.get('view_size')
        super().__init__(ascii_map, num_agents, render, **kwargs)
        self.apple_points = None
        self.share_reward = share_reward

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
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            grid = map_with_agents
            if self.view_size is None:
                agent = FinderAgent(agent_id, spawn_point, rotation, grid)
            else:
                agent = FinderAgent(agent_id, spawn_point, rotation, grid, view_len=self.view_size)
            self.agents[agent_id] = agent

    def custom_reset(self):
        for agent in self.agents.values():
            self.spawn_random_apple(chr(agent.int_id + 65))

    def custom_action(self, agent, action):
        return []

    def custom_map_update(self):
        "See parent class"
        reward = 0
        for agent in self.agents.values():
            if agent.consumed:
                agent.consumed = False
                if self.share_reward:
                    reward += 1
                else:
                    agent.reward_this_turn += 1
                self.spawn_random_apple(chr(agent.int_id + 65))
        if self.share_reward:
            for agent in self.agents.values():
                agent.reward_this_turn += reward
        

    def spawn_random_apple(self, char='A'):
        spawned = False
        while not(spawned):
            rand_coords = np.array([np.random.randint(1, len(self.world_map) - 1), 
                                    np.random.randint(1, len(self.world_map[0]) - 1)])
            if self.world_map[rand_coords[0]][rand_coords[1]] == ' ':
                self.update_map([(rand_coords[0], rand_coords[1], char)])
                spawned = True
        