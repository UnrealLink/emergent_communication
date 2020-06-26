import numpy as np

from social_dilemmas.envs.agent import FinderAgent  # FINDER_VIEW_SIZE
from social_dilemmas.constants import FINDER_MAP
from social_dilemmas.envs.map_env import MapEnv, ACTIONS

class FinderEnv(MapEnv):

    def __init__(self, ascii_map=FINDER_MAP, num_agents=1, render=False, **kwargs):
        super().__init__(ascii_map, num_agents, render, **kwargs)
        self.apple_points = None
        self.timer = 0

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
            agent = FinderAgent(agent_id, spawn_point, rotation, grid)
            self.agents[agent_id] = agent

    def custom_reset(self):
        self.spawn_random_apple()

    def custom_action(self, agent, action):
        return []

    def custom_map_update(self):
        "See parent class"
        # spawn an apple if needed
        self.timer += 1
        needed = False
        for agent in self.agents.values():
            if agent.consumed:
                needed = True
                agent.consumed = False
                break
        if needed:
            self.timer = 0
            self.spawn_random_apple()

    def spawn_random_apple(self):
        spawned = False
        while not(spawned):
            rand_coords = np.array([np.random.randint(1, len(self.world_map) - 1), 
                                    np.random.randint(1, len(self.world_map[0]) - 1)])
            conflict = False
            for agent in self.agents.values():
                if (agent.pos == rand_coords).all():
                    conflict = True
            if not(conflict):
                self.update_map([(rand_coords[0], rand_coords[1], 'A')])
                spawned = True
        