import numpy as np

from social_dilemmas.envs.agent import TargetAgent  # TARGET_VIEW_SIZE
from social_dilemmas.constants import TARGET_MAP
from social_dilemmas.envs.map_env import MapEnv, ACTIONS

class TargetEnv(MapEnv):

    def __init__(self, ascii_map=TARGET_MAP, num_agents=1, render=False, share_reward=True, **kwargs):
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

        spawn_point = self.spawn_points[0]
        rotation = self.spawn_rotation(fixed=True)
        grid = map_with_agents
        if self.view_size is None:
            agent0 = TargetAgent('agent-0', spawn_point, rotation, grid)
        else:
            agent0 = TargetAgent('agent-0', spawn_point, rotation, grid, view_len=self.view_size)
        self.agents['agent-0'] = agent0

        if self.num_agents == 2:
            spawn_point = self.spawn_points[1]
            rotation = self.spawn_rotation()
            grid = map_with_agents
            agent1 = TargetAgent('agent-1', spawn_point, rotation, grid, view_len=np.max(grid.shape)-2)
            self.agents['agent-1'] = agent1

    def custom_reset(self):
        self.spawn_random_apple()

    def custom_action(self, agent, action):
        return []

    def custom_map_update(self):
        "See parent class"
        if self.agents['agent-0'].consumed:
            self.agents['agent-0'].consumed = False
            self.agents['agent-0'].reward_this_turn += 1
            self.spawn_random_apple()


    def spawn_random_apple(self, char='A'):
        map_with_agents = self.get_map_with_agents()
        spawned = False
        while not spawned:
            rand_coords = np.array([np.random.randint(1, len(self.world_map) - 1),
                                    np.random.randint(1, len(self.world_map[0]) - 1)])
            if map_with_agents[rand_coords[0]][rand_coords[1]] == ' ':
                self.update_map([(rand_coords[0], rand_coords[1], char)])
                spawned = True
        