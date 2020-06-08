"""Launch training with chosen policy for each agent"""

import numpy as np
import os
import sys
import shutil
import argparse
from tqdm import tqdm

import utils.utility_funcs as utility_funcs

from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv

from algorithms.DQN import DQNAgent


parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, 
                    default='harvest',
                    help="Name of the environment to rollout. Can be cleanup or harvest.")
parser.add_argument("--agents", type=int, 
                    default=5,
                    help="Number of agents.")
parser.add_argument("--horizon", type=int, 
                    default=100,
                    help="Number of steps.")


class Controller(object):

    def __init__(self, env_name='cleanup', num_agents=5):
        self.env_name = env_name
        self.env = None
        if env_name == 'harvest':
            print('Initializing Harvest environment')
            self.env = HarvestEnv(num_agents=num_agents, render=False)
        elif env_name == 'cleanup':
            print('Initializing Cleanup environment')
            self.env = CleanupEnv(num_agents=num_agents, render=False)
        else:
            print('Error! Not a valid environment type')
            return

        self.num_agents = num_agents
        self.agents_policies = {f"agent-{i}":DQNAgent(self.env.agents[f"agent-{i}"]) 
                                for i in range(self.num_agents)}


    def rollout(self, horizon=100):
        """ Rollout several timesteps of an episode of the environment.

        Args:
            horizon: The number of timesteps to roll out.
        """
        reward_hist = []

        obs, rewards, dones, info, = self.env.reset(), {}, {}, {}

        for i in tqdm(range(horizon)):
            actions = {f"agent-{i}":self.agents_policies[f"agent-{i}"].step(
                obs[f"agent-{i}"],
                rewards.get(f"agent-{i}", 0),
                dones.get(f"agent-{i}", False),
                info,
            ) for i in range(self.num_agents)}
            
            for agent_policy in self.agents_policies.values():
                agent_policy.optimize()

            obs, rewards, dones, info, = self.env.step(actions)

            reward_hist.append(rewards['agent-0'])

        return reward_hist


def main():
    args = parser.parse_args()
    controller = Controller(env_name=args.env, num_agents=args.agents)
    rewards = controller.rollout(horizon=args.horizon)
    print(np.mean(np.array(rewards).reshape(-1, 100), axis=1))


if __name__ == '__main__':
    main()
