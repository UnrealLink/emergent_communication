"""Launch training with chosen policy for each agent"""

import numpy as np
import os
import sys
import shutil
import argparse
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

import utils.utility_funcs as utility_funcs

from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.finder import FinderEnv

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
        elif env_name == 'finder':
            print('Initializing Finder environment')
            self.env = FinderEnv(num_agents=num_agents, render=True)
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
        times = []
        timer = 0
        obs, rewards, dones, info, = self.env.reset(), {}, {}, {}

        x_data = []
        y_data = []
        plt.plot(x_data, y_data)

        for i in tqdm(range(horizon)):
            timer += 1
            actions = {f"agent-{i}":self.agents_policies[f"agent-{i}"].step(
                obs[f"agent-{i}"],
                rewards.get(f"agent-{i}", 0),
                dones.get(f"agent-{i}", False),
                info,
            ) for i in range(self.num_agents)}
            
            for agent_policy in self.agents_policies.values():
                agent_policy.optimize()

            obs, rewards, dones, info, = self.env.step(actions)
            
            if rewards['agent-0'] > 0:
                times.append(timer)
                timer = 0
                if len(times)%10 == 0:
                    x_data.append(len(times)/10)
                    y_data.append(np.mean(times[-10:]))
                    plt.gca().lines[0].set_xdata(x_data)
                    plt.gca().lines[0].set_ydata(y_data)
                    plt.gca().relim()
                    plt.gca().autoscale_view()
                    plt.pause(0.05)

        plt.show()
        self.agents_policies['agent-0'].save()
        return times

    def render_rollout(self):

        eval_horizon = 50
        shape = self.env.world_map.shape
        full_obs = [np.zeros((shape[0], shape[1], 3), dtype=np.uint8) for i in range(eval_horizon)]
        
        obs, rewards, dones, info, = self.env.reset(), {}, {}, {}

        for i in tqdm(range(eval_horizon)):
            actions = {f"agent-{i}":self.agents_policies[f"agent-{i}"].step(
                obs[f"agent-{i}"],
                rewards.get(f"agent-{i}", 0),
                dones.get(f"agent-{i}", False),
                info,
            ) for i in range(self.num_agents)}

            obs, rewards, dones, info, = self.env.step(actions)

            rgb_arr = self.env.map_to_colors()
            full_obs[i] = rgb_arr.astype(np.uint8)

        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '/videos'
        print(path)
        utility_funcs.make_video_from_rgb_imgs(full_obs, path, fps=8, video_name='eval')


def main():
    args = parser.parse_args()
    controller = Controller(env_name=args.env, num_agents=args.agents)
    times = controller.rollout(horizon=args.horizon)
    controller.render_rollout()


if __name__ == '__main__':
    main()
