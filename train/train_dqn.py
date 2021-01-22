"""
Launch training with DQN.
"""

import os
import sys
import shutil
import argparse
import logging
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

import utils.utility_funcs as utility_funcs

from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.finder import FinderEnv
from social_dilemmas.envs.treasure import TreasureEnv
from social_dilemmas.envs.target import TargetEnv

from algorithms.dqn import DQNAgent

env_map = {
    'finder': FinderEnv,
    'treasure': TreasureEnv,
    'harvest': HarvestEnv,
    'cleanup': CleanupEnv,
    'target': TargetEnv,
}

def get_args():
    """
    Get arguments
    """
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='finder', type=str, help='environment name')
    parser.add_argument('--agents', default=2, type=int, help='number of agents in environment')
    parser.add_argument('--render', default=False, action='store_true', help='renders the atari environment')
    parser.add_argument('--test', default=False, action='store_true', help='sets lr=0, chooses most likely actions')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='set random seed')
    parser.add_argument('--horizon', default=3000, type=int, help='max number of steps')
    parser.add_argument('--view-size', default=-1, type=int, help='view size of agents (0 takes env default)')
    parser.add_argument('--save', default=None, type=str, help='save directory name')
    parser.add_argument('--checkpoint', default=None, type=int, help='checkpoint to load in save dir (default max)')
    parser.add_argument('--cpu-only', default=False, action='store_true', help='prevent gpu usage')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.render:
        args.test = True
    if args.test:
        args.lr = 0 # don't train in test mode
    if args.cpu_only:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if args.save is None:
        args.save_dir = os.path.join(dir_path, os.path.join("saves", args.env))
    else:
        args.save_dir = os.path.join(dir_path, os.path.join("saves", args.save))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    env = env_map[args.env](num_agents=args.agents)    
    agents = {f"agent-{i}":DQNAgent(env.agents[f"agent-{i}"], lr=args.lr) for i in range(args.agents)}

    logger = logging.getLogger('DQN')
    utility_funcs.setup_logger(logger, args)

    logger.debug(f"Run parameters:\n" +
        f"\t Env: {args.env}\n" +
        f"\t Agents: {args.agents}\n" +
        f"\t Learning rate: {args.lr}\n" +
        f"\t Seed: {args.seed}\n" +
        f"\t Env map:")

    logger.debug(env.base_map)

    for agent_name, policy in agents.items():
        policy.try_load(args.save_dir, agent_name, logger, args.checkpoint)

    obs, rewards, dones, info, = env.reset(), {}, {}, {}

    for i in tqdm(range(args.horizon)):
        actions = {f"agent-{i}":agents[f"agent-{i}"].step(
            obs[f"agent-{i}"],
            rewards.get(f"agent-{i}", 0),
            dones.get(f"agent-{i}", False),
            info,
        ) for i in range(args.agents)}
        
        for agent_policy in agents.values():
            agent_policy.optimize()
            
        if args.render:
            env.render()
            time.sleep(0.5)

        obs, rewards, dones, info, = env.step(actions)
    
    for agent_name, policy in agents.items():
        path = os.path.join(args.save_dir, f'model.{agent_name}.{args.horizon}.tar')
        policy.save(path)
