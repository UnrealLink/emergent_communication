"""
Script to measure instantaneous coordination from a trained model from A3C.
"""

import os
import argparse
import logging
import random
import time
import numpy as np
import matplotlib.pyplot as plt

import torch

from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.finder import FinderEnv
from social_dilemmas.envs.treasure import TreasureEnv
from social_dilemmas.envs.target import TargetEnv

import utils.utility_funcs as utility_funcs

from algorithms.a3c import A3CPolicy, preprocess_messages, preprocess_obs, get_comm

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
    parser.add_argument('--agents', default=1, type=int, help='number of agents in environment')
    parser.add_argument('--horizon', default=1000, type=int, help='number of steps to measure on')
    parser.add_argument('--seed', default=1, type=int, help='set random seed')
    parser.add_argument('--vocab', default=5, type=int, help='vocabulary size for communication')
    parser.add_argument('--view-size', default=0, type=int, help='view size of agents (0 takes env default)')
    parser.add_argument('--hidden', default=64, type=int, help='hidden size of GRU')
    parser.add_argument('--noise', default=0., type=float, help='noise in comm channel')
    parser.add_argument('--save', default=None, type=str, help='save directory name')
    parser.add_argument('--cpu-only', default=False, action='store_true', help='prevent gpu usage')
    parser.add_argument('--render', default=False, action='store_true', help='render at each time step')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if args.save is None:
        args.save_dir = os.path.join(dir_path, f'saves/{args.env}')
    else:
        args.save_dir = os.path.join(dir_path, f'saves/{args.save}')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.communication = True

    if args.cpu_only:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.view_size == 0:
        args.view_size = None

    logger = logging.getLogger('Measure')
    utility_funcs.setup_logger(logger, args)

    logger.debug("Loading env and agents...")
    env = env_map[args.env](num_agents=args.agents, seed=args.seed, view_size=args.view_size)
    torch.manual_seed(args.seed)
    args.num_actions = env.action_space.n

    models = {
        f'agent-{i}': A3CPolicy(channels=3,
                                memsize=args.hidden,
                                num_actions=args.num_actions,
                                communication=args.communication,
                                vocab_size=args.vocab,
                                n_agents=args.agents).to(device)
        for i in range(args.agents)
    }

    for agent_name, model in models.items():
        if model.try_load(args.save_dir, agent_name, logger) == 0:
            logger.warning("No trained models found.")
        model.eval()

    # Matrices for tracking messages/actions co-occurences
    IC = np.zeros((args.vocab, args.num_actions))
    A = np.zeros(args.num_actions)
    M = np.zeros(args.vocab)

    logger.debug("Starting rollout...")
    obs = env.reset()
    states = {
        agent_name: preprocess_obs(ob, device=device)
        for agent_name, ob in obs.items()
    }
    messages = {
        agent_name: 0
        for agent_name in models.keys()
    }

    # fig = plt.gcf()

    for _ in range(args.horizon):
        actions = {}
        messages = {}
        get_comm(env, messages, args.noise)
        flat_messages = preprocess_messages(messages, args.vocab, device=device)
        for agent_name, model in models.items():
            value, logp = model((states[agent_name], flat_messages))

            action = torch.argmax(logp).data
            actions[agent_name] = int(action.cpu().numpy())

        # Update measures here
        IC[messages[f"agent-0"]][actions[f"agent-0"]] += 1
        A[actions['agent-0']] += 1
        M[messages['agent-0']] += 1

        obs, rewards, dones, _ = env.step(actions)

        states = {
            agent_name: preprocess_obs(ob, device=device)
            for agent_name, ob in obs.items()
        }

        if args.render:
            env.render()
            time.sleep(0.5)

        # fig.clf()

        # map_with_agents = env.get_map_with_agents()
        # rgb_arr = env.map_to_colors(map_with_agents)
        # plt.imshow(rgb_arr, interpolation='nearest')
        # plt.imshow(states['agent-0'].cpu().numpy().squeeze(0).transpose((1, 2, 0))/255, interpolation='nearest')

        # fig.show()
        # fig.canvas.draw()

        if dones["__all__"]:
            obs = env.reset()
            states = {
                agent_name: preprocess_obs(ob, device=device)
                for agent_name, ob in obs.items()
            }
            messages = {
                agent_name: 0
                for agent_name in models.keys()
            }

    logger.debug("Rollout done.")
    logger.debug("Computing measures")

    ic = 0
    for m in range(args.vocab):
        for a in range(args.num_actions):
            ic += 0 if IC[m, a] == 0 else IC[m, a]/args.horizon * np.log(IC[m, a]/(np.sum(IC[m, :])*np.sum(IC[:, a])/args.horizon))
    logger.info(f"IC: {ic}")
    # print(IC)
    # print(A)
    # print(M)
