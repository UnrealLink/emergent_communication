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

from algorithms.a3c import A3CPolicy, EasySpeakerPolicy, preprocess_messages, preprocess_obs, get_comm

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
    parser.add_argument('--env', default='target', type=str, help='environment name')
    parser.add_argument('--agents', default=1, type=int, help='number of agents in environment')
    parser.add_argument('--horizon', default=1000, type=int, help='number of steps to measure on')
    parser.add_argument('--seed', default=0, type=int, help='set random seed')
    parser.add_argument('--vocab', default=5, type=int, help='vocabulary size for communication')
    parser.add_argument('--view-size', default=0, type=int, help='view size of agents (0 takes env default)')
    parser.add_argument('--hidden', default=64, type=int, help='hidden size of GRU')
    parser.add_argument('--noise', default=0., type=float, help='noise in comm channel')
    parser.add_argument('--save', default=None, type=str, help='save directory name')
    parser.add_argument('--checkpoint', default=None, type=int, help='checkpoint to load in save dir (default max)')
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

    logger.info("Loading env and agents...")
    env = env_map[args.env](num_agents=args.agents, seed=args.seed, view_size=args.view_size)
    torch.manual_seed(args.seed)
    args.num_actions = env.action_space.n

    models = {
        'agent-0': A3CPolicy(channels=3,
                             memsize=args.hidden,
                             num_actions=args.num_actions,
                             vocab_size=args.vocab,
                             n_agents=args.agents).share_memory().to(device),
        'agent-1': EasySpeakerPolicy(input=args.vocab, vocab_size=args.vocab).to(device)
    }

    for agent_name, model in models.items():
        if model.try_load(args.save_dir, agent_name, logger, checkpoint=args.checkpoint) == 0:
            logger.warning("No trained models found.")
        model.eval()

    # Matrices for tracking messages/actions co-occurences
    CIC = 0.

    logger.info("Starting rollout...")
    obs = env.reset()
    states = {
        agent_name: preprocess_obs(ob, device=device)
        for agent_name, ob in obs.items()
    }

    for _ in range(args.horizon):
        actions = {}
        messages = {}
        get_comm(env, messages, args.noise)
        perfect_message = preprocess_messages(messages, args.vocab, device=device)

       # get speaker message
        comm_value, comm_logp = models['agent-1'](perfect_message)

        message = torch.exp(comm_logp).multinomial(num_samples=1).data[0]
        messages['agent-0'] = message.cpu().numpy()[0]

        real_messages = preprocess_messages(messages, args.vocab, device=device)

        # select listener action
        value, logp = models['agent-0']((states['agent-0'], real_messages))
        action = torch.exp(logp).multinomial(num_samples=1).data[0]
        actions['agent-0'] = action.cpu().numpy()[0]

        # Update measures here
        pas, pms = [0]*args.num_actions, [0]*args.vocab
        pams = np.zeros((args.num_actions, args.vocab))
        for a in range(args.num_actions):
            for m in range(args.vocab):
                pms[m] = pms[m] if pms[m] != 0 else torch.exp(comm_logp[0][m]).detach().cpu().numpy()
                fake_messages = {'agent-1' : torch.tensor([m]).to(device)}

                fake_transmission = preprocess_messages(fake_messages, args.vocab, device=device)
                fake_value, fake_logp = models['agent-0']((states['agent-0'], fake_transmission))
                
                pams[a, m] = torch.exp(fake_logp[0][a]).detach().cpu().numpy() * pms[m]
                pas[a] += pams[a, m] / args.num_actions
        for a in range(args.num_actions):
            for m in range(args.vocab):
                if pams[a, m] * pas[a] * pms[m] != 0:
                    CIC += pams[a, m] * np.log(pams[a, m] / (pas[a] * pms[m])) / args.horizon
                else:
                    logger.debug('zero value in probas')


        obs, rewards, dones, _ = env.step(actions)

        states = {
            agent_name: preprocess_obs(ob, device=device)
            for agent_name, ob in obs.items()
        }

        if args.render:
            env.render()
            time.sleep(0.5)

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

    logger.info("Rollout done.")
    logger.info("Computing measures")

    logger.info(f"CIC: {CIC}")
