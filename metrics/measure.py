"""
Script to measure instantaneous coordination from a trained model from A3C.
"""

import os
import argparse
import logging

import torch

from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.finder import FinderEnv

import utils.utility_funcs as utility_funcs

from algorithms.a3c import A3CPolicy, preprocess_messages, preprocess_obs

env_map = {
    'finder': FinderEnv,
    'harvest': HarvestEnv,
    'cleanup': CleanupEnv,
}

def get_args():
    """
    Get arguments
    """
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='finder', type=str, help='environment name')
    parser.add_argument('--agents', default=5, type=int, help='number of agents in environment')
    parser.add_argument('--horizon', default=1000, type=int, help='number of steps to measure on')
    parser.add_argument('--seed', default=1, type=int, help='set random seed')
    parser.add_argument('--vocab', default=10, type=int, help='vocabulary size for communication')
    parser.add_argument('--hidden', default=128, type=int, help='hidden size of GRU')
    parser.add_argument('--save', default=None, type=str, help='save directory name')
    parser.add_argument('--cpu-only', default=False, action='store_true', help='prevent gpu usage')
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

    logger = logging.getLogger('Measure')
    utility_funcs.setup_logger(logger, args)

    logger.info("Loading env and agents...")
    env = env_map[args.env](num_agents=args.agents)
    env.seed(args.seed)
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

    logger.info("Starting rollout...")
    obs = env.reset()
    states = {
        agent_name: preprocess_obs(ob, device=device)
        for agent_name, ob in obs.items()
    }
    hxs = {
        agent_name: torch.zeros(1, args.hidden).to(device)
        for agent_name in models.keys()
    }
    messages = {
        agent_name: 0
        for agent_name in models.keys()
    }

    for _ in range(args.horizon):
        actions = {}
        flat_messages = preprocess_messages(messages, args.vocab, device=device)
        for agent_name, model in models.items():
            value, logp, comm_value, comm_logp, hx = model((states[agent_name],
                                                            flat_messages,
                                                            hxs[agent_name]))
            hxs[agent_name] = hx

            action = torch.exp(logp).multinomial(num_samples=1).data[0]
            actions[agent_name] = action.cpu().numpy()[0]

            message = torch.exp(logp).multinomial(num_samples=1).data[0]
            messages[agent_name] = message.cpu().numpy()[0]

        obs, rewards, dones, _ = env.step(actions)

        states = {
            agent_name: preprocess_obs(ob, device=device)
            for agent_name, ob in obs.items()
        }

        if dones["__all__"]:
            obs = env.reset()
            states = {
                agent_name: preprocess_obs(ob, device=device)
                for agent_name, ob in obs.items()
            }
            hxs = {
                agent_name: torch.zeros(1, 256).to(device)
                for agent_name in models.keys()
            }
            messages = {
                agent_name: 0
                for agent_name in models.keys()
            }

    logger.info("Rollout done.")
