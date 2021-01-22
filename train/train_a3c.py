"""
Module to launch A3C algorithm.
Most of the code is taken from https://github.com/greydanus/baby-a3c.
"""

import os
import argparse
import logging

import torch
import torch.multiprocessing as mp

import utils.utility_funcs as utility_funcs

from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.finder import FinderEnv
from social_dilemmas.envs.treasure import TreasureEnv
from social_dilemmas.envs.target import TargetEnv

from algorithms.a3c import A3CPolicy, SharedAdam, SharedRMSprop, train

os.environ['OMP_NUM_THREADS'] = '1'

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
    parser.add_argument('--processes', default=8, type=int, help='number of processes to train with')
    parser.add_argument('--render', default=False, action='store_true', help='renders the atari environment')
    parser.add_argument('--test', default=False, action='store_true', help='sets lr=0, chooses most likely actions')
    parser.add_argument('--rnn-steps', default=20, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--optim', default='adam', type=str, help='can be either adam or rmsprop')
    parser.add_argument('--seed', default=1, type=int, help='set random seed')
    parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
    parser.add_argument('--tau', default=1.0, type=float, help='generalized advantage estimation discount')
    parser.add_argument('--horizon', default=8e7, type=float, help='max number of steps')
    parser.add_argument('--hidden', default=64, type=int, help='hidden size of GRU')
    parser.add_argument('--communication', default=False, action='store_true', help='add communication')
    parser.add_argument('--vocab', default=5, type=int, help='vocabulary size for communication')
    parser.add_argument('--view-size', default=-1, type=int, help='view size of agents (0 takes env default)')
    parser.add_argument('--noise', default=0., type=float, help='noise in comm channel')
    parser.add_argument('--save', default=None, type=str, help='save directory name')
    parser.add_argument('--cpu-only', default=False, action='store_true', help='prevent gpu usage')
    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method('spawn') # this must not be in global scope

    args = get_args()
    if args.render:
        args.processes = 1
        # args.test = True # render mode -> test mode w one process
    if args.test:
        args.lr = 0 # don't train in render mode
    if not args.communication:
        args.vocab = 0
    if args.cpu_only:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if args.save is None:
        args.save_dir = os.path.join(dir_path, os.path.join("saves", args.env))
    else:
        args.save_dir = os.path.join(dir_path, os.path.join("saves", args.save))

    if args.view_size == -1:
        args.view_size = None

    args.env_maker = env_map[args.env]
    single_env = args.env_maker(num_agents=1)
    args.num_actions = single_env.action_space.n # get the action space of this game
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir) # make dir to save models etc.

    logger = logging.getLogger('A3C')
    utility_funcs.setup_logger(logger, args)

    logger.debug(f"Run parameters:\n" +
        f"\t Env: {args.env}\n" +
        f"\t Processes: {args.processes}\n" +
        f"\t Agents: {args.agents}\n" +
        f"\t Learning rate: {args.lr}\n" +
        f"\t Communication: {args.communication}\n" +
        f"\t Vocab size: {args.vocab}\n" +
        f"\t LSTM size: {args.hidden}\n" +
        f"\t View size: {args.view_size}\n" +
        f"\t Seed: {args.seed}\n" +
        f"\t Env map:")

    logger.debug(single_env.base_map)

    logger.info("Creating shared models and optimizers.")
    torch.manual_seed(args.seed)
    shared_models = {
        f'agent-{i}': A3CPolicy(channels=3,
                                memsize=args.hidden,
                                num_actions=args.num_actions,
                                communication=args.communication,
                                vocab_size=args.vocab,
                                n_agents=args.agents).share_memory().to(device)
        for i in range(args.agents)
    }
    if args.optim == 'rmsprop':
        shared_optimizers = {
            f'agent-{i}': SharedRMSprop(shared_models[f'agent-{i}'].parameters(), lr=args.lr)
            for i in range(args.agents)
        }
    else:
        shared_optimizers = {
            f'agent-{i}': SharedAdam(shared_models[f'agent-{i}'].parameters(), lr=args.lr)
            for i in range(args.agents)
        }
    shared_schedulers = {
        f'agent-{i}': torch.optim.lr_scheduler.StepLR(shared_optimizers[f'agent-{i}'], 1000000, 0.99)
        for i in range(args.agents)
    }

    info = {
        info_name: torch.DoubleTensor([0]).share_memory_() 
        for info_name in ['run_epr', 'run_loss', 'episodes', 'frames', 'start_frames']
    }

    logger.info("Loading previous shared models parameters.")
    frames = []
    for agent_name, shared_model in shared_models.items():
        frames.append(shared_model.try_load(args.save_dir, agent_name, logger) * 1e6)
    if min(frames) != max(frames):
        logger.warning("Loaded models do not have the same number of training frames between agents")
    info['frames'] += max(frames)
    info['start_frames'] += max(frames)

    logger.info("Launching processes...")
    processes = []
    for rank in range(args.processes):
        p = mp.Process(target=train, args=(shared_models, shared_optimizers, shared_schedulers, rank, args, info))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
