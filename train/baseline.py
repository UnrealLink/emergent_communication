import argparse

import ray
from ray import tune
from ray.tune import run_experiments
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy

from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.cleanup import CleanupEnv

from models.base_model import BaseModel

parser = argparse.ArgumentParser()

parser.add_argument('--exp-name', type=str, default=None,
                    help='Name of the ray_results experiment directory where results are stored.')
parser.add_argument("--env", type=str, default='harvest',
                    help='Name of the environment to rollout. Can be cleanup or harvest.')
parser.add_argument('--algorithm', type=str, default='A3C',
                    help='Name of the rllib algorithm to use.')
parser.add_argument('--num-agents', type=int, default=5,
                    help='Number of agent policies')
parser.add_argument('--train-batch-size', type=int, default=30000,
                    help='Size of the total dataset over which one epoch is computed.')
parser.add_argument('--checkpoint-frequency', type=int, default=20,
                    help='Number of steps before a checkpoint is saved.')
parser.add_argument('--training-iterations', type=int, default=25,
                    help='Total number of steps to train for')
parser.add_argument('--num-cpus', type=int, default=2,
                    help='Number of available CPUs')
parser.add_argument('--num-gpus', type=int, default=1,
                    help='Number of available GPUs')
parser.add_argument('--use-gpus-for-workers', action="store_true",
                    help='Set to true to run workers on GPUs rather than CPUs')
parser.add_argument('--use-gpu-for-driver', action="store_true",
                    help='Set to true to run driver on GPU rather than CPU.')
parser.add_argument('--num-workers-per-device', type=float, default=2.,
                    help='Number of workers to place on a single device (CPU or GPU)')

harvest_default_params = {
    'lr_init': 0.00136,
    'lr_final': 0.000028,
    'entropy_coeff': .000687}

cleanup_default_params = {
    'lr_init': 0.00126,
    'lr_final': 0.000012,
    'entropy_coeff': .00176}


def setup(env, hparams, algorithm, train_batch_size, num_cpus, num_gpus,
          num_agents, use_gpus_for_workers=False, use_gpu_for_driver=False,
          num_workers_per_device=1):

    if env == 'harvest':
        def env_creator(_):
            return HarvestEnv(num_agents=num_agents)
        single_env = HarvestEnv()
    else:
        def env_creator(_):
            return CleanupEnv(num_agents=num_agents)
        single_env = CleanupEnv()

    env_name = env + "_env"
    register_env(env_name, env_creator)

    obs_space = single_env.observation_space
    act_space = single_env.action_space

    # Each policy can have a different configuration (including custom model)
    def gen_policy():
        return (PPOTFPolicy, obs_space, act_space, {})

    # Setup PPO with an ensemble of `num_policies` different policies
    policies = {}
    for i in range(num_agents):
        policies['agent-' + str(i)] = gen_policy()

    def policy_mapping_fn(agent_id):
        return agent_id

    # # register the custom model
    # model_name = "conv_to_fc_net"
    # ModelCatalog.register_custom_model(model_name, ConvToFCNet)

    agent_cls = get_agent_class(algorithm)
    config = agent_cls._default_config.copy()

    # information for replay
    config['env_config']['func_create'] = env_creator
    config['env_config']['env_name'] = env_name
    config['env_config']['run'] = algorithm

    # Calculate device configurations
    gpus_for_driver = int(use_gpu_for_driver)
    cpus_for_driver = 1 - gpus_for_driver
    if use_gpus_for_workers:
        spare_gpus = (num_gpus - gpus_for_driver)
        num_workers = int(spare_gpus * num_workers_per_device)
        num_gpus_per_worker = spare_gpus / num_workers
        num_cpus_per_worker = 0
    else:
        spare_cpus = (num_cpus - cpus_for_driver)
        num_workers = int(spare_cpus * num_workers_per_device)
        num_gpus_per_worker = 0
        num_cpus_per_worker = spare_cpus / num_workers

    # hyperparams
    config.update({
                "train_batch_size": train_batch_size,
                "horizon": 1000,
                "lr_schedule":
                [[0, hparams['lr_init']],
                    [20000000, hparams['lr_final']]],
                "num_workers": num_workers,
                "num_gpus": gpus_for_driver,  # The number of GPUs for the driver
                "num_cpus_for_driver": cpus_for_driver,
                "num_gpus_per_worker": num_gpus_per_worker,   # Can be a fraction
                "num_cpus_per_worker": num_cpus_per_worker,   # Can be a fraction
                "entropy_coeff": hparams['entropy_coeff'],
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": policy_mapping_fn,
                },
                "model" : {
                    "conv_filters":[
                        [6, [3, 3], 1],
                        [32, [15, 15], 1]
                    ],
                    "fcnet_hiddens": [32, 32],
                    "use_lstm": True,
                    "lstm_cell_size": 128,
                },
                "use_pytorch":True,
                "use_exec_api": False
    })
    return algorithm, env_name, config


def main():
    args = parser.parse_args()
    ray.init(num_cpus=args.num_cpus)
    if args.env == 'harvest':
        hparams = harvest_default_params
    else:
        hparams = cleanup_default_params
    alg_run, env_name, config = setup(args.env, hparams, args.algorithm,
                                      args.train_batch_size,
                                      args.num_cpus,
                                      args.num_gpus, args.num_agents,
                                      args.use_gpus_for_workers,
                                      args.use_gpu_for_driver,
                                      args.num_workers_per_device)

    if args.exp_name is None:
        exp_name = args.env + '_' + args.algorithm
    else:
        exp_name = args.exp_name
    print('Commencing experiment', exp_name)

    run_experiments({
        exp_name: {
            "run": alg_run,
            "env": env_name,
            "stop": {
                "training_iteration": args.training_iterations
            },
            'checkpoint_freq': args.checkpoint_frequency,
            "config": config,
        }
    })


if __name__ == '__main__':
    main()

