"""
Functions and classes for A3C algorithm.
Most of the code is taken from https://github.com/greydanus/baby-a3c.
"""

import os
import time
import glob
import logging

import numpy as np
from scipy.signal import lfilter
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.utility_funcs as utility_funcs

os.environ['OMP_NUM_THREADS'] = '1'


class A3CPolicy(nn.Module):
    """
    Actor-critic model for A3C algorithm.
    """
    def __init__(self, channels, memsize, num_actions,
                 communication=False, message_size=0, n_agents=5):
        super(A3CPolicy, self).__init__()
        self.communication = communication
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gru = nn.GRUCell(32 * 5 * 5 + message_size * n_agents, memsize)
        self.critic_head = nn.Linear(memsize, 1)
        self.actor_head = nn.Linear(memsize, num_actions)
        if self.communication:
            self.comm_critic_head = nn.Linear(memsize, 1)
            self.comm_actor_head = nn.Linear(memsize, message_size)

    # pylint: disable=arguments-differ
    def forward(self, inputs):
        if self.communication:
            visual, state, messages = inputs
        else:
            visual, state = inputs
        visual = F.elu(self.conv1(visual))
        visual = F.elu(self.conv2(visual))
        visual = F.elu(self.conv3(visual))
        visual = F.elu(self.conv4(visual))
        if self.communication:
            full_input = torch.cat(visual.view(-1, 32 * 5 * 5), messages)
        else:
            full_input = visual.view(-1, 32 * 5 * 5)
        state = self.gru(full_input.view(-1, 32 * 5 * 5), (state))
        return self.critic_head(state), self.actor_head(state), state

    def try_load(self, save_dir, agent_name, logger=None):
        """
        Try to load saved models from save_dir
        """
        paths = glob.glob(os.path.join(save_dir, f'*.{agent_name}.*.tar'))
        step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            index = np.argmax(ckpts)
            step = ckpts[index]
            self.load_state_dict(torch.load(paths[index]))
        if logger is not None:
            if step == 0:
                logger.info("\tno saved models")
            else:
                logger.info("\tloaded model: {}".format(paths[index]))
        return step


class SharedAdam(torch.optim.Adam):
    """
    Extends a pytorch optimizer so it shares grads across processes
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                # pylint: disable=line-too-long
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = param.data.new().resize_as_(param.data).zero_().share_memory_()
                state['exp_avg_sq'] = param.data.new().resize_as_(param.data).zero_().share_memory_()

        # pylint: disable=unused-variable
        def step(self, closure=None):
            for group in self.param_groups:
                for param in group['params']:
                    if param.grad is None:
                        continue
                    self.state[param]['shared_steps'] += 1
                    # a "step += 1"  comes later
                    self.state[param]['step'] = self.state[param]['shared_steps'][0] - 1
            super.step(closure)


def cost_func(args, values, logps, actions, rewards):
    """
    Compute loss for A3C training
    """
    np_values = values.view(-1).data.numpy()

    # generalized advantage estimation using \delta_t residuals (a policy gradient method)
    delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]
    logpys = logps.gather(1, actions.clone().detach().view(-1, 1))
    gen_adv_est = discount(delta_t, args.gamma * args.tau)
    policy_loss = -(logpys.view(-1) *
                    torch.FloatTensor(gen_adv_est.copy())).sum()

    # l2 loss over value estimator
    rewards[-1] += args.gamma * np_values[-1]
    discounted_r = discount(np.asarray(rewards), args.gamma)
    # pylint: disable=not-callable
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = .5 * (discounted_r - values[:-1, 0]).pow(2).sum()

    # entropy definition, for entropy regularization
    entropy_loss = (-logps * torch.exp(logps)).sum()
    return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss


def train(shared_models, shared_optimizers, shared_schedulers, rank, args, info):
    """
    A3C worker training function
    """
    logger = logging.getLogger('A3C' + args.env)
    utility_funcs.setup_logger(logger, args)

    logger.info(f"Process {rank} started")
    # make a local (unshared) environment
    env = args.env_maker(num_agents=args.agents)
    env.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)  # seed everything
    models = {
        f'agent-{i}': A3CPolicy(channels=3,
                                memsize=args.hidden,
                                num_actions=args.num_actions)
        for i in range(args.agents)
    }  # local/unshared models
    obs = env.reset()
    states = {
        agent_name: preprocess_obs(ob)
        for agent_name, ob in obs.items()
    }  # get first state
    hxs = {
        f'agent-{i}': None
        for i in range(args.agents)
    }

    start_time = last_disp_time = time.time()
    episode_length, epr, eploss = 0, 0, 0  # bookkeeping
    dones = {
        f'agent-{i}': True
        for i in range(args.agents)
    }

    # openai baselines uses 40M frames...we'll use 80M
    while info['frames'][0] <= 8e7 or args.test:
        for agent_name, model in models.items():
            # sync with shared model
            model.load_state_dict(shared_models[agent_name].state_dict())

        hxs = {
            # rnn activation vector
            agent_name: torch.zeros(
                1, 256) if dones[agent_name] else hx.detach()
            for agent_name, hx in hxs.items()
        }

        # save values for computing gradients
        values_hist = {
            f'agent-{i}': []
            for i in range(args.agents)
        }
        logps_hist = {
            f'agent-{i}': []
            for i in range(args.agents)
        }
        actions_hist = {
            f'agent-{i}': []
            for i in range(args.agents)
        }
        rewards_hist = {
            f'agent-{i}': []
            for i in range(args.agents)
        }

        for _ in range(args.rnn_steps):
            episode_length += 1
            actions = {}
            for agent_name, model in models.items():
                value, logit, hx = model((states[agent_name], hxs[agent_name]))
                logp = F.log_softmax(logit, dim=-1)
                # logp.max(1)[1].data if args.test else
                action = torch.exp(logp).multinomial(num_samples=1).data[0]
                actions[agent_name] = action.numpy()[0]

                values_hist[agent_name].append(value)
                logps_hist[agent_name].append(logp)
                actions_hist[agent_name].append(action)

            obs, rewards, dones, _ = env.step(actions)

            if args.render:
                env.render(time=1000)

            states = {
                agent_name: preprocess_obs(ob)
                for agent_name, ob in obs.items()
            }

            # epr += rewards['agent-0']
            epr += np.sum([reward for reward in rewards.values()])

            for agent_name, reward in rewards.items():
                rewards[agent_name] = np.clip(reward, -1, 1)  # clip reward
                rewards_hist[agent_name].append(rewards[agent_name])

            # should set done to true once a certain number of timesteps is reached since
            # one episode shouldn't be played for too long

            info['frames'].add_(1)
            num_frames = int(info['frames'].item())

            if num_frames % 2e5 == 0:  # save every 0.2M frames
                logger.info(f"{num_frames/1e6:.2f}M frames: saving models as \
                               model.<agent_name>.{num_frames/1e5:.0f}.tar")
                for agent_name, shared_model in shared_models.items():
                    model_path = os.path.join(args.save_dir,
                                              f'model.{agent_name}.{num_frames/1e5:.0f}.tar')
                    torch.save(shared_model.state_dict(), model_path)

            if dones["__all__"]:  # update shared data
                info['episodes'] += 1
                interp = 1 if info['episodes'][0] == 1 else args.horizon
                info['run_epr'].mul_(1-interp).add_(interp * epr)
                info['run_loss'].mul_(1-interp).add_(interp * eploss)

            if rank == 0 and time.time() - last_disp_time > 60:  # print info ~ every minute
                elapsed = time.strftime(
                    "%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                logger.info(f"time {elapsed}, " +
                            f"episodes {info['episodes'].item():.0f}, " +
                            f"frames {num_frames/1e6:.2f}M, " +
                            f"mean epr {info['run_epr'].item():.2f}, " +
                            f"run loss {info['run_loss'].item():.2f}")
                last_disp_time = time.time()

            if dones["__all__"]:  # maybe print info.
                episode_length, epr, eploss = 0, 0, 0
                obs = env.reset()
                states = {
                    agent_name: preprocess_obs(ob)
                    for agent_name, ob in obs.items()
                }

        for agent_name, model in models.items():
            if dones[agent_name]:
                next_value = torch.zeros(1, 1)
            else:
                next_value = model((states[agent_name], hxs[agent_name]))[0]

            values_hist[agent_name].append(next_value.detach())

            loss = cost_func(args,
                             torch.cat(values_hist[agent_name]),
                             torch.cat(logps_hist[agent_name]),
                             torch.cat(actions_hist[agent_name]),
                             np.asarray(rewards_hist[agent_name]))
            eploss += loss.item()
            shared_optimizers[agent_name].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(models[agent_name].parameters(), 40)

            for param, shared_param in zip(models[agent_name].parameters(), shared_models[agent_name].parameters()):
                if shared_param.grad is None:
                    shared_param._grad = param.grad  # sync gradients with shared model
            shared_optimizers[agent_name].step()
            if dones["__all__"]:
                shared_schedulers[agent_name].step()


# Utils

def preprocess_obs(obs):
    """
    Transfom w x h x c rgb numpy array to 1 x c x h x c torch tensor
    """
    obs = obs.astype('float32')
    obs = cv2.resize(obs, dsize=(80, 80))
    obs = obs.transpose((2, 0, 1))
    obs = torch.from_numpy(obs)
    obs = obs.unsqueeze(0)
    return obs


def discount(x, gamma):
    """
    discount rewards with a gamma rate.
    """
    return lfilter([1], [1, -gamma], x[::-1])[::-1]
