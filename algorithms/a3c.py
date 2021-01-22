"""
Functions and classes for A3C algorithm.
Most of the code is taken from https://github.com/greydanus/baby-a3c.
"""

import os
import time
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt
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
    def __init__(self, channels, memsize, num_actions, vocab_size=0, n_agents=5):
        super(A3CPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.hidden = nn.Linear(128 + vocab_size * n_agents, memsize)
        self.critic_head = nn.Linear(memsize, 1)
        self.actor_head = nn.Linear(memsize, num_actions)

    # pylint: disable=arguments-differ
    def forward(self, inputs):
        visual, messages = inputs
        visual = F.relu(self.conv1(visual))
        visual = F.relu(self.conv2(visual))
        visual = F.relu(self.conv3(visual))
        full_input = torch.cat((visual.view(-1, 128), messages), 1)
        hidden = self.hidden(full_input)
        value = self.critic_head(hidden)
        logp = F.log_softmax(self.actor_head(hidden), dim=-1)
        return value, logp

    def try_load(self, save_dir, agent_name, logger=None, checkpoint=None):
        """
        Try to load saved models from save_dir
        """
        paths = glob.glob(os.path.join(save_dir, f'*.{agent_name}.*.tar'))
        step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            index = ckpts.index(checkpoint) if checkpoint is not None else np.argmax(ckpts)
            step = ckpts[index]
            try:
                self.load_state_dict(torch.load(paths[index]))
            except Exception as e:
                if logger is not None:
                    logger.error(e)
                step = 0
        if logger is not None:
            if step == 0:
                logger.info("\tno saved models")
            else:
                logger.info("\tloaded model: {}".format(paths[index]))
        return step

class EasySpeakerPolicy(nn.Module):
    """
    Small MLP for model.
    """
    def __init__(self, input=4, hidden=10, vocab_size=5):
        super(EasySpeakerPolicy, self).__init__()
        self.hidden_layer = nn.Linear(input, hidden)
        self.actor = nn.Linear(hidden, vocab_size)
        self.critic = nn.Linear(hidden, 1)

    # pylint: disable=arguments-differ
    def forward(self, inputs):
        hidden = F.relu(self.hidden_layer(inputs))
        logp = F.log_softmax(self.actor(hidden), dim=-1)
        value = self.critic(hidden)
        return value, logp

    def try_load(self, save_dir, agent_name, logger=None, checkpoint=None):
        """
        Try to load saved models from save_dir
        """
        paths = glob.glob(os.path.join(save_dir, f'*.{agent_name}.*.tar'))
        step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            index = ckpts.index(checkpoint) if checkpoint is not None else np.argmax(ckpts)
            step = ckpts[index]
            try:
                self.load_state_dict(torch.load(paths[index]))
            except Exception as e:
                if logger is not None:
                    logger.error(e)
                step = 0
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

class SharedRMSprop(torch.optim.RMSprop):
    """
    Extends a pytorch optimizer so it shares grads across processes
    """
    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-8, weight_decay=0):
        super(SharedRMSprop, self).__init__(params, lr, alpha, eps, weight_decay)
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                # pylint: disable=line-too-long
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['square_avg'] = param.data.new().resize_as_(param.data).zero_().share_memory_()

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


def cost_func(args, values, logps, actions, rewards, device):
    """
    Compute loss for A3C training
    """
    np_values = values.view(-1).data.cpu().numpy()

    # generalized advantage estimation using \delta_t residuals (a policy gradient method)
    delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]
    logpys = logps.gather(1, actions.clone().detach().view(-1, 1))
    gen_adv_est = discount(delta_t, args.gamma * args.tau)
    policy_loss = -(logpys.view(-1) *
                    torch.FloatTensor(gen_adv_est.copy()).to(device)).sum()

    # l2 loss over value estimator
    rewards[-1] += args.gamma * np_values[-1]
    discounted_r = discount(np.asarray(rewards), args.gamma)
    # pylint: disable=not-callable
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32).to(device)
    value_loss = .5 * (discounted_r - values[:-1, 0]).pow(2).sum()

    # entropy definition, for entropy regularization
    entropy_loss = (-logps * torch.exp(logps)).sum()
    return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss


def train(shared_models, shared_optimizers, shared_schedulers, rank, args, info):
    """
    A3C worker training function
    """
    logger = logging.getLogger('A3C')
    utility_funcs.setup_logger(logger, args)
    if args.cpu_only:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    logger.info(f"Process {rank} started")

    # make a local (unshared) environment
    env = args.env_maker(num_agents=args.agents, seed=args.seed + rank, view_size=args.view_size)
    torch.manual_seed(args.seed + rank)  # seed everything
    models = {
        'agent-0': A3CPolicy(channels=3,
                             memsize=args.hidden,
                             num_actions=args.num_actions,
                             vocab_size=args.vocab,
                             n_agents=args.agents).share_memory().to(device),
        'agent-1': EasySpeakerPolicy(input=args.vocab, vocab_size=args.vocab).to(device)
    }  # local/unshared models
    obs = env.reset()
    states = {
        agent_name: preprocess_obs(ob, device=device)
        for agent_name, ob in obs.items()
    }  # get first state

    start_time = last_disp_time = time.time()
    start_frames = last_disp_frame = int(info["start_frames"].item())
    episode_length, epr, eploss, last_nb_ep = 0, 0, 0, 0  # bookkeeping
    dones = {
        f'agent-{i}': True
        for i in range(args.agents)
    }

    # openai baselines uses 40M frames...we'll use 80M
    while info['frames'][0] <= args.horizon or args.test:
        for agent_name, model in models.items():
            # sync with shared model
            model.load_state_dict(shared_models[agent_name].state_dict())

        # save values for computing gradients
        values_hist = {
            'agent-0': [],
            'agent-1': []
        }
        logps_hist = {
            'agent-0': [],
            'agent-1': []
        }
        actions_hist = {
            'agent-0': [],
            'agent-1': []
        }
        rewards_hist = {
            'agent-0': [],
            'agent-1': []
        }

        for _ in range(args.rnn_steps):
            episode_length += 1
            actions = {}
            messages = {}
            get_comm(env, messages, args.noise)
            perfect_message = preprocess_messages(messages, args.vocab, device=device)

            # get speaker message
            comm_value, comm_logp = models['agent-1'](perfect_message)

            if args.test:
                message = torch.argmax(comm_logp).unsqueeze(0).data
                messages['agent-0'] = int(message.cpu().numpy()[0])
            else:
                message = torch.exp(comm_logp).multinomial(num_samples=1).data[0]
                messages['agent-0'] = message.cpu().numpy()[0]

            values_hist['agent-1'].append(comm_value)
            logps_hist['agent-1'].append(comm_logp)
            actions_hist['agent-1'].append(message)

            real_messages = preprocess_messages(messages, args.vocab, device=device)

            # select listener action
            value, logp = models['agent-0']((states['agent-0'], real_messages))

            if args.test:
                action = torch.argmax(logp).unsqueeze(0).data
                actions['agent-0'] = int(action.cpu().numpy()[0])
            else:
                action = torch.exp(logp).multinomial(num_samples=1).data[0]
                actions['agent-0'] = action.cpu().numpy()[0]

            values_hist['agent-0'].append(value)
            logps_hist['agent-0'].append(logp)
            actions_hist['agent-0'].append(action)

            obs, rewards, dones, _ = env.step(actions)

            if args.render:
                env.render()
                time.sleep(0.5)

            states = {
                agent_name: preprocess_obs(ob, device=device)
                for agent_name, ob in obs.items()
            }

            rewards['agent-1'] = rewards['agent-0']

            epr += np.sum([reward for reward in rewards.values()])

            for agent_name, reward in rewards.items():
                rewards[agent_name] = np.clip(reward, -1, 1)  # clip reward
                rewards_hist[agent_name].append(rewards[agent_name])

            info['frames'].add_(1)
            num_frames = int(info['frames'].item())

            if num_frames % 1e6 == 0:  # save every 1M frames
                logger.info(f"{num_frames/1e6:.2f}M frames: saving models as \
                               model.<agent_name>.{num_frames/1e6:.0f}.tar")
                for agent_name, shared_model in shared_models.items():
                    model_path = os.path.join(args.save_dir,
                                              f'model.{agent_name}.{num_frames/1e6:.0f}.tar')
                    torch.save(shared_model.state_dict(), model_path)

            if dones["__all__"]:  # update shared data
                info['episodes'] += 1
                info['run_epr'].add_(epr)

            # if rank == 0 and time.time() - last_disp_time > 60:  # print info ~ every minute
            if rank == 0 and num_frames - last_disp_frame > 10000:  # print info ~ every 10 000 frames
                last_disp_frame = num_frames
                elapsed = time.strftime(
                    "%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                logger.info(f"time {elapsed}, " +
                            f"episodes {info['episodes'].item():.0f}, " +
                            f"frames {num_frames/1e6:.2f}M, " +
                            f"throughput {(num_frames - start_frames)/(time.time()-start_time):.2f}f/s, " +
                            f"mean epr {info['run_epr'].item()/max(info['episodes'].item()-last_nb_ep, 1):.2f}, ")
                last_disp_time = time.time()
                last_nb_ep = info['episodes'].item()
                info['run_epr'].add_(-info['run_epr'].item())

            if dones["__all__"]:  # maybe print info.
                episode_length, epr, eploss = 0, 0, 0
                obs = env.reset()
                states = {
                    agent_name: preprocess_obs(ob, device=device)
                    for agent_name, ob in obs.items()
                }

        if dones["__all__"]:
            next_value = torch.zeros(1, 1).to(device)
            values_hist['agent-0'].append(next_value.detach())
            values_hist['agent-1'].append(next_value.detach())
        else:
            get_comm(env, messages, args.noise)
            perfect_message = preprocess_messages(messages, args.vocab, device=device)

            # get speaker message
            comm_value, comm_logp = models['agent-1'](perfect_message)

            message = torch.exp(comm_logp).multinomial(num_samples=1).data[0]
            messages['agent-0'] = message.cpu().numpy()[0]

            real_messages = preprocess_messages(messages, args.vocab, device=device)

            # select listener action
            next_value, _ = models['agent-0']((states['agent-0'], real_messages))

            values_hist['agent-1'].append(comm_value.detach())
            values_hist['agent-0'].append(next_value.detach())
        
        for agent_name, model in models.items():
            loss = cost_func(args,
                             torch.cat(values_hist[agent_name]),
                             torch.cat(logps_hist[agent_name]),
                             torch.cat(actions_hist[agent_name]),
                             np.asarray(rewards_hist[agent_name]),
                             device=device)
            eploss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(models[agent_name].parameters(), 40)

        for agent_name in models.keys():
            for param, shared_param in zip(models[agent_name].parameters(), shared_models[agent_name].parameters()):
                if shared_param.grad is None:
                    shared_param._grad = param.grad  # sync gradients with shared model
            shared_optimizers[agent_name].step()
            shared_optimizers[agent_name].zero_grad()
            shared_schedulers[agent_name].step()


# Utils

def preprocess_obs(obs, device):
    """
    Transfom w x h x c rgb numpy array to 1 x c x h x c torch tensor
    """
    obs = obs.astype('float32')
    obs = cv2.resize(obs, dsize=(10, 10))/255
    obs = obs.transpose((2, 0, 1))
    obs = torch.from_numpy(obs)
    obs = obs.unsqueeze(0)
    return obs.to(device)

def preprocess_messages(messages, vocab_size, device):
    """
    Transform a dict of int messages into a flattened one hot tensor
    """
    indices = np.array([message for message in messages.values()]).flatten()
    one_hot = np.zeros((indices.size, vocab_size)).astype('float32')
    one_hot[np.arange(indices.size), indices] = 1
    one_hot = one_hot.flatten()
    processed_messages = torch.from_numpy(one_hot)
    processed_messages = processed_messages.unsqueeze(0)
    return processed_messages.to(device)

def discount(x, gamma):
    """
    discount rewards with a gamma rate.
    """
    return lfilter([1], [1, -gamma], x[::-1])[::-1]

def get_comm(env, messages, noise=0.):
    if np.random.random() < noise:
        for agent_name, agent in env.agents.items():
            messages[agent_name] = np.random.randint(1, 5)
    else:
        i, j = np.where(env.world_map == 'A')
        for agent_name, agent in env.agents.items():
            if len(i) == 0:
                messages[agent_name] = 0
            elif i > agent.pos[0]:
                messages[agent_name] = 1
            elif i < agent.pos[0]:
                messages[agent_name] = 2
            elif j < agent.pos[1]:
                messages[agent_name] = 3
            elif j > agent.pos[1]:
                messages[agent_name] = 4
            else:
                messages[agent_name] = 0
