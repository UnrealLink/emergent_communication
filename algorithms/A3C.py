import os
import sys
import gym
import time
import glob
import argparse
import logging

import numpy as np
from scipy.signal import lfilter
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

os.environ['OMP_NUM_THREADS'] = '1'


class A3CPolicy(nn.Module): # an actor-critic neural network
    def __init__(self, channels, memsize, num_actions):
        super(A3CPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gru = nn.GRUCell(32 * 5 * 5, memsize)
        self.critic_linear, self.actor_linear = nn.Linear(memsize, 1), nn.Linear(memsize, num_actions)

    def forward(self, inputs, train=True, hard=False):
        inputs, hx = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        hx = self.gru(x.view(-1, 32 * 5 * 5), (hx))
        return self.critic_linear(hx), self.actor_linear(hx), hx

    def try_load(self, save_dir, agent_name):
        paths = glob.glob(os.path.join(save_dir, f'*.{agent_name}.*.tar'))
        step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step == 0 else print("\tloaded model: {}".format(paths[ix]))
        return step


class SharedAdam(torch.optim.Adam): # extend a pytorch optimizer so it shares grads across processes
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1 # a "step += 1"  comes later
            super.step(closure)


def cost_func(args, values, logps, actions, rewards):
    np_values = values.view(-1).data.numpy()

    # generalized advantage estimation using \delta_t residuals (a policy gradient method)
    delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]
    logpys = logps.gather(1, actions.clone().detach().view(-1,1))
    gen_adv_est = discount(delta_t, args.gamma * args.tau)
    policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()
    
    # l2 loss over value estimator
    rewards[-1] += args.gamma * np_values[-1]
    discounted_r = discount(np.asarray(rewards), args.gamma)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = .5 * (discounted_r - values[:-1,0]).pow(2).sum()

    entropy_loss = (-logps * torch.exp(logps)).sum() # entropy definition, for entropy regularization
    return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss


def train(shared_models, shared_optimizers, rank, args, info):
    logging.basicConfig(filename=os.path.join(args.save_dir, 'log.txt'), 
                        level=logging.DEBUG)
    
    env = args.env_maker(num_agents=args.agents) # make a local (unshared) environment
    env.seed(args.seed + rank) ; torch.manual_seed(args.seed + rank) # seed everything
    models = {
        f'agent-{i}': A3CPolicy(channels=3, 
                               memsize=args.hidden, 
                               num_actions=args.num_actions)
        for i in range(args.agents)
    } # local/unshared models
    obs = env.reset()
    states = {
        agent_name: preprocess_obs(ob)
        for agent_name, ob in obs.items()
    } # get first state
    hxs = {
        f'agent-{i}': None
        for i in range(args.agents)
    }

    start_time = last_disp_time = time.time()
    episode_length, epr, eploss = 0, 0, 0 # bookkeeping
    dones = {
        f'agent-{i}': True
        for i in range(args.agents)
    }

    while info['frames'][0] <= 8e7 or args.test: # openai baselines uses 40M frames...we'll use 80M
        for agent_name, model in models.items():
            model.load_state_dict(shared_models[agent_name].state_dict()) # sync with shared model

        hxs = {
            agent_name: torch.zeros(1, 256) if dones[agent_name] else hx.detach()  # rnn activation vector
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

        for step in range(args.rnn_steps):
            episode_length += 1
            actions = {}
            for agent_name, model in models.items():
                value, logit, hx = model((states[agent_name], hxs[agent_name]))
                logp = F.log_softmax(logit, dim=-1)
                action = torch.exp(logp).multinomial(num_samples=1).data[0] #logp.max(1)[1].data if args.test else
                actions[agent_name] = action.numpy()[0]

                values_hist[agent_name].append(value)
                logps_hist[agent_name].append(logp)
                actions_hist[agent_name].append(action)
            
            obs, rewards, dones, _ = env.step(actions)
            
            if args.render: env.render()

            states = {
                agent_name: preprocess_obs(ob)
                for agent_name, ob in obs.items()
            }

            # epr += rewards['agent-0']
            epr += np.sum([reward for reward in rewards.values()])

            for agent_name, reward in rewards.items():
                rewards[agent_name] = np.clip(reward, -1, 1) # clip reward
                rewards_hist[agent_name].append(rewards[agent_name])
            
            # should set done to true once a certain number of timesteps is reached since
            # one episode shouldn't be played for too long
            
            info['frames'].add_(1) ; num_frames = int(info['frames'].item())

            if num_frames % 2e5 == 0: # save every 2M frames
                logging.info(f"{num_frames/1e6:.2f}M frames: saving models as \
                               model.<agent_name>.{num_frames/1e5:.0f}.tar")
                for agent_name, shared_model in shared_models.items():
                    torch.save(shared_model.state_dict(),
                               os.path.join(args.save_dir, f'model.{agent_name}.{num_frames/1e5:.0f}.tar'))

            if dones["__all__"]: # update shared data
                info['episodes'] += 1
                interp = 1 if info['episodes'][0] == 1 else 1 - args.horizon
                info['run_epr'].mul_(1-interp).add_(interp * epr)
                info['run_loss'].mul_(1-interp).add_(interp * eploss)

            if rank == 0 and time.time() - last_disp_time > 60: # print info ~ every minute
                elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                logging.info(f"time {elapsed}, episodes {info['episodes'].item():.0f}, \
                               frames {num_frames/1e6:.2f}M, mean epr {info['run_epr'].item():.2f}, \
                               run loss {info['run_loss'].item():.2f}")
                last_disp_time = time.time()

            if dones["__all__"]: # maybe print info.
                episode_length, epr, eploss = 0, 0, 0
                obs = env.reset()
                states = {
                    agent_name: preprocess_obs(ob)
                    for agent_name, ob in obs.items()
                }

        for agent_name, model in models.items():
            if dones[agent_name]:
                next_value = torch.zeros(1,1)
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
                if shared_param.grad is None: shared_param._grad = param.grad # sync gradients with shared model
            shared_optimizers[agent_name].step()


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

discount = lambda x, gamma: lfilter([1],[1,-gamma],x[::-1])[::-1] # discounted rewards one liner
