""" DQN Agent implementation, inspired from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html """

import gym
import math
import random
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from algorithms.base import BaseTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters

# TODO these parameters should be arguments
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Replay Buffer

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer(object):
    """
    A buffer for storing trajectories experienced by an agent interacting with the environment.
    """

    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# DQN

class ValueNetwork(nn.Module):
    """
    Deep Q Network
    """

    def __init__(self, input_size, outputs, input_channels=3):
        super(ValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(12)

        def conv2d_size_out(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(input_size[0]))
        convh = conv2d_size_out(conv2d_size_out(input_size[1]))
        linear_input_size = convw * convh * 12

        self.head = nn.Linear(linear_input_size, outputs)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.head(x.view(x.size(0), -1))


class DQNAgent(BaseTrainer):
    """
    DQN Agent
    """

    def __init__(self, agent):
        self.n_actions = agent.action_space.n
        w, h, c = agent.observation_space.shape
        self.input_size = (w, h)
        self.input_channels = c

        self.policy_net = ValueNetwork(self.input_size, self.n_actions, self.input_channels).to(device)
        self.target_net = ValueNetwork(self.input_size, self.n_actions, self.input_channels).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayBuffer(10000)

        self.last_state = None
        self.last_action = None

        self.steps_done = 0

    def select_action(self, state):
        """

        """
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.float)

    def optimize(self):
        """

        """
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                      device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Update target net
        if self.steps_done % TARGET_UPDATE:
            self.target_net.load_state_dict(self.policy_net.state_dict())


    def step(self, obs, reward, done, info={}):
        """
        
        """

        state = preprocess_obs(obs)
        reward = torch.tensor([reward], device=device)

        if self.last_state:
            self.memory.push(self.last_state, self.last_action, state, reward)

        action = self.select_action(state)
        self.last_action = action

        return action.detach().item()

    

# Utils

def preprocess_obs(obs):
    """
    Transfom w x h x c rgb numpy array to 1 x c x h x c torch tensor
    """
    obs = obs.astype('float32') 
    obs = obs.transpose((2, 0, 1))
    obs = torch.from_numpy(obs)
    obs = obs.unsqueeze(0).to(device)
    return obs