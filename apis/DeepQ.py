# -*- coding:utf-8 -*-
__author__ = 'Mingqi Yuan'
"""
Implementation of the Deep Q-learning algorithm.
"""

from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.dirname(__file__) + os.sep + '../')

from nn.QNetworkDiscrete import QND
from replayer.DQNReplayer import DQNReplayer

class DeepQ:
    def __init__(
            self,
            device,
            state_dim,
            action_dim,
            qnet_kwargs,
            gamma=0.99, 
            epsilon=0.001,
            replayer_capacity=10000,
            replayer_initial_transitions=5000,
            batch_size=64,
            lr=1e-3
    ):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon

        self.batch_size = batch_size
        self.lr = lr
        self.replayer = DQNReplayer(replayer_capacity)
        self.replayer_initial_transitions = replayer_initial_transitions

        self.eval_net = QND(qnet_kwargs)
        self.target_net = QND(qnet_kwargs)
        self.eval_net.to(self.device)
        self.target_net.to(self.device)

        self.optimizer_eval_net = optim.Adam(self.eval_net.parameters(), lr=self.lr)

        self.target_net.load_state_dict(self.eval_net.state_dict())

    def learn(self, state, action, reward, next_state, done):
        self.replayer.store(state, action, reward, next_state, done)

        if self.replayer.count >= self.replayer_initial_transitions:
            states_, actions_, rewards_, next_states_, dones_ = \
                self.replayer.sample(self.batch_size)

            batch_state = torch.FloatTensor(states_).to(self.device)
            batch_action = torch.from_numpy(actions_).to(self.device)
            batch_reward = torch.FloatTensor(rewards_).to(self.device)
            batch_next_state = torch.FloatTensor(next_states_).to(self.device)
            batch_done = torch.from_numpy(dones_).to(self.device)

            next_qs = self.target_net(batch_next_state)
            next_max_qs, _ = torch.max(next_qs, dim=1)
            q_targets = batch_reward + self.gamma * torch.logical_not(batch_done) * next_max_qs
            qs = self.eval_net(batch_state)
            qs = torch.gather(qs, 1, batch_action.unsqueeze(1)).squeeze(1)

            ''' update network '''
            self.optimizer_eval_net.zero_grad()
            eval_net_loss = F.mse_loss(qs, q_targets)
            eval_net_loss.backward()
            self.optimizer_eval_net.step()

            if done:
                self.target_net.load_state_dict(self.eval_net.state_dict())

    def decide(self, state):
        # epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_tensor = torch.FloatTensor(state).to(self.device)
        state_tensor = torch.unsqueeze(state_tensor, 0)
        qs = self.eval_net(state_tensor)
        action = torch.argmax(qs, dim=1)
        return action.detach().cpu().numpy()[0]

    def save(self, save_dir, epoch):
        torch.save(self.eval_net, '{}/ql_policy_epoch{}.pth'.format(save_dir, epoch))
