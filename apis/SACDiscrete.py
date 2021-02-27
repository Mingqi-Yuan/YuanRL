# -*- coding:utf-8 -*-
__author__ = 'Mingqi Yuan'
"""
Implementation of the Deep deterministic policy gradient algorithm.
"""

from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Categorical
import numpy as np
import torch
import copy
import sys
import os

sys.path.append('..')

from nn.ActorDiscrete import ActorDis
from nn.CriticStateOnly import CriticSO
from replayer.DQNReplayer import DQNReplayer

class SACDiscrete:
    def __init__(
            self,
            device,
            state_dim,
            action_dim,
            actor_kwargs,
            critic_kwargs,
            det=False,
            gamma=0.99,
            tau=0.005,
            replayer_capacity=10000,
            replayer_initial_transitions=5000,
            lr=1e-4,
            batch_size=64,
            batches=1
    ):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.batch_size = batch_size
        self.batches = batches

        self.replayer = DQNReplayer(replayer_capacity)
        self.replayer_initial_transitions = replayer_initial_transitions

        # self.entropy_target = 0.98 * (-np.log(1 / self.action_dim))
        # self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        # self.alpha = self.log_alpha.exp()
        self.alpha = 0.02

        self.det = det
        self.actor = ActorDis(actor_kwargs)
        self.qf1_eval = CriticSO(critic_kwargs)
        self.qf1_target = CriticSO(critic_kwargs)
        self.qf2_eval = CriticSO(critic_kwargs)
        self.qf2_target = CriticSO(critic_kwargs)

        self.actor.to(self.device)
        self.qf1_eval.to(self.device)
        self.qf1_target.to(self.device)
        self.qf2_eval.to(self.device)
        self.qf2_target.to(self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optimizer_qf1 = optim.Adam(self.qf1_eval.parameters(), lr=self.lr)
        self.optimizer_qf2 = optim.Adam(self.qf2_eval.parameters(), lr=self.lr)
        # self.optimizer_alpha = optim.Adam([self.log_alpha], lr=self.lr)

        ''' load weights '''
        self.qf1_target.load_state_dict(self.qf1_eval.state_dict())
        self.qf2_target.load_state_dict(self.qf2_eval.state_dict())

    def update_target_net(self, target_net, eval_net):
        for target_params, params in zip(target_net.parameters(), eval_net.parameters()):
            target_params.data.copy_(
                target_params.data * (1.0 - self.tau) + params.data * self.tau
            )

    def learn(self, state, action, reward, next_state, done):
        self.replayer.store(state, action, reward, next_state, done)

        if self.replayer.count >= self.replayer_initial_transitions:
            for b in range(self.batches):
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = \
                    self.replayer.sample(self.batch_size)

                batch_state = torch.FloatTensor(batch_state).to(self.device)
                batch_action = torch.from_numpy(batch_action).to(self.device)
                batch_reward = torch.FloatTensor(batch_reward).to(self.device)
                batch_next_state = torch.FloatTensor(batch_next_state).to(self.device)
                batch_done = torch.FloatTensor(batch_done).to(self.device)

                ''' update action value function '''
                with torch.no_grad():
                    probs_next = self.actor(batch_next_state).clamp(1e-4, 0.999)
                    qs1_next = self.qf1_target(batch_next_state)
                    qs2_next = self.qf2_target(batch_next_state)
                    qs_next = torch.min(qs1_next, qs2_next)
                    vs_next = probs_next * (qs_next - self.alpha * torch.log(probs_next))
                    vs_next = torch.sum(vs_next, dim=1)
                    qs_target = batch_reward + self.gamma * torch.logical_not(batch_done) * vs_next

                self.optimizer_qf1.zero_grad()
                self.optimizer_qf2.zero_grad()
                qs1 = self.qf1_eval(batch_state).gather(dim=1, index=batch_action.unsqueeze(1))
                qs2 = self.qf2_eval(batch_state).gather(dim=1, index=batch_action.unsqueeze(1))
                qf1_loss = F.mse_loss(qs1[:, 0], qs_target)
                qf2_loss = F.mse_loss(qs2[:, 0], qs_target)
                qf1_loss.backward()
                qf2_loss.backward()
                self.optimizer_qf1.step()
                self.optimizer_qf2.step()

                ''' update actor '''
                probs = self.actor(batch_state).clamp(1e-4, 0.999)
                with torch.no_grad():
                    qs1 = self.qf1_eval(batch_state)
                    qs2 = self.qf2_eval(batch_state)
                    qs = torch.min(qs1, qs2)

                self.optimizer_actor.zero_grad()
                actor_loss = self.alpha * probs * torch.log(probs) - probs * qs
                actor_loss = actor_loss.sum(dim=1).mean()
                actor_loss.backward()
                self.optimizer_actor.step()

                # ''' update temperature '''
                # self.optimizer_alpha.zero_grad()
                # log_probs = (probs * torch.log(probs)).sum(-1)
                # alpha_loss = - (self.log_alpha * (log_probs.detach() + self.entropy_target)).mean()
                # alpha_loss.backward()
                # self.optimizer_alpha.step()
                # self.alpha = self.log_alpha.exp()

                ''' update target Q '''
                self.update_target_net(self.qf1_target, self.qf1_eval)
                self.update_target_net(self.qf2_target, self.qf2_eval)

    def decide(self, state):
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float()
            state_tensor = torch.unsqueeze(state_tensor, dim=0)

            prob = self.actor(state_tensor.to(self.device)).clamp(1e-4, 0.999)
            ''' deterministic policy '''
            if self.det:
                action = torch.argmax(prob, dim=1)
            else:
                policy = Categorical(prob)
                action = policy.sample()

        return action.detach().cpu().numpy()[0]

    def save(self, save_dir, epoch):
        torch.save(self.actor, '{}/sacd_policy_epoch{}.pth'.format(save_dir, epoch))