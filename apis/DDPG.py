# -*- coding:utf-8 -*-
__author__ = 'Mingqi Yuan'
"""
Implementation of the Deep deterministic policy gradient algorithm.
"""

from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import torch
import sys
import os

sys.path.append('..')

from nn.ActorContinuous import ActorCont
from nn.CriticStateAction import CriticSA
from replayer.DQNReplayer import DQNReplayer
from noise.OrnsteinUhlenbeckProcess import OUProcess

class DDPG:
    def __init__(
            self,
            device,
            state_dim,
            action_dim,
            action_low,
            action_high,
            actor_kwargs,
            critic_kwargs,
            replayer_capacity=20000,
            replayer_initial_transitions=2000,
            gamma=0.99,
            tau=0.005,
            noise_scale=0.1,
            explore=True,
            batches=1,
            batch_size=64,
            lr=1e-3
    ):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.gamma = gamma
        self.tau = tau
        self.noise_scale = noise_scale
        self.noise = OUProcess(size=(action_dim,))
        self.noise.reset()
        self.explore = explore
        self.batches = batches
        self.batch_size = batch_size
        self.lr = lr
        self.replayer_initial_transitions = replayer_initial_transitions
        self.replayer = DQNReplayer(replayer_capacity)

        self.actor_eval = ActorCont(actor_kwargs)
        self.actor_target = ActorCont(actor_kwargs)
        self.critic_eval = CriticSA(critic_kwargs)
        self.critic_target = CriticSA(critic_kwargs)
        self.actor_eval.to(self.device)
        self.actor_target.to(self.device)
        self.critic_eval.to(self.device)
        self.critic_target.to(self.device)

        self.optimizer_actor_eval = optim.Adam(self.actor_eval.parameters(), lr=self.lr)
        self.optimizer_critic_eval = optim.Adam(self.critic_eval.parameters(), lr=self.lr)

        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.critic_target.load_state_dict(self.critic_eval.state_dict())

    def update_target_net(self, target_net, eval_net):
        for target_params, params in zip(target_net.parameters(), eval_net.parameters()):
            target_params.data.copy_(
                target_params.data * (1.0 - self.tau) + params.data * self.tau
            )

    def decide(self, state):
        if self.explore and self.replayer.count < \
                self.replayer_initial_transitions:
            return np.random.uniform(self.action_low, self.action_high)

        state_tensor = torch.FloatTensor(state).to(self.device)
        state_tensor = torch.unsqueeze(state_tensor, dim=0)
        action = self.actor_eval(state_tensor).detach().cpu().numpy()[0]

        if self.explore:
            noise = self.noise()
            action = np.clip(action + noise, self.action_low, self.action_high)[0]
        return action

    def learn(self, state, action, reward, next_state, done):
        self.replayer.store(state, action, reward, next_state, done)

        if self.replayer.count >= self.replayer_initial_transitions:
            if done:
                self.noise.reset()
            for batch in range(self.batches):
                states_, actions_, rewards_, next_states_, dones_ = \
                    self.replayer.sample(self.batch_size)

                batch_state = torch.FloatTensor(states_).to(self.device)
                batch_action = torch.FloatTensor(actions_).to(self.device)
                batch_reward = torch.FloatTensor(rewards_).to(self.device)
                batch_next_state = torch.FloatTensor(next_states_).to(self.device)
                batch_done = torch.from_numpy(dones_).to(self.device)

                ''' update critic '''
                batch_next_action = self.actor_target(batch_next_state)
                noise_tensor = (0.2 * torch.randn_like(batch_action.unsqueeze(1), dtype=torch.float))
                batch_next_action = (batch_next_action + noise_tensor).clamp(self.action_low, self.action_high)
                next_qs = self.critic_target(batch_next_state, batch_next_action)
                qs = self.critic_eval(batch_state, batch_action.unsqueeze(1)).squeeze(1)
                q_targets = batch_reward + self.gamma * torch.logical_not(batch_done) * next_qs[:, 0]

                self.optimizer_critic_eval.zero_grad()
                critic_loss = F.mse_loss(qs, q_targets)
                critic_loss.backward()
                self.optimizer_critic_eval.step()

                ''' update actor '''
                action_tensor = self.actor_eval(batch_state)
                action_tensor = action_tensor.clamp(self.action_low, self.action_high)
                q_tensor = self.critic_eval(batch_state, action_tensor)

                self.optimizer_actor_eval.zero_grad()
                actor_loss = - q_tensor.mean()
                actor_loss.backward()
                self.optimizer_actor_eval.step()

                ''' update target networks '''
                self.update_target_net(self.actor_target, self.actor_eval)
                self.update_target_net(self.critic_target, self.critic_eval)


    def save(self, save_dir, epoch):
        torch.save(self.actor_eval, '{}/ddpg_policy_epoch{}.pth'.format(save_dir, epoch))
