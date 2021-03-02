# -*- coding:utf-8 -*-
__author__ = 'Mingqi Yuan'
"""
Implementation of the Proximal Policy Optimization.
"""

from torch import optim
from torch.nn import functional as F
from torch.distributions import Categorical
import pandas as pd
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.dirname(__file__) + os.sep + '../')

from nn.ActorDiscrete import ActorDis
from nn.CriticStateOnly import CriticSO
from replayer.PPOReplayer import PPOReplayer

class PPO:
    def __init__(
            self,
            device,
            state_dim,
            action_dim,
            actor_kwargs,
            critic_kwargs,
            det=False,
            gamma=0.99,
            lambd=0.99,
            lr=1e-4,
            batch_size=64,
            batches=1
    ):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lambd = lambd
        self.lr = lr
        self.batch_size = batch_size
        self.batches = batches

        self.det = det
        self.actor = ActorDis(actor_kwargs)
        self.critic = CriticSO(critic_kwargs)
        self.actor.to(self.device)
        self.critic.to(self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.trajectory = []
        self.replayer = PPOReplayer()

    def learn(self, state, action, reward, next_state, done):
        self.trajectory.append([state, action, reward, done])

        if done:
            ''' reconstruct the data '''
            df = pd.DataFrame(
                np.array(self.trajectory, dtype=object).reshape(-1, 4),
                columns=['state', 'action', 'reward', 'done'])
            state_tensor = torch.FloatTensor(np.stack(df['state'])).to(self.device)
            action_tensor = torch.from_numpy(np.stack(df['action']).astype('int64')).to(self.device)
            v_tensor = self.critic(state_tensor)
            df['v'] = v_tensor.detach().cpu().numpy()
            prob_tensor = self.actor(state_tensor)
            pi_tensor = prob_tensor.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
            df['prob'] = pi_tensor.detach().cpu().numpy()
            df['next_v'] = df['v'].shift(-1).fillna(0.)
            df['u'] = df['reward'] + self.gamma * df['next_v']
            df['delta'] = df['u'] - df['v']
            df['return'] = df['reward']
            df['advantage'] = df['delta']
            for i in df.index[-2::-1]:
                df.loc[i, 'return'] += self.gamma * df.loc[i + 1, 'return']
                df.loc[i, 'advantage'] += self.gamma * self.lambd * df.loc[i + 1, 'advantage']

            self.replayer.store(df)

            for i in range(self.batches):
                states_, actions_, old_probs_, advantages_, returns_ = \
                    self.replayer.sample(size=self.batch_size)

                batch_state = torch.FloatTensor(states_).to(self.device)
                batch_action = torch.from_numpy(actions_).to(self.device)
                batch_old_prob = torch.FloatTensor(old_probs_).to(self.device)
                batch_advantage = torch.FloatTensor(advantages_).to(self.device)
                batch_return = torch.FloatTensor(returns_).to(self.device)

                ''' update actor '''
                pi_tensor = self.actor(batch_state).gather(
                    1, batch_action.unsqueeze(1)).squeeze(1)
                surrogate_advantage_tensor = (pi_tensor / batch_old_prob) * batch_advantage
                clip_times_advantage_tensor = 0.1 * surrogate_advantage_tensor
                max_surrogate_advantage_tensor = batch_advantage + torch.where(
                    batch_advantage > 0.,
                    clip_times_advantage_tensor,
                    -clip_times_advantage_tensor)

                clipped_surrogate_advantage_tensor = torch.min(
                    surrogate_advantage_tensor, max_surrogate_advantage_tensor)
                actor_loss_tensor = -clipped_surrogate_advantage_tensor.mean()
                self.optimizer_actor.zero_grad()
                actor_loss_tensor.backward()
                self.optimizer_actor.step()

                ''' update critic '''
                pred_vs = self.critic(batch_state)
                critic_loss_tensor = F.mse_loss(pred_vs[:, 0], batch_return)
                self.optimizer_critic.zero_grad()
                critic_loss_tensor.backward()
                self.optimizer_critic.step()

            ''' reset the replayer '''
            self.trajectory = []
            self.replayer = PPOReplayer()

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
        torch.save(self.actor, '{}/ppo_policy_epoch{}.pth'.format(save_dir, epoch))

