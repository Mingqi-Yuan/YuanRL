# -*- coding:utf-8 -*-
__author__ = 'Mingqi Yuan'
"""
Implementation of the Trust Region Policy Optimization.
"""

from torch import optim, autograd
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

class TRPO:
    def __init__(
            self,
            device,
            state_dim,
            action_dim,
            actor_kwargs,
            critic_kwargs,
            det=False,
            gamma=0.99,
            max_kl=0.01,
            lr=1e-4,
            batch_size=64,
            batches=1,
            max_grad_norm=0.5
    ):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.max_kl = max_kl
        self.lr = lr
        self.batch_size = batch_size
        self.batches = batches
        self.max_grad_norm = max_grad_norm

        self.det = det
        self.actor = ActorDis(actor_kwargs)
        self.critic = CriticSO(critic_kwargs)
        self.actor.to(self.device)
        self.critic.to(self.device)

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
                df.loc[i, 'advantage'] += self.gamma * df.loc[i + 1, 'advantage']

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
                ''' get first order gradient: g '''
                pi_tensor = self.actor(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
                surrogate_advantage_tensor = (pi_tensor / batch_old_prob) * batch_advantage
                actor_loss_tensor = surrogate_advantage_tensor.mean()
                actor_loss_grads = autograd.grad(actor_loss_tensor, self.actor.parameters())
                actor_loss_grads = torch.cat([grad.view(-1) for grad in actor_loss_grads]).detach()

                ''' get the conjugate gradient: Fx=g '''
                def f(x):
                    prob_tensor = self.actor(batch_state)
                    prob_old_tensor = prob_tensor.detach()
                    kld_tensor = (prob_old_tensor * torch.log(
                        (prob_old_tensor / prob_tensor).clamp(1e-6, 1e6))).sum(axis=1)
                    kld_loss_tensor = kld_tensor.mean()
                    grads = autograd.grad(kld_loss_tensor, self.actor.parameters(), create_graph=True)
                    flatten_grad_tensor = torch.cat([grad.view(-1) for grad in grads])
                    grad_matmul_x = torch.dot(flatten_grad_tensor, x)
                    grad_grads = autograd.grad(grad_matmul_x, self.actor.parameters())
                    flatten_grad_grad = torch.cat([grad.contiguous().view(-1) for grad in grad_grads]).detach()
                    fx = flatten_grad_grad + x * 0.01

                    return fx

                x, fx = self.conjugate_gradient(f, actor_loss_grads)
                # ... calculate natural gradient: sqrt(...) g
                natural_gradient_tensor = torch.sqrt(2 * self.max_kl / torch.dot(fx, x)) * x

                # ... line search
                def set_actor_net_params(flatten_params):
                    begin = 0
                    for param in self.actor.parameters():
                        end = begin + param.numel()
                        param.data.copy_(flatten_params[begin:end].view(param.size()))
                        begin = end

                old_params = torch.cat([param.view(-1) for param in self.actor.parameters()])
                expected_improve = torch.dot(actor_loss_grads, natural_gradient_tensor)
                for learning_step in [0., ] + [.5 ** j for j in range(10)]:
                    new_params = old_params + learning_step * natural_gradient_tensor
                    set_actor_net_params(new_params)
                    new_pi_tensor = self.actor(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
                    new_pi_tensor = new_pi_tensor.detach()
                    surrogate_tensor = (new_pi_tensor / pi_tensor) * batch_advantage
                    objective = surrogate_tensor.mean().item()
                    if np.isclose(learning_step, 0.):
                        old_objective = objective
                    else:
                        # Armijo condition
                        if objective - old_objective > 0.1 * expected_improve * learning_step:
                            break # improved, save the weight
                        else:
                            set_actor_net_params(old_params)

                ''' update critic '''
                pred_vs = self.critic(batch_state)
                critic_loss_tensor = F.mse_loss(pred_vs[:, 0], batch_return)
                self.optimizer_critic.zero_grad()
                critic_loss_tensor.backward()
                torch.nn.utils.clip_grad_norm(self.critic.parameters(), self.max_grad_norm)
                self.optimizer_critic.step()

            ''' reset the replayer '''
            self.trajectory = []
            self.replayer = PPOReplayer()

    def conjugate_gradient(self, f, b, iter_count=10, epsilon=1e-12, tol=1e-6):
        x = b * 0.
        r = b.clone()
        p = b.clone()
        rho = torch.dot(r, r)
        for i in range(iter_count):
            z = f(p)
            alpha = rho / (torch.dot(p, z) + epsilon)
            x += alpha * p
            r -= alpha * z
            rho_new = torch.dot(r, r)
            p = r + (rho_new / rho) * p
            rho = rho_new
            if rho < tol:
                break
        return x, f(x)

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
        torch.save(self.actor, '{}/trpo_policy_epoch{}.pth'.format(save_dir, epoch))