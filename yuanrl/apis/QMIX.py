# -*- coding:utf-8 -*-
__author__ = 'Mingqi Yuan'
"""
Implementation of the QMIX algorithm.
"""

from torch import optim
from torch.nn import functional as F
import itertools
import torch
import sys
import os

sys.path.append(os.path.dirname(__file__) + os.sep + '../')

from yuanrl.nn.QMIXBackbone import MultiAgent, MixingNet
from yuanrl.replayer.QMIXReplayer import QMIXReplayer

'''
gs: global state
ls: local state
ga: global action
la: local action
h: hidden information
'''

class QMIX:
    def __init__(
            self,
            device,
            gs_dim,
            ga_dim,
            ls_dim,
            la_dim,
            agents_num,
            agent_kwargs,
            mixing_kwargs,
            gamma=0.99,
            epsilon=0.01,
            replayer_capacity=10000,
            replayer_initial_transitions=5000,
            batches=1,
            batch_size=64,
            lr=1e-3
    ):
        self.device = device
        self.gs_dim = gs_dim
        self.ga_dim = ga_dim
        self.ls_dim = ls_dim
        self.la_dim = la_dim
        self.agents_num = agents_num

        self.replayer = QMIXReplayer(replayer_capacity)
        self.replayer_initial_transitions = replayer_initial_transitions

        self.gamma = gamma
        self.epsilon = epsilon
        self.batches = batches
        self.batch_size = batch_size
        self.lr = lr

        self.eval_agents = MultiAgent(device=self.device, agents_num=self.agents_num, agent_kwargs=agent_kwargs)
        self.target_agents = MultiAgent(device=self.device, agents_num=self.agents_num, agent_kwargs=agent_kwargs)
        ''' load weights '''
        self.target_agents.load_state_dict(self.eval_agents.state_dict())
        self.eval_agents.to(self.device)
        self.target_agents.to(self.device)

        self.mixingnet = MixingNet(kwargs=mixing_kwargs)
        self.mixingnet.to(self.device)

        self.optimizer = optim.Adam(
            itertools.chain(self.mixingnet.parameters(), self.eval_agents.parameters()),
            lr=self.lr)

    def learn(self, gs, ls, ph, la, reward, next_gs, next_ls, h, done):
        self.replayer.store(gs, ls, ph, la, reward, next_gs, next_ls, h, done)

        if self.replayer.count >= self.replayer_initial_transitions:
            gs_, ls_, ph_, la_, rewards_, next_gs_, next_ls_, h_, dones_ = \
                self.replayer.sample(self.batch_size)

            batch_gs = torch.FloatTensor(gs_).to(self.device) # [batch_size, agents_num, gs_dim]
            batch_ls = torch.FloatTensor(ls_).to(self.device) # [batch_size, agents_num, ls_dim]
            batch_ph = torch.FloatTensor(ph_).to(self.device) # [batch_size, agents_num, ph_dim]
            batch_la = torch.from_numpy(la_).to(self.device) # [batch_size, 1, agents_num]
            batch_reward = torch.FloatTensor(rewards_).to(self.device) # [batch_size, 1]
            batch_next_gs = torch.FloatTensor(next_gs_).to(self.device) # [batch_size, agents_num, gs_dim]
            batch_next_ls = torch.FloatTensor(next_ls_).to(self.device) # [batch_size, agents_num, ls_dim]
            batch_h = torch.FloatTensor(h_).to(self.device) # [batch_size, agents_num, h_dim]
            batch_done = torch.FloatTensor(dones_).to(self.device) # [batch_size, 1]

            # print(batch_ls.size(), batch_ph.size(), batch_la.size())

            all_local_qs, _ = self.eval_agents(batch_ls, batch_ph, flag='train') # [batch_size, agents_num, la_dim]
            all_next_local_qs, _ = self.target_agents(batch_next_ls, batch_h, flag='train') # [batch_size, agents_num, la_dim]
            # print('1', all_local_qs.size(), all_next_local_qs.size())
            # print('2', batch_la[0], all_local_qs[0])
            all_local_qs = torch.gather(all_local_qs, dim=2, index=batch_la[:, 0, :].unsqueeze(2))[:, :, 0] # [batch_size, agents_num]
            # print('3', all_local_qs[0], all_local_qs.size())
            all_max_next_local_qs, _ = torch.max(all_next_local_qs, dim=2)
            # print('4', all_max_next_local_qs.size())

            q_tot = self.mixingnet(batch_gs, all_local_qs.unsqueeze(1))
            next_max_q_tot = self.mixingnet(batch_next_gs, all_max_next_local_qs.unsqueeze(1))
            q_targets = batch_reward + self.gamma * batch_done * next_max_q_tot

            self.optimizer.zero_grad()
            loss = F.mse_loss(q_tot, q_targets)
            loss.backward()
            self.optimizer.step()

            ''' update target networks '''
            self.target_agents.load_state_dict(self.eval_agents.state_dict())

    def decide(self, local_state, ph):
        with torch.no_grad():
            all_local_qs, all_h = self.eval_agents(local_state, ph, flag='eval')
            local_action = torch.argmax(all_local_qs[0], dim=1)
            # epsilon-greedy policy
            epilon_probs = torch.rand(size=(1, self.agents_num))
            random_action = torch.randint(self.la_dim, size=(1, self.agents_num))
            local_action = torch.where(
                epilon_probs < self.epsilon,
                random_action,
                local_action)
        '''
        global_action shape: [1, agents_num]
        all_h shape: [agents_num, hidden dim]
        '''

        return local_action.detach().cpu().numpy(), all_h[0].detach().cpu().numpy()

    def save(self, save_dir, epoch):
        torch.save(self.eval_agents, '{}/qmix_policy_epoch{}.pth'.format(save_dir, epoch))


