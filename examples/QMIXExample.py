# -*- coding:utf-8 -*-
__author__ = 'Mingqi Yuan'
"""
An example of the QMIX.
"""
import logging
import torch
import sys

logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        stream=sys.stdout, datefmt='%H:%M:%S')

from yuanrl.apis.QMIX import QMIX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agents_num = 5
gs_dim = 12
ls_dim = 7
ga_dim = agents_num
la_dim = 4
hidden_dim = 64

agent_kwargs = {'input_dim':ls_dim, 'output_dim':la_dim, 'hidden_dim':hidden_dim}
mixing_kwargs = {
    'hyper_input_dim':gs_dim,
    'mixing_input_dim':agents_num,
    'mixing_output_dim':1
                 }

agent = QMIX(
    device=device,
    gs_dim=gs_dim,
    ga_dim=ga_dim,
    ls_dim=ls_dim,
    la_dim=la_dim,
    agents_num=agents_num,
    agent_kwargs=agent_kwargs,
    mixing_kwargs=mixing_kwargs,
    replayer_initial_transitions=10,
    batch_size=33
)

import numpy as np

for game in range(1000):
    gs = np.random.rand(gs_dim)
    ls = np.random.rand(agents_num, ls_dim)
    ph = torch.zeros(size=(agents_num, hidden_dim))

    episode_reward = 0
    while True:
        la, h = agent.decide(ls, ph)

        # next_gs, next_ls, reward, done = env.step(la)
        next_gs = np.random.rand(gs_dim)
        next_ls = np.random.rand(agents_num, ls_dim)
        reward = 1.
        done = False

        agent.learn(gs, ls, ph, la, reward, next_gs, next_ls, h, done)
        episode_reward += reward

        if done:
            break

        gs = next_gs
        ls = next_ls
        ph = h

    logging.info('Episode={}, Reward={}'.format(game+1, episode_reward))