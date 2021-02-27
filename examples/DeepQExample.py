# -*- coding:utf-8 -*-
__author__ = 'Mingqi Yuan'
"""
An example of the Deep Q-learning.
"""

import logging
import torch
import gym
import sys
import os

logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        stream=sys.stdout, datefmt='%H:%M:%S')
sys.path.append('..')

from apis.DeepQ import DeepQ

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('MountainCar-v0')
env.seed(0)

qnet_kwargs = {'input_dim': env.observation_space.shape[0], 'output_dim': env.action_space.n}
agent = DeepQ(
    device=device,
    qnet_kwargs=qnet_kwargs,
    state_dim=2,
    action_dim=3,
    lr=1e-3
)

for game in range(1000):
    state = env.reset()
    episode_reward = 0
    while True:
        action = agent.decide(state)
        next_state, reward, done, info = env.step(action)
        # env.render()
        episode_reward += reward

        agent.learn(state, action, reward, next_state, done)

        if done:
            break

        state = next_state

    logging.info('Episode={}, Reward={}'.format(game+1, episode_reward))