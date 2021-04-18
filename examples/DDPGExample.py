# -*- coding:utf-8 -*-
__author__ = 'Mingqi Yuan'
"""
An example of the DDPG.
"""

import logging
import torch
import gym
import sys

logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        stream=sys.stdout, datefmt='%H:%M:%S')
sys.path.append('..')

from yuanrl.apis.DDPG import DDPG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('Pendulum-v0')
env.seed(0)
actor_kwargs = {'input_dim': env.observation_space.shape[0], 'output_dim': 1}
critic_kwargs = {'input_dim': env.observation_space.shape[0]+env.action_space.shape[0], 'output_dim': 1}

agent = DDPG(
    device=device,
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    action_high=env.action_space.high[0],
    action_low=env.action_space.low[0],
    actor_kwargs=actor_kwargs,
    critic_kwargs=critic_kwargs,
    replayer_initial_transitions=3000,
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