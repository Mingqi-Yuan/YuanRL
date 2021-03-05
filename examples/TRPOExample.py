# -*- coding:utf-8 -*-
__author__ = 'Mingqi Yuan'
"""
An example of the TRPO.
"""
import logging
import torch
import gym
import sys

logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        stream=sys.stdout, datefmt='%H:%M:%S')
sys.path.append('..')

from yuanrl.sarl.apis.TRPO import TRPO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('Acrobot-v1')
env.seed(0)
actor_kwargs = {'input_dim': env.observation_space.shape[0], 'output_dim': env.action_space.n}
critic_kwargs = {'input_dim': env.observation_space.shape[0], 'output_dim': 1}

agent = TRPO(
    device=device,
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    actor_kwargs=actor_kwargs,
    critic_kwargs=critic_kwargs,
    det=False,
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