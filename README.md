<div align='center'>
    <img src= 'https://img-blog.csdnimg.cn/20210225200756161.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjQ5OTIzNg==,size_16,color_FFFFFF,t_70#pic_center' width=932px height=249px>
</div>


**YuanRL is a repository which provides Python implementations of the Single-Agent Reinforcement Learning (SARL) and Multi-Agent Reinforcement Learning (MARL) algorithms.**

# Installation

* Install the package from the PYPI:
```
pip install yuanrl -i https://pypi.org/simple/ 
```

* Get the repository with git:
```
git clone https://github.com/Mingqi-Yuan/YuanRL.git
```

* Run the following command to get dependency:

```
pip install -r requirements.txt
```
# Architecture
We consider dividing the RL algorithms into several parts:

* **apis**: Main frameworks of the RL algorithms;
* **nn**: Backbones of the networks in deep RL;
* **replayer**: Replayers for storing and sampling experiences;
* **noise**: Optional noise generators towards continuous tasks.

# Implementations

* Single-Agent RL algorithms

| Algorithm | Type | On/Off-policy | Available task |Paper | Code |
| ------- | ------- | ------- | ------- | ------- | ------- |
| Deep Q-learning | Value-based | Off-policy | Discrete | [[Paper]](https://arxiv.org/pdf/1312.5602.pdf) | [[Code]](yuanrl/sarl/apis/DeepQ.py) |
| Proximal Policy Optimization | Policy-based | On-policy| Discrete |[[Paper]](https://arxiv.org/abs/1707.06347) | [[Code]](yuanrl/sarl/apis/PPO.py) |
| Trust Region Policy Optimization | Policy-based | On-policy| Discrete |[[Paper]](https://spinningup.openai.com/en/latest/algorithms/trpo.html#) | [[Code]](yuanrl/sarl/apis/TRPO.py) |
| Deep Deterministic Policy Gradient | Policy-based | Off-policy| Continuous |[[Paper]](https://arxiv.org/pdf/1509.02971.pdf) | [[Code]](yuanrl/sarl/apis/DDPG.py) |
| Soft Actor-Critic Discrete | Policy-based | Off-policy| Discrete |[[Paper]](https://arxiv.org/pdf/1910.07207) | [[Code]](yuanrl/sarl/apis/SACDiscrete.py) |

Note that the "available task" here means the implemented version of the algorithm, which does not represent that an
algorithm can only handle single-type task. For instance, the PPO can handle not only discrete tasks but also continuous tasks, but we
only implement the discrete version here. 

* Multi-Agent RL algorithms

| Algorithm | Type | On/Off-policy | Available task |Paper | Code |
| ------- | ------- | ------- | ------- | ------- | ------- |
| QMIX | Value-based | Off-policy | Discrete |[[Paper]](http://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf) | [[Code]](yuanrl/marl/apis/QMIX.py) |

# Example
Run the following example code for training a PPO agent:

```python
import logging
import torch
import gym
import sys
import os

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    stream=sys.stdout, datefmt='%H:%M:%S')
sys.path.append('..')

from yuanrl.sarl.apis.PPO import PPO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('Acrobot-v1')
env.seed(0)
actor_kwargs = {'input_dim': env.observation_space.shape[0], 'output_dim': env.action_space.n}
critic_kwargs = {'input_dim': env.observation_space.shape[0], 'output_dim': 1}

agent = PPO(
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

    logging.info('Episode={}, Reward={}'.format(game + 1, episode_reward))
```
# Acknowledgement
I'd like to appreciate for the excellent book and code written by [Dr. Zhiqing Xiao](https://github.com/ZhiqingXiao/rl-book) .
