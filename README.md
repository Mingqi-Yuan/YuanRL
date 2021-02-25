<div align='center'>
    <img src= 'https://mail.qq.com/cgi-bin/ftnDownload302?sid=9nmejk1-YL_IL2YG&fid=2%2F4ab6a72d-58c5-4259-923f-8dba83b6f145&code=b6968546&k=5f363936026bc9c9edf5e67f1e35061956575b00590206524f0301550d180004570f140f0a06521b5a525b570006560004070d03383d345a0d515618485b53365f' width=620px height=175px>
</div>


**YuanRL is a repository which provides Python implementations of the Reinforcement Learning (RL) algorithms.**

# Installation
Get the repository:
```
$ git clone https://github.com/Mingqi-Yuan/YuanRL.git
```

Run the following command to get dependency:

```
$ pip install -r requirements.txt
```
# Architecture
We consider dividing the RL algorithms into several parts:

* **apis**: Main frameworks of the RL algorithms;
* **nn**: Backbones of the networks in deep RL;
* **replayer**: Replayers for storing and sampling experiences;
* **noise**: Optional noise generators towards continuous tasks.

# Implementations

| Algorithm | Type | Paper |Code|
| ------- | ------- | ------- | ------- |
| Deep Q-learning | Value-based | [[Paper]](https://arxiv.org/pdf/1312.5602.pdf) | [[Code]](apis/DeepQ.py) |
| Proximal Policy Optimization | Policy-based | [[Ppaer]](https://arxiv.org/abs/1707.06347) | [[Code]](apis/PPO.py) |
| Deep Deterministic Policy Gradient | Policy-based | [[Ppaer]](https://arxiv.org/pdf/1509.02971.pdf) | [[Code]](apis/DDPG.py) |
| Soft Actor-Critic Discrete | Policy-based | [[Ppaer]](https://arxiv.org/pdf/1910.07207) | [[Code]](apis/SACDiscrete.py) |

# Example
Run the following example code for training a Deep Q-learning agent:
```python
import torch
import gym
import sys
import os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from apis.DeepQ import DeepQ

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make('MountainCar-v0')
qnet_kwargs = {'input_dim': env.observation_space.shape[0], 'output_dim': env.action_space.n}
agent = DeepQ(
    device=device,
    qnet_kwargs=qnet_kwargs,
    state_dim=2,
    action_dim=3,
    lr=1e-2
)

for game in range(1000):
    state = env.reset()
    episode_reward = 0
    while True:
        action = agent.decide(state)
        next_state, reward, done, info = env.step(action)
        env.render()
        episode_reward += reward

        agent.learn(state, action, reward, next_state, done)

        if done:
            break

        state = next_state

    print('INFO: Round={}, Episode reward={}'.format(game+1, episode_reward))
```
