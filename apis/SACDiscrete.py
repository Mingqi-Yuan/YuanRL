import tensorflow as tf
import numpy as np
import pandas as pd

class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                columns=['observation', 'action', 'reward',
                'next_observation', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in
                self.memory.columns)

class SACAgent:
    def __init__(self, env, actor_kwargs, critic_kwargs,
            replayer_capacity=10000, gamma=0.99, alpha=0.99,
            batches=1, batch_size=64, net_learning_rate=0.995):
        observation_dim = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.alpha = alpha
        self.net_learning_rate = net_learning_rate # 网络学习速度
        self.batches = batches
        self.batches = batches
        self.batch_size = batch_size

        self.replayer = DQNReplayer(replayer_capacity)

        def sac_loss(y_true, y_pred):
            # 损失函数。参数y_true是Q(*, action_n)，y_pred是pi(*, action_n)
            qs = alpha * tf.math.xlogy(y_pred, y_pred) - y_pred * y_true
            return tf.reduce_sum(qs, axis=-1)

        self.actor_net = self.build_network(input_size=observation_dim,
                                            output_size=self.action_n, output_activation=tf.nn.softmax,
                                            loss=sac_loss, **actor_kwargs)  # 执行者网络
        self.q0_net = self.build_network(input_size=observation_dim,
                                         output_size=self.action_n, **critic_kwargs)  # 动作价值网络
        self.q1_net = self.build_network(input_size=observation_dim,
                                         output_size=self.action_n, **critic_kwargs)  # 动作价值网络
        self.v_evaluate_net = self.build_network(
            input_size=observation_dim, output_size=1, **critic_kwargs)
        self.v_target_net = self.build_network(input_size=observation_dim, output_size=1, **critic_kwargs)

        self.update_target_net(self.v_target_net, self.v_evaluate_net)

    def build_network(self, input_size, hidden_sizes, output_size,
            activation=tf.nn.relu, output_activation=None,
            loss=tf.keras.losses.mse, learning_rate=0.01): # 构建网络
        model = tf.keras.Sequential()
        for layer, hidden_size in enumerate(hidden_sizes):
            kwargs = {'input_shape' : (input_size,)} if layer == 0 else {}
            model.add(tf.keras.layers.Dense(units=hidden_size,
                    activation=activation, **kwargs)) # 隐藏层
        model.add(tf.keras.layers.Dense(units=output_size,
                activation=output_activation)) # 输出层
        optimizer = tf.keras.optimizers.Adam(learning_rate) # 优化器
        model.compile(optimizer=optimizer, loss=loss)
        return model

    def update_target_net(self, target_net, evaluate_net, learning_rate=1.):
        target_weights = target_net.get_weights()
        evaluate_weights = evaluate_net.get_weights()
        average_weights = [(1. - learning_rate) * t + learning_rate * e
                           for t, e in zip(target_weights, evaluate_weights)]
        target_net.set_weights(average_weights)

    def decide(self, observation):
        probs = self.actor_net.predict(observation[np.newaxis])[0]  # 计算概率
        action = np.random.choice(self.action_n, p=probs)
        return action

    def learn(self, observation, action, reward, next_observation, done):
        self.replayer.store(observation, action, reward, next_observation,
                            done)  # 存储经验

        if done:
            for batch in range(self.batches):
                observations, actions, rewards, next_observations, \
                dones = self.replayer.sample(self.batch_size)  # 经验回放
                pis = self.actor_net.predict(observations)
                q0s = self.q0_net.predict(observations)
                q1s = self.q1_net.predict(observations)

                self.actor_net.fit(observations, q0s, verbose=0)  # 更新执行者

                q01s = np.minimum(q0s, q1s)
                entropic_q01s = q01s - self.alpha * np.log(pis)
                v_targets = (pis * entropic_q01s).mean(axis=1)
                self.v_evaluate_net.fit(observations, v_targets, verbose=0)

                next_vs = self.v_target_net.predict(next_observations)
                q_targets = rewards + \
                            self.gamma * (1. - dones) * next_vs[:, 0]
                q0s[range(self.batch_size), actions] = q_targets
                q1s[range(self.batch_size), actions] = q_targets
                self.q0_net.fit(observations, q0s, verbose=0)
                self.q1_net.fit(observations, q1s, verbose=0)  # 更新动作价值网络

                self.update_target_net(self.v_target_net, self.v_evaluate_net, self.net_learning_rate)

import gym
env = gym.make('MountainCar-v0')

actor_kwargs = {'hidden_sizes' : [100,], 'learning_rate' : 0.001}
critic_kwargs = {'hidden_sizes' : [100,], 'learning_rate' : 0.001}
agent = SACAgent(
    env=env,
    actor_kwargs=actor_kwargs,
    critic_kwargs=critic_kwargs
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
