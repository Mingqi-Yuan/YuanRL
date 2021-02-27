# -*- coding:utf-8 -*-
__author__ = 'Mingqi Yuan'
"""
Implementation of the Ornstein Uhlenbeck Process.
"""

import numpy as np

class OUProcess:
    def __init__(self, size, mu=0., sigma=1., theta=.15, dt=.01):
        self.size = size
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt

    def reset(self, x=0.):
        self.x = x * np.ones(self.size)

    def __call__(self):
        n = np.random.normal(size=self.size)
        self.x += (self.theta * (self.mu - self.x) * self.dt +
                       self.sigma * np.sqrt(self.dt) * n)

        return self.x
