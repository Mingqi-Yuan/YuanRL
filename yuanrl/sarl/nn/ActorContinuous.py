# -*- coding:utf-8 -*-
__author__ = 'Mingqi Yuan'
"""
Implementation of the actor network towards the continuous action space.
"""

from torch import nn, optim
import torch

class ActorCont(nn.Module):
    def __init__(self, kwargs):
        super(ActorCont, self).__init__()

        self.fc1 = nn.Linear(kwargs['input_dim'], 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, kwargs['output_dim'])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)

        return x
