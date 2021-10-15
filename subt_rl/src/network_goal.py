import os
import copy
import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        init_w: float = 3e-3,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        kernel = 3
        stride = 1
        self.conv = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=kernel, stride=stride),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=kernel, stride=stride),
            nn.ReLU()
        )
        in_dim = 241
        dim = 32 * (in_dim - 2*(kernel-stride))

        self.linear1 = nn.Linear(dim, 512)
        self.linear2 = nn.Linear(512+30, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 128)

        self.out = nn.Linear(128, out_dim)

        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        # split to feature, track
        x, other = torch.split(state, 4*241, dim=1)

        # expand to [batch, channel*4, features]
        x = x.reshape(state.shape[0], 4, -1)

        # print(x.shape,a.shape)

        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = torch.cat((x, other), dim=-1)
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        act = self.out(x)
        # linear action : sigmoid
        # angular action : tanh
        linear_act = act[:, 0]
        angular_act = act[:, 1]
        linear_act = torch.sigmoid(linear_act)
        angular_act = torch.tanh(angular_act)
        action = torch.stack((linear_act, angular_act), dim=1)

        return action


class Critic(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        init_w: float = 3e-3,
    ):
        """Initialize."""
        super(Critic, self).__init__()

        kernel = 3
        stride = 1
        self.conv = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=kernel, stride=stride),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=kernel, stride=stride),
            nn.ReLU()
        )
        in_dim = 241
        dim = 32 * (in_dim - 2*(kernel-stride))

        self.linear1 = nn.Linear(dim, 512)
        self.linear2 = nn.Linear(512+30+2, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)

        self.out.weight.data.uniform_(-init_w, init_w)
        self.out.bias.data.uniform_(-init_w, init_w)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Forward method implementation."""

        # split to feature
        x, other = torch.split(state, 4*241, dim=1)

        # expand to [batch, channel*4, features]
        x = x.reshape(state.shape[0], 4, -1)

        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = torch.cat((x, other), dim=-1)
        x = torch.cat((x, action), dim=-1)
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        value = self.out(x)

        return value
