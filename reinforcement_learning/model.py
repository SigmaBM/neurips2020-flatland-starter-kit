import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    """Dueling Q-network (https://arxiv.org/abs/1511.06581)"""

    def __init__(self, state_size, action_size, hidsize1=128, hidsize2=128):
        super(DuelingQNetwork, self).__init__()

        # value network
        self.fc1_val = nn.Linear(state_size, hidsize1)
        self.fc2_val = nn.Linear(hidsize1, hidsize2)
        self.fc4_val = nn.Linear(hidsize2, 1)

        # advantage network
        self.fc1_adv = nn.Linear(state_size, hidsize1)
        self.fc2_adv = nn.Linear(hidsize1, hidsize2)
        self.fc4_adv = nn.Linear(hidsize2, action_size)

    def forward(self, x):
        val = F.relu(self.fc1_val(x))
        val = F.relu(self.fc2_val(val))
        val = self.fc4_val(val)

        # advantage calculation
        adv = F.relu(self.fc1_adv(x))
        adv = F.relu(self.fc2_adv(adv))
        adv = self.fc4_adv(adv)

        return val + adv - adv.mean()


class Actor(nn.Module):
    def __init__(self, ob_size, ac_size, hid_size=128):
        super().__init__()

        # policy network (only for discrete action space)
        self.fc1_pi = nn.Linear(ob_size, hid_size)
        self.fc2_pi = nn.Linear(hid_size, hid_size)
        self.fc3_pi = nn.Linear(hid_size, ac_size)
    
    def forward(self, obs):
        pi = F.relu(self.fc1_pi(obs))
        pi = F.relu(self.fc2_pi(pi))
        pi = self.fc3_pi(pi)

        return pi


class LocalCritic(nn.Module):
    def __init__(self, ob_size, ac_size, hid_size=128):
        super().__init__()

        self.fc1_q = nn.Linear(ob_size + ac_size, hid_size)
        self.fc2_q = nn.Linear(hid_size, hid_size)
        self.fc3_q = nn.Linear(hid_size, 1)

    def forward(self, obs, acts):
        q = F.relu(self.fc1_q(torch.cat((obs, acts), dim=1)))
        q = F.relu(self.fc2_q(q))
        q = self.fc3_q(q)

        return q
