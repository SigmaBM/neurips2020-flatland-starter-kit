import torch
import random
import numpy as np
from collections import namedtuple, deque, Iterable

"""Replay buffer stores all observations, actions, rewards, etc"""
Experience = namedtuple("Experience", field_names=["obs_n", "act_n", "reward", "next_obs_n", "done", "act_mask"])

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        # self.memory = deque(maxlen=buffer_size)
        self.memory = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device

        self._next_idx = 0

    def add(self, obs_n, act_n, reward, next_obs_n, done, act_mask):
        """Add a new experience to memory.
        
        Params
        ======
            obs_n:      array (n_agent, ob_size)
            act_n:      array (n_agent, ac_size)
            reward:     float
            next_obs_n: array (n_agent, ob_size)
            done:       bool
            act_mask:   array (n_agent, )
        """
        e = Experience(obs_n, act_n, reward, next_obs_n, done, act_mask)
        if self._next_idx >= len(self.memory):
            self.memory.append(e)
        else:
            self.memory[self._next_idx] = e
        self._next_idx = (self._next_idx + 1) % self.buffer_size

    def sample_idxes(self):
        """Randomly sample a batch of experiences from memory."""
        idxes = random.sample([i for i in range(len(self.memory))], k=self.batch_size)

        return idxes

    def get(self, idxes):
        experiences = []
        for idx in idxes:
            experiences.append(self.memory[idx])

        obs_n = torch.from_numpy(np.array([e.obs_n for e in experiences])).float().to(self.device)           # (batch_size, n_agent, ob_size)
        act_n = torch.from_numpy(np.array([e.act_n for e in experiences])).float().to(self.device)           # (batch_size, n_agent, ac_size)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences])).float().to(self.device)        # (batch_size, )
        next_obs_n = torch.from_numpy(np.array([e.next_obs_n for e in experiences])).float().to(self.device) # (batch_size, n_agent, ob_size)
        dones = torch.from_numpy(np.array([e.done for e in experiences])).float().to(self.device)            # (batch_size, )
        act_mask = torch.from_numpy(np.array([e.act_mask for e in experiences])).bool().to(self.device)  # (batch_size, n_agent)

        return obs_n, act_n, rewards, next_obs_n, dones, act_mask

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


Experience_2 = namedtuple("Experience", field_names=["obs_n", "act_n", "reward", "next_obs_n", "done", "act_mask", "agent_id"])


class ReplayBufferParamSharing:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        # self.memory = deque(maxlen=buffer_size)
        self.memory = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device

        self._next_idx = 0

    def add(self, obs_n, act_n, reward, next_obs_n, done, act_mask, agent_id):
        """Add a new experience to memory.
        
        Params
        ======
            obs_n:      array (n_agent, ob_size)
            act_n:      array (n_agent, ac_size)
            reward:     float
            next_obs_n: array (n_agent, ob_size)
            done:       array (n_agent, )
            act_mask:   array (n_agent, )
            agent_id:   int
        """
        e = Experience_2(obs_n, act_n, reward, next_obs_n, done, act_mask, agent_id)
        if self._next_idx >= len(self.memory):
            self.memory.append(e)
        else:
            self.memory[self._next_idx] = e
        self._next_idx = (self._next_idx + 1) % self.buffer_size

    def sample_idxes(self):
        """Randomly sample a batch of experiences from memory."""
        idxes = random.sample([i for i in range(len(self.memory))], k=self.batch_size)

        return idxes

    def get(self, idxes):
        experiences = []
        for idx in idxes:
            experiences.append(self.memory[idx])

        obs_n = torch.from_numpy(np.array([e.obs_n for e in experiences])).float().to(self.device)           # (batch_size, n_agent, ob_size)
        act_n = torch.from_numpy(np.array([e.act_n for e in experiences])).float().to(self.device)           # (batch_size, n_agent, ac_size)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences])).float().to(self.device)        # (batch_size, )
        next_obs_n = torch.from_numpy(np.array([e.next_obs_n for e in experiences])).float().to(self.device) # (batch_size, n_agent, ob_size)
        dones = torch.from_numpy(np.array([e.done for e in experiences])).float().to(self.device)        # (batch_size, )
        act_mask = torch.from_numpy(np.array([e.act_mask for e in experiences])).bool().to(self.device)      # (batch_size, n_agent)
        agent_ids = torch.from_numpy(np.array([e.agent_id for e in experiences])).int().to(self.device)      # (batch_size, ) 

        return obs_n, act_n, rewards, next_obs_n, dones, act_mask, agent_ids

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)