import torch
import random
import numpy as np
from collections import namedtuple, deque, Iterable
from reinforcement_learning.utils.segment_tree import SumSegmentTree, MinSegmentTree


Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

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
        self.valid = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device

        self._next_idx = 0

    def add(self, state, action, reward, next_state, done, mask):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        # self.memory.append(e)
        if self._next_idx >= len(self.memory):
            self.memory.append(e)
            self.valid.append(mask)
        else:
            self.memory[self._next_idx] = e
            self.valid[self._next_idx] = mask
        self._next_idx = (self._next_idx + 1) % self.buffer_size

    def sample_idxes(self):
        """Randomly sample a batch of experiences from memory."""
        idxes = random.sample(list(np.asarray(self.valid).nonzero()[0]), k=self.batch_size)
        # experiences = random.sample(self.memory, k=self.batch_size)

        return idxes

    def get(self, idxes):
        experiences = []
        for idx in idxes:
            experiences.append(self.memory[idx])

        states = torch.from_numpy(self._v_stack_impr([e.state for e in experiences if e is not None])) \
            .float().to(self.device)
        actions = torch.from_numpy(self._v_stack_impr([e.action for e in experiences if e is not None])) \
            .long().to(self.device)
        rewards = torch.from_numpy(self._v_stack_impr([e.reward for e in experiences if e is not None])) \
            .float().to(self.device)
        next_states = torch.from_numpy(self._v_stack_impr([e.next_state for e in experiences if e is not None])) \
            .float().to(self.device)
        dones = torch.from_numpy(self._v_stack_impr([e.done for e in experiences if e is not None]).astype(np.uint8)) \
            .float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal valid memory."""
        # return len(self.memory)
        return len(np.asarray(self.valid).nonzero()[0])

    def _v_stack_impr(self, states):
        # For states with shape (1, ob_size)
        sub_dim = len(states[0]) if isinstance(states[0], Iterable) else 1
        np_states = np.reshape(np.array(states), (len(states), sub_dim))
        return np_states


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized replay buffer. (adapted from openai/baselines/deepq)"""
    def __init__(self, action_size, buffer_size, batch_size, device, alpha):
        """Initialize a Prioritized ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            alpha (float): how much prioritization is used
                (0 - no prioritization, 1 - full prioritization)
        """
        raise NotImplementedError
        super().__init__(action_size, buffer_size, batch_size, device)
        assert alpha >= 0
        self._alpha = alpha  

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, state, action, reward, next_state, done, mask):
        idx = self._next_idx
        super().add(state, action, reward, next_state, done, mask)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha
    
    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self.memory) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res
    
    def sample(self, beta):
        """Sample a batch of experiences.

        Parameters
        ----------
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        """
        assert beta > 0

        idxes = self._sample_proportional(self.batch_size)

        weights, experiences = [], []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self.memory)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self.memory)) ** (-beta)
            weights.append(weight / max_weight)
            experiences.append(self.memory[idx])
        weights = np.array(weights)

        states = torch.from_numpy(self._v_stack_impr([e.state for e in experiences if e is not None])) \
            .float().to(self.device)
        actions = torch.from_numpy(self._v_stack_impr([e.action for e in experiences if e is not None])) \
            .long().to(self.device)
        rewards = torch.from_numpy(self._v_stack_impr([e.reward for e in experiences if e is not None])) \
            .float().to(self.device)
        next_states = torch.from_numpy(self._v_stack_impr([e.next_state for e in experiences if e is not None])) \
            .float().to(self.device)
        dones = torch.from_numpy(self._v_stack_impr([e.done for e in experiences if e is not None]).astype(np.uint8)) \
            .float().to(self.device)

        return states, actions, rewards, next_states, dones, weights, idxes
    
    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.memory)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)