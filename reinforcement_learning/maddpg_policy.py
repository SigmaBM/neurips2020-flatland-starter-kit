import os
import copy
import torch
import random
import pickle
import torch.nn as nn
import numpy as np

from torch.optim import Adam
from model import Actor, LocalCritic
from replay_buffer import ReplayBuffer
from reinforcement_learning.utils.misc import gumbel_softmax, onehot_from_logits


class MADDPGPolicy(object):
    def __init__(self, ob_size, ac_size, n_agent, idx, parameters, evaluation_mode=False):
        self.evaluation_mode = evaluation_mode

        self.ob_size = ob_size
        self.ac_size = ac_size
        self.id = idx
        self.n_agent = n_agent
        self.hid_size = 1

        if not evaluation_mode:
            self.hid_size = parameters.hidden_size
            self.buffer_size = parameters.buffer_size
            self.batch_size = parameters.batch_size
            self.update_every = parameters.update_every
            self.learning_rate = parameters.learning_rate
            self.tau = parameters.tau
            self.gamma = parameters.gamma
            self.buffer_min_size = parameters.buffer_min_size

        # Device
        if parameters.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            # print("🐇 Using GPU")
        else:
            self.device = torch.device("cpu")
            # print("🐢 Using CPU")
        
        self.p = Actor(ob_size, ac_size, self.hid_size).to(self.device)
        self.q = LocalCritic(ob_size, ac_size * n_agent, self.hid_size).to(self.device)

        if not evaluation_mode:
            self.target_p = copy.deepcopy(self.p)
            self.target_q = copy.deepcopy(self.q)

            self.p_optimizer = Adam(self.p.parameters(), lr=self.learning_rate)
            self.q_optimizer = Adam(self.q.parameters(), lr=self.learning_rate)

            self.memory = ReplayBuffer(ac_size, self.buffer_size, self.batch_size, self.device)

            self.t_step = 0
            self.pi_loss = 0.0
            self.vf_loss = 0.0
    
    def act(self, obs, explore=False):
        """
        Inputs:
            obs: (batch_size, ob_size)
        Outputs:
            actions: (batch_size, ac_size) - one hot vector
        """
        obs = torch.from_numpy(obs).float().to(self.device)
        pi = self.p(obs)
        if explore:
            action = gumbel_softmax(pi, hard=True)
        else:
            action = onehot_from_logits(pi)
        return action
    
    def update_memory(self, obs, action, reward, next_obs, done):
        assert not self.evaluation_mode, "Policy has been initialized for evaluation only."

        # Save experience in replay memory
        self.memory.add(obs, action, reward, next_obs, done)
    
    def learn(self, agents):
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.buffer_min_size and len(self.memory) > self.batch_size:
            # Time to learn!
            idxes = random.sample([i for i in range(len(self.memory))], k=self.batch_size)

            obs_n, act_n, rew_n, next_obs_n, done_n = [], [], [], [], []
            for i in range(self.n_agent):
                obs, act, rew, next_obs, done = agents[i].memory.get(idxes)
                obs_n.append(obs)
                act_n.append(act)
                rew_n.append(rew)
                next_obs_n.append(next_obs)
                done_n.append(done)
            
            # 1. Update critic
            next_act_n = [onehot_from_logits(agents[i].target_p(obs_n[i])) for i in range(self.n_agent)]
            next_act_cat = torch.cat(tuple(next_act_n), dim=1)
            # y_i = r_i + gamma * Q_target(o_i, a_1, a_2, ..., a_n) * (1 - treminal_i)
            target_q = rew_n[self.id].view(-1, 1) + self.gamma * self.target_q(next_obs_n[self.id], next_act_cat) * (1 - done_n[self.id].view(-1, 1))

            act_cat = torch.cat(tuple(act_n), dim=1)
            q = self.q(obs_n[self.id], act_cat)
            self.vf_loss = torch.nn.MSELoss()(q, target_q)

            self.q_optimizer.zero_grad()
            self.vf_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), 0.5)
            self.q_optimizer.step()

            # 2. Update actor
            pi = self.p(obs_n[self.id])
            act = gumbel_softmax(pi, hard=True)
            act_n[self.id] = act
            act_cat = torch.cat(tuple(act_n), dim=1)
            pg_loss = -self.q(obs_n[self.id], act_cat).mean()

            p_reg = (pi**2).mean()
            self.pi_loss = pg_loss + p_reg * 1e-3
            
            self.p_optimizer.zero_grad()
            self.pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.p.parameters(), 0.5)
            self.p_optimizer.step()

            self.soft_update()

    def soft_update(self):
        for target_param, real_param in zip(self.target_q.parameters(), self.q.parameters()):
            target_param.data.copy_(self.tau * real_param.data + (1.0 - self.tau) * target_param.data)
        for target_param, real_param in zip(self.target_p.parameters(), self.p.parameters()):
            target_param.data.copy_(self.tau * real_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.q.state_dict(), filename + ".q")
        torch.save(self.p.state_dict(), filename + '.p')
        torch.save(self.target_q.state_dict(), filename + ".target_q")
        torch.save(self.target_p.state_dict(), filename + ".target_p")

    def load(self, filename):
        if os.path.exists(filename + ".q"):
            self.q.load_state_dict(torch.load(filename + ".q"))
        if os.path.exists(filename + ".p"):
            self.p.load_state_dict(torch.load(filename + ".p"))
        if os.path.exists(filename + ".target_q"):
            self.target_q.load_state_dict(torch.load(filename + ".target_q"))
        if os.path.exists(filename + ".target_p"):
            self.target_p.load_state_dict(torch.load(filename + ".target_p"))

    def save_replay_buffer(self, filename):
        memory = self.memory.memory
        with open(filename, 'wb') as f:
            pickle.dump(list(memory)[-500000:], f)

    def load_replay_buffer(self, filename):
        with open(filename, 'rb') as f:
            self.memory.memory = pickle.load(f)

    def test(self):
        self.act(np.array([[0] * self.ob_size]))
        self.learn()