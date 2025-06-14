import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity, action_dim):
        self.buffer = deque(maxlen=capacity)
        self.action_dim = action_dim

    def push(self, state, action, reward, next_state, done, legal_next):
        self.buffer.append((state, action, reward, next_state, done, legal_next))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, legal_next = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            np.array(legal_next, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        obs_dim,
        action_dim,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        warmup_steps=10_000,
        decay_steps=50_000,
        lr=3e-4,
        batch_size=64,
        target_sync=500,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(200000, action_dim)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.epsilon = epsilon_start
        self.total_steps = 0
        self.target_sync = target_sync
        self.action_dim = action_dim

    def select_action(self, state, legal_actions=None, eval=False):
        if legal_actions is None:
            legal_actions = list(range(self.action_dim))

        if (not eval) and random.random() < self.epsilon:
            return random.choice(legal_actions)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.q_net(state_t).cpu().numpy().flatten()

        legal_q = [q_vals[a] for a in legal_actions]
        best_idx = int(np.argmax(legal_q))
        return legal_actions[best_idx]

    def store_transition(self, *args):
        self.replay_buffer.push(*args)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones, legal_next = (
            self.replay_buffer.sample(self.batch_size)
        )
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        legal_next = torch.FloatTensor(legal_next).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_all = self.target_net(next_states)
            illegal_mask = legal_next == 0
            next_q_all[illegal_mask] = -float("inf")
            next_q = next_q_all.max(1)[0].unsqueeze(1)
            target = rewards + self.gamma * next_q * (1 - dones)
        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.total_steps % self.target_sync == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def step(self, num_envs=1):
        self.total_steps += num_envs
        t = self.total_steps
        if t < self.warmup_steps:
            self.epsilon = self.epsilon_start
        elif t < self.warmup_steps + self.decay_steps:
            frac = (t - self.warmup_steps) / self.decay_steps
            self.epsilon = (
                self.epsilon_start - (self.epsilon_start - self.epsilon_end) * frac
            )
        else:
            self.epsilon = self.epsilon_end
