import random
from collections import deque

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.advantage = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x) + x)
        x = F.relu(self.fc3(x) + x)
        adv = self.advantage(x)
        val = self.value(x)
        return val + adv - adv.mean(dim=1, keepdim=True)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, action_dim, alpha=0.6):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.action_dim = action_dim
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done, legal_next):
        max_prio = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done, legal_next))
        self.priorities.append(max_prio)

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return None
        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        states, actions, rewards, next_states, dones, legal_next = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            np.array(legal_next, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32).reshape(-1, 1),
        )

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = float(prio)

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
        prio_alpha=0.6,
        prio_beta_start=0.4,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.replay_buffer = PrioritizedReplayBuffer(200000, action_dim, alpha=prio_alpha)
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
        self.prio_beta_start = prio_beta_start

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
        sample = self.replay_buffer.sample(self.batch_size, beta=min(1.0, self.prio_beta_start + self.total_steps * 1e-6))
        if sample is None:
            return None
        states, actions, rewards, next_states, dones, legal_next, indices, weights = sample
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        legal_next = torch.FloatTensor(legal_next).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_all_online = self.q_net(next_states)
            illegal_mask = legal_next == 0
            next_q_all_online[illegal_mask] = -float("inf")
            best_next_actions = next_q_all_online.max(1)[1].unsqueeze(1)
            next_q_all_target = self.target_net(next_states)
            next_q_all_target[illegal_mask] = -float("inf")
            next_q = next_q_all_target.gather(1, best_next_actions)
            target = rewards + self.gamma * next_q * (1 - dones)
        td_error = target - q_values
        loss = (weights_t * nn.functional.smooth_l1_loss(q_values, target, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        prios = td_error.abs().detach().cpu().numpy() + 1e-5
        self.replay_buffer.update_priorities(indices, prios)

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
