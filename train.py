import gymnasium as gym
import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in indices))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)


def create_env(num_envs=4):
    env_fns = [lambda: gym.make('CartPole-v1') for _ in range(num_envs)]
    return gym.vector.SyncVectorEnv(env_fns)


def main():
    num_envs = 4
    env = create_env(num_envs)

    # Unpack reset
    state, _ = env.reset()

    replay_buffer = ReplayBuffer()
    total_steps = 1000

    for _ in range(total_steps):
        # Random policy for demonstration
        actions = env.action_space.sample()

        # Unpack step result and compute dones
        next_state, rewards, terminated, truncated, _ = env.step(actions)
        dones = np.logical_or(terminated, truncated)

        # Update replay buffer with dones
        for i in range(num_envs):
            replay_buffer.add(state[i], actions[i], rewards[i], next_state[i], dones[i])

        # Reset finished envs
        done_indices = np.nonzero(dones)[0]
        if len(done_indices) > 0:
            if hasattr(env, 'reset_done'):
                state[done_indices], _ = env.reset_done(done_indices)
                state[~dones] = next_state[~dones]
            else:
                for idx in done_indices:
                    state[idx], _ = env.reset()
                state[~dones] = next_state[~dones]
        else:
            state = next_state

    print('Collected', len(replay_buffer), 'transitions')


if __name__ == '__main__':
    main()
