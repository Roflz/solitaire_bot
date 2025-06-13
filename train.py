import os
import numpy as np
import torch
from gym.vector import SyncVectorEnv

from solitaire_env import KlondikeEnv
from dqn_agent import DQNAgent


def make_env():
    return KlondikeEnv()


def train(total_steps=10000000, num_envs=32, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    env = SyncVectorEnv([make_env for _ in range(num_envs)])
    agent = DQNAgent(env.single_observation_space.shape[0], env.single_action_space.n)
    state, _ = env.reset()
    wins = np.zeros(num_envs, dtype=int)

    for step in range(1, total_steps + 1):
        actions = [agent.select_action(s) for s in state]
        next_state, rewards, dones, truncs, _ = env.step(actions)
        done_flags = np.logical_or(dones, truncs)
        for i in range(num_envs):
            agent.store_transition(state[i], actions[i], rewards[i], next_state[i], done_flags[i])
            if done_flags[i] and rewards[i] > 0:
                wins[i] += 1
        agent.update()
        agent.step()
        state = next_state

        if step % 10000 == 0:
            avg_win = wins.sum() / (num_envs)
            print(f"Step {step}: avg wins {avg_win}")
            torch_path = os.path.join(checkpoint_dir, f"dqn_{step}.pth")
            torch.save(agent.q_net.state_dict(), torch_path)
            wins[:] = 0


if __name__ == "__main__":
    train()
