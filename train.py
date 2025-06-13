import os
import numpy as np
import torch
from gym.vector import SyncVectorEnv
import time
from tqdm import trange

from solitaire_env import KlondikeEnv
from dqn_agent import DQNAgent


def make_env():
    return KlondikeEnv()


def train(total_steps=10000000, num_envs=32, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    sample_env = KlondikeEnv()
    agent = DQNAgent(sample_env.observation_space.shape[0], sample_env.action_space.n)
    print(
        f"Starting training: total_steps={total_steps}, num_envs={num_envs}, device={agent.device}"
    )
    start_time = time.time()
    env = SyncVectorEnv([make_env for _ in range(num_envs)])
    state, _ = env.reset()
    wins = np.zeros(num_envs, dtype=int)

    for step in trange(1, total_steps + 1, desc="Training", unit="step"):
        actions = [agent.select_action(s) for s in state]
        next_state, rewards, dones, truncs, _ = env.step(actions)
        done_flags = np.logical_or(dones, truncs)
        for i in range(num_envs):
            agent.store_transition(state[i], actions[i], rewards[i], next_state[i], done_flags[i])
            if done_flags[i] and rewards[i] > 0:
                wins[i] += 1
        loss = agent.update()
        agent.step()
        if loss is not None:
            print(f"[Step {step}] loss={loss:.4f}   ε={agent.epsilon:.3f}")
        state = next_state

        if step % 10000 == 0:
            avg_win = wins.sum() / (num_envs)
            torch_path = os.path.join(checkpoint_dir, f"dqn_{step}.pth")
            torch.save(agent.q_net.state_dict(), torch_path)
            elapsed = time.time() - start_time
            print(f"\u2192 Step {step}: avg wins={avg_win:.3f}   elapsed={elapsed/60:.1f}m")
            print(f"  Saved checkpoint: {torch_path}")
            wins[:] = 0

    total_time = (time.time() - start_time) / 3600
    print(f"Training complete in {total_time:.1f} hours")


if __name__ == "__main__":
    train()
