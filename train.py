import os
import time
import numpy as np
import torch
from gym.vector import SyncVectorEnv
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from solitaire_env import KlondikeEnv
from dqn_agent import DQNAgent


def run_evaluation(agent, num_games):
    eval_env = SyncVectorEnv([make_env for _ in range(num_games)])
    state, _ = eval_env.reset()
    wins = 0
    done_flags = np.zeros(num_games, dtype=bool)
    while not done_flags.all():
        actions = [agent.select_action(s, eval=True) for s in state]
        next_state, rewards, dones, truncs, _ = eval_env.step(actions)
        wins += rewards.sum()
        state = next_state
        done_flags = np.logical_or(dones, truncs)
    return wins / num_games


def make_env():
    return KlondikeEnv()


def train(total_steps=10000000, num_envs=32, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Logging intervals
    log_interval = 1000
    tb_interval = 500
    plot_interval = 2000
    eval_interval = 10000
    num_eval_games = 100

    writer = SummaryWriter(log_dir="runs/solitaire")
    metrics = {"loss": [], "train_win_rate": [], "eval_win_rate": []}
    steps = []

    sample_env = KlondikeEnv()
    agent = DQNAgent(sample_env.observation_space.shape[0], sample_env.action_space.n)
    print(
        f"Starting training: total_steps={total_steps}, num_envs={num_envs}, device={agent.device}"
    )
    start_time = time.time()
    env = SyncVectorEnv([make_env for _ in range(num_envs)])
    state, _ = env.reset()
    wins = np.zeros(num_envs, dtype=int)
    recent_win_rate = 0

    for step in trange(
        1,
        total_steps + 1,
        desc="Training",
        unit="step",
        mininterval=5,
        leave=True,
    ):
        actions = [agent.select_action(s) for s in state]
        next_state, rewards, dones, truncs, _ = env.step(actions)
        done_flags = np.logical_or(dones, truncs)
        for i in range(num_envs):
            agent.store_transition(state[i], actions[i], rewards[i], next_state[i], done_flags[i])
            if done_flags[i] and rewards[i] > 0:
                wins[i] += 1
        loss = agent.update()
        agent.step()
        state = next_state

        if step % log_interval == 0 and loss is not None:
            recent_win_rate = wins.sum() / num_envs
            print(
                f"[Step {step}] loss={loss:.4f}, ε={agent.epsilon:.2f}, win_rate={recent_win_rate:.3f}"
            )
            wins[:] = 0

        if step % tb_interval == 0 and loss is not None:
            writer.add_scalar("train/loss", loss, step)
            writer.add_scalar("train/ε", agent.epsilon, step)
            writer.add_scalar("train/win_rate", recent_win_rate, step)

        if step % eval_interval == 0:
            old_eps = agent.epsilon
            agent.epsilon = 0.0
            eval_win_rate = run_evaluation(agent, num_eval_games)
            writer.add_scalar("eval/win_rate", eval_win_rate, step)
            print(f">>> Eval @ {step}: win_rate={eval_win_rate:.3f}")
            agent.epsilon = old_eps

        if step % plot_interval == 0 and loss is not None:
            steps.append(step)
            metrics["loss"].append(loss)
            metrics["train_win_rate"].append(recent_win_rate)
            metrics["eval_win_rate"].append(
                eval_win_rate if "eval_win_rate" in locals() else np.nan
            )
            plt.figure()
            plt.plot(steps, metrics["train_win_rate"], label="Train Win Rate")
            plt.plot(steps, metrics["eval_win_rate"], label="Eval Win Rate")
            plt.xlabel("Steps")
            plt.ylabel("Win Rate")
            plt.legend()
            plt.savefig(os.path.join(plots_dir, f"win_rate_{step}.png"))
            plt.close()

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
    writer.close()


if __name__ == "__main__":
    train()
