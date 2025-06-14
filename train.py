import os
import time
import numpy as np
import torch
from gym.vector import SyncVectorEnv
from tqdm import trange, tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from solitaire_env import KlondikeEnv
from dqn_agent import DQNAgent


class AdaptiveEpsilon:
    """Decay epsilon when evaluation reward plateaus."""

    def __init__(self, agent, decay_factor=0.9, min_eps=0.02, patience=3):
        self.agent = agent
        self.decay_factor = decay_factor
        self.min_eps = min_eps
        self.patience = patience
        self.best_score = -float("inf")
        self.bad_epochs = 0

    def step(self, eval_score):
        if eval_score > self.best_score:
            self.best_score = eval_score
            self.bad_epochs = 0
            return

        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            new_eps = max(self.min_eps, self.agent.epsilon * self.decay_factor)
            if new_eps < self.agent.epsilon:
                tqdm.write(
                    f"Decaying ε from {self.agent.epsilon:.3f} to {new_eps:.3f} (eval plateau)"
                )
            self.agent.epsilon = new_eps
            self.bad_epochs = 0


def run_evaluation(agent, num_games):
    # create num_games parallel envs
    eval_env = SyncVectorEnv([make_env for _ in range(num_games)])

    state, _ = eval_env.reset()
    # track cumulative reward for each parallel game
    total_rewards = np.zeros(num_games, dtype=float)
    done_flags = np.zeros(num_games, dtype=bool)

    # cap to avoid infinite loops if games never terminate
    max_eval_steps = KlondikeEnv().max_moves * 5
    eval_steps = 0

    while not done_flags.all() and eval_steps < max_eval_steps:
        # always act greedily during eval
        legal_lists = [
            np.flatnonzero(eval_env.envs[i]._legal_mask()).tolist()
            for i in range(num_games)
        ]
        actions = [
            agent.select_action(state[i], legal_lists[i], eval=True)
            for i in range(num_games)
        ]
        next_state, rewards, dones, truncs, _ = eval_env.step(actions)

        # accumulate per-step rewards
        total_rewards += rewards

        state = next_state
        done_flags = np.logical_or(dones, truncs)
        eval_steps += 1

    eval_env.close()
    # average total reward per game
    return total_rewards.mean()


def make_env():
    return KlondikeEnv()


def train(total_steps=1000000, num_envs=32, checkpoint_dir="checkpoints"):
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
    metrics = {"loss": [], "train_avg_reward": [], "eval_avg_reward": []}
    steps = []

    sample_env = KlondikeEnv()
    agent = DQNAgent(
        sample_env.observation_space.shape[0],
        sample_env.action_space.n,
    )
    eps_scheduler = AdaptiveEpsilon(agent, decay_factor=0.9, min_eps=0.02, patience=3)

    ckpts = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith("dqn_") and f.endswith(".pth")
    ]
    start_step = 0
    if ckpts:
        ckpts.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        last_ckpt = ckpts[-1]
        start_step = int(last_ckpt.split("_")[1].split(".")[0])
        path = os.path.join(checkpoint_dir, last_ckpt)
        state_dict = torch.load(path)
        try:
            agent.q_net.load_state_dict(state_dict)
        except RuntimeError as e:
            print(
                f"Could not load checkpoint '{path}' due to mismatched architecture: {e}"
            )
            print("Starting training from scratch.")
            start_step = 0
        else:
            agent.target_net.load_state_dict(agent.q_net.state_dict())
            agent.total_steps = start_step
            if agent.total_steps < agent.warmup_steps:
                agent.epsilon = agent.epsilon_start
            elif agent.total_steps < agent.warmup_steps + agent.decay_steps:
                frac = (agent.total_steps - agent.warmup_steps) / agent.decay_steps
                agent.epsilon = (
                    agent.epsilon_start
                    - (agent.epsilon_start - agent.epsilon_end) * frac
                )
            else:
                agent.epsilon = agent.epsilon_end
            print(f"Loaded checkpoint '{path}' (starting from step {start_step})")

    print(
        f"Starting training: total_steps={total_steps}, num_envs={num_envs}, device={agent.device}"
    )
    print(
        "TensorBoard logging to runs/solitaire - run 'tensorboard --logdir runs/solitaire' to view"
    )
    start_time = time.time()
    env = SyncVectorEnv([make_env for _ in range(num_envs)])
    state, _ = env.reset()
    recent_reward = 0.0
    avg_r_last = 0.0

    for step in trange(
        start_step + 1,
        total_steps + 1,
        desc="Training",
        unit="step",
        leave=True,
    ):
        legal_lists = [
            np.flatnonzero(env.envs[i]._legal_mask()).tolist() for i in range(num_envs)
        ]
        actions = [
            agent.select_action(state[i], legal_lists[i]) for i in range(num_envs)
        ]
        next_state, rewards, dones, truncs, _ = env.step(actions)
        done_flags = np.logical_or(dones, truncs)
        next_legal_lists = [
            np.flatnonzero(env.envs[i]._legal_mask()).tolist() for i in range(num_envs)
        ]
        masks_next = np.zeros((num_envs, agent.action_dim), dtype=np.float32)
        for i in range(num_envs):
            masks_next[i, next_legal_lists[i]] = 1.0
            agent.store_transition(
                state[i],
                actions[i],
                rewards[i],
                next_state[i],
                done_flags[i],
                masks_next[i],
            )
        recent_reward += rewards.mean()
        loss = None
        if len(agent.replay_buffer) > agent.warmup_steps:
            loss = agent.update()
        agent.step(num_envs)
        state = next_state

        if step % log_interval == 0 and loss is not None:
            avg_r_last = recent_reward / log_interval
            tqdm.write(
                f"[Step {step}] loss={loss:.4f}  ε={agent.epsilon:.3f}  avg_reward={avg_r_last:.3f}"
            )
            recent_reward = 0.0

        if step % tb_interval == 0 and loss is not None:
            writer.add_scalar("train/loss", loss, step)
            writer.add_scalar("train/ε", agent.epsilon, step)
            writer.add_scalar("train/avg_reward", avg_r_last, step)

        if step % eval_interval == 0:
            old_eps = agent.epsilon
            agent.epsilon = 0.0
            tqdm.write(f"Running evaluation at step {step}...")
            eval_avg_reward = run_evaluation(agent, num_eval_games)
            writer.add_scalar("eval/avg_reward", eval_avg_reward, step)
            tqdm.write(f">>> Eval @ {step}: avg_reward={eval_avg_reward:.3f}")
            agent.epsilon = old_eps
            eps_scheduler.step(eval_avg_reward)

        if step % plot_interval == 0 and loss is not None:
            steps.append(step)
            metrics["loss"].append(loss)
            metrics["train_avg_reward"].append(avg_r_last)
            metrics["eval_avg_reward"].append(
                eval_avg_reward if "eval_avg_reward" in locals() else np.nan
            )
            plt.figure()
            plt.plot(steps, metrics["train_avg_reward"], label="Train Avg Reward")
            plt.plot(steps, metrics["eval_avg_reward"], label="Eval Avg Reward")
            plt.xlabel("Steps")
            plt.ylabel("Avg Reward")
            plt.legend()
            plot_path = os.path.join(plots_dir, f"avg_reward_{step}.png")
            plt.savefig(plot_path)
            tqdm.write(f"Saved plot: {plot_path}")
            plt.close()

        if step % 10000 == 0:
            torch_path = os.path.join(checkpoint_dir, f"dqn_{step}.pth")
            torch.save(agent.q_net.state_dict(), torch_path)
            elapsed = time.time() - start_time
            tqdm.write(
                f"\u2192 Step {step}: checkpoint saved   elapsed={elapsed/60:.1f}m"
            )
            tqdm.write(f"  Saved checkpoint: {torch_path}")

    total_time = (time.time() - start_time) / 3600
    print(f"Training complete in {total_time:.1f} hours")
    writer.close()


if __name__ == "__main__":
    train()
