import os
import torch

from solitaire_env import KlondikeEnv
from dqn_agent import DQNAgent

MODEL_PATH = os.path.join('checkpoints', 'dqn_final.pth')

num_games = 100
wins = 0
losses = 0

env = KlondikeEnv()

use_agent = False
if os.path.exists(MODEL_PATH):
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    agent.q_net.load_state_dict(torch.load(MODEL_PATH, map_location=agent.device))
    use_agent = True

for _ in range(num_games):
    state = env.reset()
    done = False
    while not done:
        if use_agent:
            action = agent.select_action(state, eval=True)
        else:
            moves = env._legal_moves()
            action = 0 if moves else env.action_space.n - 1
        state, reward, done, _ = env.step(action)
    if reward > 0:
        wins += 1
    else:
        losses += 1

print(f"Out of {num_games} games:")
print(f"Wins: {wins}")
print(f"Losses: {losses}")
