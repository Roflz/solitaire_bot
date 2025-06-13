# Solitaire Bot Reinforcement Learning

This project simulates Klondike Solitaire entirely in memory and can train a Deep Q-Network (DQN) to play via self‑play.

## Installation

Install dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
```

## Training

Use `train.py` to launch vectorised self‑play. By default it runs 32 parallel environments and saves model checkpoints under `checkpoints/`.

```bash
python train.py
```

## Evaluating

After training, run `main_digital.py` to play games using the latest checkpoint. If no model is present the environment falls back to simple rule-based play.

```bash
python main_digital.py
```

## Environment Notes

- Waste pile flips draw **three cards at a time**. When empty, the waste is recycled back into the pile.
- Tableau builds downward in alternating colours. Foundations build Ace to King by suit.
- Observation space is a one-hot vector describing card locations (tableau columns, foundations, waste, waste pile) plus a flag for face-up status.
- Action space contains all legal moves for the current state plus a special action that flips the waste pile.

Parallel environments allow millions of transitions per training session, letting the DQN learn an effective policy for Klondike Solitaire.
