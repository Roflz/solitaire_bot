import numpy as np
import gym
from gym import spaces

from scripts import game_setup


# `is_valid_column_move` from `scripts.evaluate_moves` is reproduced here to
# avoid importing the entire module (which requires GUI dependencies).
def is_valid_column_move(card, target_card):
    """Return True if ``card`` can be placed onto ``target_card`` in a column."""
    card_rank, card_suit = card[:-1], card[-1]
    target_rank, target_suit = target_card[:-1], target_card[-1]
    if not card_rank:
        return False

    red_suits = {"H", "D"}
    black_suits = {"C", "S"}
    if (card_suit in red_suits and target_suit in red_suits) or (
        card_suit in black_suits and target_suit in black_suits
    ):
        return False

    rank_order = "A23456789TJQK"
    return rank_order.index(card_rank) + 1 == rank_order.index(target_rank)


# Simple Klondike environment using 3-card waste cycles.
# Tableau columns contain full card names, with hidden cards
# represented by their position in face_down_counts.

RANK_ORDER = "A23456789TJQK"
SUITS = ["H", "D", "C", "S"]
CARD_LIST = [f"{r}{s}" for s in SUITS for r in RANK_ORDER]
CARD_INDEX = {c: i for i, c in enumerate(CARD_LIST)}


class KlondikeEnv(gym.Env):
    """Gym-style environment wrapping the digital solitaire logic."""

    def __init__(self):
        super().__init__()
        self.num_locations = 13  # 7 columns + 4 foundations + waste + waste pile
        # Fixed action mapping
        # 0: flip waste
        # 1-7: tableau i -> foundation
        # 8-14: waste -> tableau i
        # 15: waste -> foundation
        # 16-43: foundation suit -> tableau j (suit*7 + j)
        # 44+: tableau i -> tableau j starting at index k
        #     encoded as 44 + ((i*7 + j) * MAX_SEQ_START) + k
        self.MAX_SEQ_START = 20
        self.action_space = spaces.Discrete(44 + 49 * self.MAX_SEQ_START)
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(52 * (self.num_locations + 1),), dtype=np.float32
        )
        self.state = None
        self.face_down_counts = None
        self.no_move_since_recycle = False
        # Rough upper bound on the number of moves in a single game.
        # Used by training utilities to cap evaluation length.
        self.max_moves = 200

    # --- Helpers ----------------------------------------------------------
    def _can_play_foundation(self, card):
        suit = card[1]
        pile = self.state["foundations"][suit]
        if not pile:
            return card[0] == "A"
        # Compare only the rank characters. ``pile[-1]`` stores full card names
        # such as "5H", so we slice the rank before looking it up in
        # ``RANK_ORDER``.
        return RANK_ORDER.index(card[0]) == RANK_ORDER.index(pile[-1][0]) + 1

    def _can_move_to_column(self, card, column):
        if not column:
            return card[0] == "K"
        top = column[-1]
        return is_valid_column_move(card, top)

    def _can_move_sequence(self, sequence, dest_column):
        if not sequence:
            return False
        first = sequence[0]
        if not self._can_move_to_column(first, dest_column):
            return False
        # check internal ordering
        for i in range(len(sequence) - 1):
            if not is_valid_column_move(sequence[i + 1], sequence[i]):
                return False
        return True

    def _flip_waste(self):
        if not self.state["waste_pile"]:
            self.state["waste_pile"] = self.state["waste"][::-1]
            self.state["waste"] = []
        draw = min(3, len(self.state["waste_pile"]))
        for _ in range(draw):
            self.state["waste"].append(self.state["waste_pile"].pop())

    def _maybe_reveal(self, col):
        if (
            len(self.state["columns"][col]) == self.face_down_counts[col]
            and self.face_down_counts[col] > 0
        ):
            self.face_down_counts[col] -= 1

    def _apply_move(self, move):
        kind = move[0]
        if kind == "t2f":
            col = move[1]
            card = self.state["columns"][col].pop()
            self.state["foundations"][card[1]].append(card)
            self._maybe_reveal(col)
        elif kind == "w2f":
            card = self.state["waste"].pop()
            self.state["foundations"][card[1]].append(card)
        elif kind == "w2t":
            col = move[1]
            self.state["columns"][col].append(self.state["waste"].pop())
        elif kind == "f2t":
            suit, col = move[1], move[2]
            card = self.state["foundations"][suit].pop()
            self.state["columns"][col].append(card)
        elif kind == "t2t":
            from_c, start, to_c = move[1], move[2], move[3]
            seq = self.state["columns"][from_c][start:]
            self.state["columns"][to_c].extend(seq)
            self.state["columns"][from_c] = self.state["columns"][from_c][:start]
            self._maybe_reveal(from_c)

    def _legal_mask(self):
        """Return a boolean mask of legal actions for the fixed mapping."""
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        mask[0] = 1.0  # flipping is always allowed

        # tableau -> foundation
        for i, column in enumerate(self.state["columns"]):
            if len(column) > self.face_down_counts[i]:
                card = column[-1]
                if self._can_play_foundation(card):
                    mask[1 + i] = 1.0

        # waste -> foundation / tableau
        if self.state["waste"]:
            card = self.state["waste"][-1]
            if self._can_play_foundation(card):
                mask[15] = 1.0
            for j in range(7):
                if self._can_move_to_column(card, self.state["columns"][j]):
                    mask[8 + j] = 1.0

        # foundation -> tableau
        for s, suit in enumerate(SUITS):
            pile = self.state["foundations"][suit]
            if pile:
                card = pile[-1]
                for j in range(7):
                    if self._can_move_to_column(card, self.state["columns"][j]):
                        mask[16 + s * 7 + j] = 1.0

        # tableau -> tableau (all visible sequences up to MAX_SEQ_START)
        for i, column in enumerate(self.state["columns"]):
            max_start = min(len(column), self.MAX_SEQ_START)
            for start in range(self.face_down_counts[i], max_start):
                seq = column[start:]
                for j in range(7):
                    if i == j:
                        continue
                    if self._can_move_sequence(seq, self.state["columns"][j]):
                        idx = 44 + (i * 7 + j) * self.MAX_SEQ_START + start
                        mask[idx] = 1.0

        return mask

    def _check_done(self):
        return all(len(pile) == 13 for pile in self.state["foundations"].values())

    # --- Gym API ----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        gs = game_setup.initialize_solitaire(seed)
        columns = []
        face_down = []
        for idx in range(1, 8):
            key = f"Column {idx}"
            columns.append(gs["tableau"][key])
            face_down.append(len(gs["face_down_cards"][key]))
        self.state = {
            "columns": columns,
            "foundations": {"H": [], "D": [], "C": [], "S": []},
            "waste": [],
            "waste_pile": gs["remaining_deck"],
        }
        self.face_down_counts = face_down
        self._flip_waste()
        return self._get_obs(), {}

    def _decode_move(self, action):
        if 1 <= action <= 7:
            return ("t2f", action - 1)
        if 8 <= action <= 14:
            return ("w2t", action - 8)
        if action == 15:
            return ("w2f", None)
        if 16 <= action < 44:
            idx = action - 16
            s, j = divmod(idx, 7)
            return ("f2t", SUITS[s], j)
        if action >= 44:
            idx = action - 44
            pair, start = divmod(idx, self.MAX_SEQ_START)
            i, j = divmod(pair, 7)
            return ("t2t", i, start, j)
        return None

    def _is_legal(self, action):
        mask = self._legal_mask()
        return mask[action] > 0

    def _any_legal_move(self):
        mask = self._legal_mask()
        return mask[1:].any()

    def step(self, action):
        shaped_reward = 0.0
        done = False

        if action != 0 and self._is_legal(action):
            move = self._decode_move(action)
            pre_face_down = list(self.face_down_counts)
            if move[0] in ("t2f", "w2f"):
                shaped_reward += 1.0
            if move[0] in ("w2t", "w2f"):
                shaped_reward += 0.2
            self._apply_move(move)
            if move[0] in ("t2f", "t2t"):
                col = move[1]
                if pre_face_down[col] > self.face_down_counts[col]:
                    shaped_reward += 0.2
            self.no_move_since_recycle = False
        else:
            recycling = len(self.state["waste_pile"]) == 0
            self._flip_waste()
            if recycling:
                shaped_reward -= 0.2
                if self.no_move_since_recycle:
                    done = True
                self.no_move_since_recycle = not self._any_legal_move()
            else:
                self.no_move_since_recycle = (
                    self.no_move_since_recycle and not self._any_legal_move()
                )

        win = self._check_done()
        done = done or win
        reward = shaped_reward + (50.0 if win else 0.0)
        return self._get_obs(), reward, done, False, {}

    # Observation encoding
    def _get_obs(self):
        vec = np.zeros((52, self.num_locations + 1), dtype=np.float32)
        for card in CARD_LIST:
            idx = CARD_INDEX[card]
            loc, face_up = self._locate(card)
            vec[idx][loc] = 1.0
            vec[idx][-1] = 1.0 if face_up else 0.0
        return vec.flatten()

    def _locate(self, card):
        # columns
        for i, column in enumerate(self.state["columns"]):
            if card in column:
                idx = column.index(card)
                face_up = idx >= self.face_down_counts[i]
                return i, face_up
        # foundations
        for i, suit in enumerate(SUITS):
            if card in self.state["foundations"][suit]:
                return 7 + i, True
        # waste
        if card in self.state["waste"]:
            return 11, True
        # waste pile
        return 12, False

    def render(self):
        lines = []
        f_str = " ".join(f"{s}:{''.join(self.state['foundations'][s]) or '-'}" for s in SUITS)
        lines.append(f"Foundations: {f_str}")
        lines.append(
            f"Waste: {' '.join(self.state['waste']) or '-'} | Pile: {len(self.state['waste_pile'])}"
        )
        for i, col in enumerate(self.state['columns']):
            hidden = self.face_down_counts[i]
            visible = col[hidden:]
            lines.append(f"C{i}: {'X'*hidden} {' '.join(visible)}")
        return "\n".join(lines)
