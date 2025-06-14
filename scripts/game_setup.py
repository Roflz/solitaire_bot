import random


# Initialize a standard deck of ranks in the rank-suit format
def create_deck():
    suits = ['H', 'D', 'C', 'S']  # Hearts, Diamonds, Clubs, Spades
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    return [f"{rank}{suit}" for suit in suits for rank in ranks]


# Shuffle the deck
def shuffle_deck(deck):
    random.shuffle(deck)
    return deck


# Deal ranks into the tableau
def deal_tableau(deck):
    tableau = {}
    face_down_cards = {}
    face_up_cards = {}

    for i in range(1, 8):  # 7 tableau columns
        face_down_cards[f"Column {i}"] = deck[:i - 1]  # All but the last card are face down
        face_up_cards[f"Column {i}"] = [deck[i - 1]]   # The last card is face up
        tableau[f"Column {i}"] = deck[:i]             # Full column
        del deck[:i]

    return tableau, face_down_cards, face_up_cards, deck


# Initialize the game state
def initialize_solitaire(seed=None):
    if seed is not None:
        random.seed(seed)
    deck = create_deck()
    shuffled_deck = shuffle_deck(deck)

    # Deal the tableau
    tableau, face_down_cards, face_up_cards, remaining_deck = deal_tableau(shuffled_deck)

    # Initialize foundation and waste piles
    foundation = {"Hearts": [], "Diamonds": [], "Clubs": [], "Spades": []}
    waste = []

    # Game state dictionary
    game_state = {
        "tableau": tableau,
        "face_down_cards": face_down_cards,
        "face_up_cards": face_up_cards,
        "foundation": foundation,
        "waste": waste,
        "remaining_deck": remaining_deck
    }

    return game_state


# Print the game state
def print_game_state(game_state):
    print("Tableau:")
    for col, cards in game_state["tableau"].items():
        face_down = len(game_state["face_down_cards"][col])
        face_up = " ".join(game_state["face_up_cards"][col])
        print(f"{col}: {'X ' * face_down}{face_up}")

    print("\nFoundation:")
    for suit, cards in game_state["foundation"].items():
        print(f"{suit}: {' '.join(cards) if cards else 'Empty'}")

    print("\nWaste:")
    print("Empty" if not game_state["waste"] else " ".join(game_state["waste"]))

    print("\nRemaining deck:")
    print(" ".join(game_state["remaining_deck"]))


def run_game_setup():
    game_state = initialize_solitaire()
    print_game_state(game_state)
    return game_state
