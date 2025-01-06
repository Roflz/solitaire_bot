def evaluate_moves(game_state):
    """
    Evaluate possible moves based on the game state.

    Args:
        game_state (dict): The current game state.

    Returns:
        list: A list of possible moves sorted by priority.
    """
    moves = []

    # Move cards to foundations if possible
    for column_index, column in enumerate(game_state["columns"]):
        if column:
            top_card = column[-1]
            if is_playable_on_foundation(top_card, game_state["foundations"]):
                moves.append({
                    "priority": 1,  # High priority
                    "action": "move_to_foundation",
                    "from_column": column_index,
                    "card": top_card
                })
                # Perform Move
                # Update Game State
                suit = top_card[1]
                rank = top_card[0]
                game_state['foundations'][suit] = rank

    # Play waste pile card if possible
    if game_state["waste"]:
        top_waste_card = game_state["waste"][-1]
        if is_playable_on_foundation(top_waste_card, game_state["foundations"]):
            moves.append({
                "priority": 2,
                "action": "move_to_foundation",
                "from_column": "waste",
                "card": top_waste_card
            })
        elif can_move_to_column(top_waste_card, game_state["columns"]):
            moves.append({
                "priority": 3,
                "action": "move_to_column",
                "from_column": "waste",
                "to_column": get_column_to_move_to(top_waste_card, game_state["columns"]),
                "card": top_waste_card
            })

    # Uncover hidden cards
    for column_index, column in enumerate(game_state["columns"]):
        if column and has_hidden_cards(column):
            top_card = column[-1]
            if can_move_to_another_column(top_card, game_state["columns"]):
                moves.append({
                    "priority": 4,
                    "action": "move_to_column",
                    "from_column": column_index,
                    "to_column": get_column_to_move_to(top_card, game_state["columns"]),
                    "card": top_card
                })

    # Create empty columns
    for column_index, column in enumerate(game_state["columns"]):
        if not column:  # Empty column
            for other_column_index, other_column in enumerate(game_state["columns"]):
                if other_column and is_king(other_column[-1]):
                    moves.append({
                        "priority": 5,
                        "action": "move_to_column",
                        "from_column": other_column_index,
                        "to_column": column_index,
                        "card": other_column[-1]
                    })

    # Sort moves by priority (lower numbers = higher priority)
    moves.sort(key=lambda move: move["priority"])
    return moves


# Helper functions
def is_playable_on_foundation(card, foundations):
    # Check if the card can be played on the foundations
    # Example: card = "2H", foundations = {"H": "AH", "D": "AD", "S": "AS", "C": "AC"}
    suit = card[1]
    rank = card[0]
    rank_order = "A23456789TJQK"
    if not foundations[suit]:
        return rank_order.index(rank) == 0
    else:
        return rank_order.index(rank) == rank_order.index(foundations[suit][0]) + 1


def can_move_to_column(card, columns):
    # Check if the card can be moved to any column
    for column in columns:
        if column and is_valid_column_move(card, column[-1]):
            return True
    return False


def has_hidden_cards(column):
    # Check if there are hidden cards in the column
    return len(column) > 1 and isinstance(column[-2], str)  # Example condition for hidden cards


def is_king(card):
    # Check if the card is a king
    return card[0] == "K"


def get_column_to_move_to(card, columns):
    # Return the index of the best column to move the card to
    for column_index, column in enumerate(columns):
        if column and is_valid_column_move(card, column[-1]):
            return column_index
    return None


def can_move_to_another_column(card, columns):
    """
    Check if the given card can be moved to any column.

    Args:
        card (str): The card to check, e.g., "7H" (7 of Hearts).
        columns (list): The current columns, where each column is a list of cards.

    Returns:
        bool: True if the card can be moved to another column, False otherwise.
    """
    for column in columns:
        if not column:  # Empty column
            if card[0] == "K":  # Only a King can be moved to an empty column
                return True
        else:
            top_card = column[-1]
            if is_valid_column_move(card, top_card):
                return True
    return False


def is_valid_column_move(card, target_card):
    """
    Check if a card can be legally moved onto another card in a column.

    Args:
        card (str): The card to move, e.g., "7H" (7 of Hearts).
        target_card (str): The card at the top of the destination column, e.g., "8C" (8 of Clubs).

    Returns:
        bool: True if the move is valid, False otherwise.
    """
    card_rank, card_suit = card[:-1], card[-1]
    target_rank, target_suit = target_card[:-1], target_card[-1]

    # Ensure opposite colors (red vs. black)
    red_suits = {"H", "D"}
    black_suits = {"C", "S"}
    if (card_suit in red_suits and target_suit in red_suits) or \
       (card_suit in black_suits and target_suit in black_suits):
        return False

    # Ensure the card is one rank lower
    rank_order = "A23456789TJQK"
    if rank_order.index(card_rank) + 1 == rank_order.index(target_rank):
        return True

    return False