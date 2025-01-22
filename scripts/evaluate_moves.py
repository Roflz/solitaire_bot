from colorama import Fore, Style

from scripts import automate_moves, utils


def evaluate_moves(game_state):
    performed_moves = []

    while True:  # Outer loop to restart the evaluation process
        restart_outer_loop = False  # Flag to control restarting the outer loop

        # Check if there are hidden cards left in the tableau
        hidden_cards_exist = any(has_hidden_cards(column) for column in game_state["columns"])

        # If no hidden cards exist, prioritize waste pile moves
        if not hidden_cards_exist:
            if game_state["waste"]:
                top_waste_card = game_state["waste"][-1]
                if is_playable_on_foundation(top_waste_card, game_state["foundations"]):
                    performed_moves.append({
                        "priority": 2,
                        "action": "move_to_foundation",
                        "from_column": "waste",
                        "card": top_waste_card
                    })
                    # Perform Move
                    move = {
                        'action': "move_to_foundation",
                        'location': "waste",
                        'position': 0,
                        'column': []
                    }
                    automate_moves.perform_move(move)

                    # Update Game State
                    move_card_to_foundation_from_waste(game_state, top_waste_card)
                    utils.capture_window("BlueStacks App Player")
                    cards = utils.detect_suit_and_rank("C:\_\solitaire_bot\game_screenshot.bmp")
                    game_state = utils.parse_game_state(game_state, cards)
                    print_moves([performed_moves[-1]])

                    continue  # Restart the outer loop

                elif can_move_to_column(top_waste_card, game_state["columns"]):
                    to_column = get_column_to_move_to(top_waste_card, game_state["columns"])
                    performed_moves.append({
                        "priority": 3,
                        "action": "move_to_column",
                        "from_column": "waste",
                        "to_column": to_column,
                        "card": top_waste_card
                    })
                    # Perform Move
                    move = {
                        'action': "move_to_column",
                        'location': "waste",
                        'to_column': f"column_{to_column}",
                        'position': len(game_state['columns'][to_column]),
                        'to_column_list': game_state['columns'][to_column]
                    }
                    automate_moves.perform_move(move)

                    # Update Game State
                    move_card_to_column_from_waste(game_state, to_column)
                    utils.capture_window("BlueStacks App Player")
                    cards = utils.detect_suit_and_rank("C:\_\solitaire_bot\game_screenshot.bmp")
                    game_state = utils.parse_game_state(game_state, cards)
                    print_moves([performed_moves[-1]])

                    continue  # Restart the outer loop

        # Move ranks to foundations if possible
        while True:
            moves = []
            for column_index, column in enumerate(game_state["columns"]):
                if column:
                    top_card = column[-1]
                    if is_playable_on_foundation(top_card, game_state["foundations"]):
                        moves.append({
                            "priority": column.count('X'),
                            "action": "move_to_foundation",
                            "from_column": column_index,
                            "card": top_card
                        })

            if not moves:
                break

            best_move = max(moves, key=lambda move: move["priority"])
            move = {
                'action': best_move["action"],
                'location': f"column_{best_move['from_column']}",
                'position': len(game_state["columns"][best_move['from_column']]) - 1,
                'column': game_state["columns"][best_move['from_column']]
            }
            automate_moves.perform_move(move)

            # Update Game State
            move_card_to_foundation_from_column(game_state, best_move["from_column"], best_move["card"])
            utils.capture_window("BlueStacks App Player")
            cards = utils.detect_suit_and_rank("C:\_\solitaire_bot\game_screenshot.bmp")
            game_state = utils.parse_game_state(game_state, cards)
            performed_moves.append(best_move)
            print_moves([performed_moves[-1]])

            restart_outer_loop = True
            break  # Restart the outer loop

        if restart_outer_loop:
            continue  # Restart the outer loop from the beginning

        # Uncover hidden ranks
        while True:
            moves = []
            for column_index, column in enumerate(game_state["columns"]):
                if column and has_hidden_cards(column):
                    top_card = get_card_after_x(column)
                    if can_move_to_column(top_card, game_state["columns"]):
                        to_column = get_column_to_move_to(top_card, game_state["columns"])
                        moves.append({
                            "priority": column.count('X'),
                            "action": "move_to_column",
                            "from_column": column_index,
                            "to_column": to_column,
                            "card": top_card
                        })

            if not moves:
                break

            best_move = max(moves, key=lambda move: move["priority"])
            move = {
                'action': best_move["action"],
                'location': f"column_{best_move['from_column']}",
                'position_1': game_state["columns"][best_move['from_column']].index(best_move["card"]),
                'from_column_list': game_state["columns"][best_move['from_column']],
                'to_column': f"column_{best_move['to_column']}",
                'position_2': len(game_state["columns"][best_move['to_column']]),
                'to_column_list': game_state["columns"][best_move['to_column']]
            }
            automate_moves.perform_move(move)
            move_card_to_column_from_column(game_state, best_move["from_column"], best_move["card"])
            utils.capture_window("BlueStacks App Player")
            cards = utils.detect_suit_and_rank("C:\_\solitaire_bot\game_screenshot.bmp")
            game_state = utils.parse_game_state(game_state, cards)
            performed_moves.append(best_move)
            print_moves([performed_moves[-1]])

            restart_outer_loop = True
            break  # Restart the outer loop

        if restart_outer_loop:
            continue  # Restart the outer loop from the beginning

        # Create empty columns
        while True:
            moves = []
            for column_index, column in enumerate(game_state["columns"]):
                if column and not has_hidden_cards(column):
                    top_card = column[0]
                    if can_move_to_column(top_card, game_state["columns"]) and not is_king(top_card):
                        to_column = get_column_to_move_to(top_card, game_state["columns"])
                        moves.append({
                            "priority": column.count('X'),
                            "action": "move_to_column",
                            "from_column": column_index,
                            "to_column": to_column,
                            "card": top_card
                        })

            if not moves:
                break

            best_move = max(moves, key=lambda move: move["priority"])
            move = {
                'action': best_move["action"],
                'location': f"column_{best_move['from_column']}",
                'position_1': game_state["columns"][best_move['from_column']].index(best_move["card"]),
                'from_column_list': game_state["columns"][best_move['from_column']],
                'to_column': f"column_{best_move['to_column']}",
                'position_2': len(game_state["columns"][best_move['to_column']]),
                'to_column_list': game_state["columns"][best_move['to_column']]
            }
            automate_moves.perform_move(move)
            move_card_to_column_from_column(game_state, best_move["from_column"], best_move["card"])
            utils.capture_window("BlueStacks App Player")
            cards = utils.detect_suit_and_rank("C:\_\solitaire_bot\game_screenshot.bmp")
            game_state = utils.parse_game_state(game_state, cards)
            performed_moves.append(best_move)
            print_moves([performed_moves[-1]])

            restart_outer_loop = True
            break  # Restart the outer loop

        if restart_outer_loop:
            continue  # Restart the outer loop from the beginning

        if game_state["waste"]:
            top_waste_card = game_state["waste"][-1]
            if is_playable_on_foundation(top_waste_card, game_state["foundations"]):
                performed_moves.append({
                    "priority": 2,
                    "action": "move_to_foundation",
                    "from_column": "waste",
                    "card": top_waste_card
                })
                # Perform Move
                move = {
                    'action': "move_to_foundation",
                    'location': "waste",
                    'position': 0,
                    'column': []
                }
                automate_moves.perform_move(move)

                # Update Game State
                move_card_to_foundation_from_waste(game_state, top_waste_card)
                utils.capture_window("BlueStacks App Player")
                cards = utils.detect_suit_and_rank("C:\_\solitaire_bot\game_screenshot.bmp")
                game_state = utils.parse_game_state(game_state, cards)
                print_moves([performed_moves[-1]])

                continue  # Restart the outer loop

            elif can_move_to_column(top_waste_card, game_state["columns"]):
                to_column = get_column_to_move_to(top_waste_card, game_state["columns"])
                performed_moves.append({
                    "priority": 3,
                    "action": "move_to_column",
                    "from_column": "waste",
                    "to_column": to_column,
                    "card": top_waste_card
                })
                # Perform Move
                move = {
                    'action': "move_to_column",
                    'location': "waste",
                    'to_column': f"column_{to_column}",
                    'position': len(game_state['columns'][to_column]),
                    'to_column_list': game_state['columns'][to_column]
                }
                automate_moves.perform_move(move)

                # Update Game State
                move_card_to_column_from_waste(game_state, to_column)
                utils.capture_window("BlueStacks App Player")
                cards = utils.detect_suit_and_rank("C:\_\solitaire_bot\game_screenshot.bmp")
                game_state = utils.parse_game_state(game_state, cards)
                print_moves([performed_moves[-1]])

                continue  # Restart the outer loop

        break  # Exit the outer loop if no more moves are available

    return performed_moves


# Helper functions
def get_card_after_x(column):
    last_index = len(column) - 1 - column[::-1].index('X')
    if len(column) - 1 == last_index:
        return 'X'
    else:
        return column[last_index + 1]


def move_card_to_foundation_from_column(game_state, from_column, card):
    suit = card[1]
    game_state['foundations'][suit] = game_state['columns'][from_column].pop()[0]


def move_card_to_foundation_from_waste(game_state, card):
    suit = card[1]
    game_state['foundations'][suit] = game_state['waste'].pop()[0]


def move_card_to_column_from_waste(game_state, column_index):
    game_state['columns'][column_index].append(game_state['waste'].pop())


def move_card_to_column_from_column(game_state, column_index, top_card):
    column = game_state['columns'][column_index]
    target_column = get_column_to_move_to(top_card, game_state["columns"])
    game_state['columns'][target_column].extend(column[column.index(top_card):])
    game_state['columns'][column_index] = column[:column.index(top_card)]


def is_playable_on_foundation(card, foundations):
    # Check if the card can be played on the foundations
    # Example: card = "2H", foundations = {"H": "AH", "D": "AD", "S": "AS", "C": "AC"}
    suit = card[1]
    rank = card[0]
    rank_order = "A23456789TJQK"
    if not foundations[suit]:
        return rank_order.index(rank) == 0
    else:
        top_card = foundations[suit][0]
        return rank_order.index(rank) == rank_order.index(top_card) + 1


def has_hidden_cards(column):
    # Check if there are hidden ranks in the column
    return 'X' in column  # Example condition for hidden ranks


def game_state_has_hidden_cards(game_state):
    for column in game_state['columns']:
        if has_hidden_cards(column):
            return True
    return False


def is_king(card):
    # Check if the card is a king
    return card[0] == "K"


def get_column_to_move_to(card, columns):
    # Return the index of the best column to move the card to
    for column_index, column in enumerate(columns):
        if not column and is_king(card):
            return column_index
        if column and is_valid_column_move(card, column[-1]):
            return column_index
    return None


def can_move_to_column(card, columns):
    """
    Check if the given card can be moved to any column.

    Args:
        card (str): The card to check, e.g., "7H" (7 of Hearts).
        columns (list): The current columns, where each column is a list of ranks.

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
    if not card_rank:
        return False

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


def print_moves(moves):
    for move in moves:
        # Safely handle 'to_column'
        to_column = move.get('to_column')
        if isinstance(to_column, int):  # Increment only if it's an integer
            to_column += 1
        else:
            to_column = ''  # Default to an empty string if missing or not numeric

        # Safely handle 'from_column'
        from_column = move['from_column']
        if isinstance(from_column, int):  # Increment only if it's an integer
            from_column += 1
        elif isinstance(from_column, str):  # Handle strings directly
            from_column = f'"{from_column}"'  # Add quotes for display clarity

        # Display the move
        print(Fore.RED + f"{move['card']} " + Fore.GREEN + f"from Column" + Fore.RED + f" {from_column}, " + Fore.YELLOW
              + f"{move['action']}" + Fore.RED + f" {to_column}" + Style.RESET_ALL)
