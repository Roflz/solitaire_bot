import os
from scripts import utils, evaluate_moves
from scripts.automate_moves import shuffle_waste_pile
from PIL import Image
from config import SCREEN_WIDTH, SCREEN_HEIGHT


image_path = "game_screenshot.bmp"
rank_threshold = 0.75
suit_threshold = 0.9

# This one is for real time playing
screenshot = utils.capture_window("BlueStacks App Player")

# This one is for debugging
# screenshot = Image.open("game_screenshot.bmp")

cards = utils.detect_suit_and_rank(image_path,
                                   suit_threshold=suit_threshold,
                                   rank_threshold=rank_threshold)

utils.draw_regions("game_screenshot.bmp")
game_state = utils.parse_start_state(cards)

# Visualize Game State in Console
utils.visualize_game_state(game_state['columns'],
                           game_state['foundations'],
                           game_state['waste'],
                           game_state['waste_pile']
                           )

done = False
moves_this_shuffle = 0

while not done:
    # Perform Moves
    perform_moves = evaluate_moves.evaluate_moves(game_state)
    moves_this_shuffle += len(perform_moves)
    # evaluate_moves.print_moves(perform_moves)

    # stop when all cards are out of waste and no hidden cards
    if not game_state['waste_pile'] and not game_state['waste'] and not evaluate_moves.game_state_has_hidden_cards(game_state):
        done = True
        print("No more moves, finished game.")
        break

    # Shuffle waste pile if no moves and waste pile empty
    if not perform_moves and not game_state['waste_pile']:
        if moves_this_shuffle == 0:
            done = True
            print("No more moves, finished game.")
            break
        moves_this_shuffle = 0
        shuffle_waste_pile()
        game_state['waste_pile'] = game_state['waste'][::-1]
        game_state['waste'] = []

    # Cycle Waste Cards if no moves or empty
    if not game_state['waste'] or not perform_moves:
        # Shuffle out 3 waste ranks
        shuffle_waste_pile()
        game_state['waste'].extend(game_state['waste_pile'][-3:])
        game_state['waste_pile'] = game_state['waste_pile'][:-3]

    # Capture Screen
    screenshot = utils.capture_window("BlueStacks App Player")

    # Identify Cards
    cards = utils.detect_suit_and_rank(image_path,
                                       suit_threshold=suit_threshold,
                                       rank_threshold=rank_threshold)
    # utils.draw_regions("game_screenshot.bmp")

    # Parse Game State
    game_state = utils.parse_game_state(game_state, cards)

    # Visualize Game State in Console
    # utils.visualize_game_state(game_state['columns'],
    #                            game_state['foundations'],
    #                            game_state['waste'],
    #                            game_state['waste_pile']
    #                            )
