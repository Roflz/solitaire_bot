import os

import pyautogui
import pygetwindow as gw
import cv2
import numpy as np
from PIL import Image
from scripts import utils, moves, game_setup

templates_dir = os.path.join("templates", "cards")

# This one is for real time playing
# screenshot = utils.capture_window_2("BlueStacks App Player")
# screen_height, screen_width = screenshot.shape[:-1]

# This one is for debugging
# screenshot = Image.open("game_screenshot_no_waste.bmp")
# screen_height, screen_width = screenshot.size

# This one is to run digitally
start_state = game_setup.run_game_setup()

# cards = utils.find_cards("game_screenshot_no_waste.bmp",
#                          templates_dir,
#                          threshold=0.95)
# utils.draw_regions("game_screenshot_no_waste.bmp", screen_width, screen_height)
# game_state = utils.parse_game_state(cards, screen_width, screen_height)

columns = [X for X in start_state['tableau'].values()]
columns_with_cards_hidden = [[*(['X'] * (len(sublist) - 1)), sublist[-1]] if sublist else [] for sublist in columns]

game_state = {
    'columns': columns_with_cards_hidden,
    'foundations': {'H': [], 'S': [], 'D': [], 'C': []},
    'waste': start_state['waste'],
    'waste_pile': start_state['remaining_deck']
}

for i in range(20):
    # Cycle Waste Cards if no moves or empty
    if not game_state['waste'] or not perform_moves:
        # Shuffle out 3 waste cards
        game_state['waste'].extend(game_state['waste_pile'][-3:])  # Add last 3 items to 'waste'
        game_state['waste_pile'] = game_state['waste_pile'][:-3]   # Remove the last 3 items from 'waste_pile'

    # turn upside down cards upright
    for j, column in enumerate(game_state['columns']):
        if column and column[-1] == 'X':  # Ensure the column is not empty
            column[-1] = columns[j][len(column) - 1]
    utils.visualize_game_state(game_state['columns'], game_state['foundations'], game_state['waste'], game_state['waste_pile'])

    perform_moves = moves.evaluate_moves(game_state)
    for move in perform_moves:
        print(move)

    # Shuffle waste pile if no moves and waste pile empty
    if not perform_moves and not game_state['waste_pile']:
        game_state['waste_pile'], game_state['waste'] = game_state['waste'], []