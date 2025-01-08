import csv
import os

import pyautogui
import pygetwindow as gw
import cv2
import numpy as np
from PIL import Image

import scripts.utils
from scripts import utils, moves, game_setup

templates_dir = os.path.join("templates", "cards_higher_res")

# This one is for real time playing
# screenshot = utils.capture_window_2("BlueStacks App Player")
# screen_height, screen_width = screenshot.shape[:-1]

# This one is for debugging
with open('edge_testing.csv', mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Threshold", "lower_threshold", "upper_threshold", "cards_found"])  # Header row

    edge_lower_thresholds = [10 * i for i in range(1, 11)]
    edge_upper_thresholds = [10 * i + 100 for i in range(1, 11)]
    thresholds = [0.05 * i for i in range(1, 20)]
    for i in thresholds:
        for j in edge_upper_thresholds:
            for k in edge_lower_thresholds:
                screenshot = Image.open("game_screenshot_no_waste.bmp")
                screen_height, screen_width = screenshot.size

                cards = utils.find_cards_edge_matching("game_screenshot_no_waste.bmp",
                                                       templates_dir,
                                                       threshold=i,
                                                       edge_thresholds=(k, j))
                print(f"Threshold: {i}")
                print(f"Edge Upper Threshold: {j}")
                print(f"Edge Lower Threshold: {k}")
                print(f"Cards found: {len(cards)}")
                csv_writer.writerow([i, k, j, len(cards)])
# cards = utils.find_cards("game_screenshot.bmp",
#                                        templates_dir,
#                                        threshold=0.97)


utils.draw_regions("game_screenshot_no_waste.bmp", screen_width, screen_height)
game_state = utils.parse_start_state(cards, screen_width, screen_height)

# Visualize Game State in Console
utils.visualize_game_state(game_state['columns'],
                           game_state['foundations'],
                           game_state['waste'],
                           game_state['waste_pile']
                           )

done = False

while not done:
    # Perform Moves
    perform_moves = moves.evaluate_moves(game_state)
    scripts.utils.print_moves(perform_moves)

    # Shuffle waste pile if no moves and waste pile empty
    if not perform_moves and not game_state['waste_pile']:
        print('Shuffle Waste Pile')

    # Cycle Waste Cards if no moves or empty
    if not game_state['waste'] or not perform_moves:
        # Shuffle out 3 waste cards
        print('Cycle Waste Pile')
        game_state['waste'].extend(game_state['waste_pile'][-3:])
        game_state['waste_pile'] = game_state['waste_pile'][:-3]

    # Capture Screen
    screenshot = utils.capture_window_2("BlueStacks App Player")
    screen_height, screen_width = screenshot.shape[:-1]

    # Identify Cards
    cards = utils.find_cards("game_screenshot.bmp",
                             templates_dir,
                             threshold=0.95)
    utils.draw_regions("game_screenshot.bmp", screen_width, screen_height)

    # Parse Game State
    game_state = utils.parse_game_state(game_state, cards, screen_width, screen_height)

    # Visualize Game State in Console
    utils.visualize_game_state(game_state['columns'],
                               game_state['foundations'],
                               game_state['waste'],
                               game_state['waste_pile']
                               )
