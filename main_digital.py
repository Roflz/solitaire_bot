import os

import pyautogui
import pygetwindow as gw
import cv2
import numpy as np
from PIL import Image

import scripts.utils
from scripts import utils, moves, game_setup
import faulthandler


faulthandler.enable()

number_of_games = 100
wins = 0
lost = 0

for game in range(0, number_of_games):
    start_state = game_setup.run_game_setup()

    columns = [X for X in start_state['tableau'].values()]
    columns_with_cards_hidden = [[*(['X'] * (len(sublist) - 1)), sublist[-1]] if sublist else [] for sublist in columns]

    game_state = {
        'columns': columns_with_cards_hidden,
        'foundations': {'H': [], 'S': [], 'D': [], 'C': []},
        'waste': start_state['waste'],
        'waste_pile': start_state['remaining_deck']
    }

    perform_moves = []
    done = False
    reshuffles = 0
    moves_performed = 0
    moves_since_reshuffle = 0

    while not done:
        perform_moves = moves.evaluate_moves(game_state)
        scripts.utils.print_moves(perform_moves)

        # Shuffle waste pile if no moves and waste pile empty
        if not perform_moves and not game_state['waste_pile']:
            game_state['waste_pile'], game_state['waste'] = game_state['waste'], []
            reshuffles += 1
            moves_since_reshuffle = 0

        # Cycle Waste Cards if no moves or empty
        if not game_state['waste'] or not perform_moves:
            # Shuffle out 3 waste cards
            game_state['waste'].extend(game_state['waste_pile'][-3:])  # Add last 3 items to 'waste'
            game_state['waste_pile'] = game_state['waste_pile'][:-3]   # Remove the last 3 items from 'waste_pile'

        # turn upside down cards upright
        for j, column in enumerate(game_state['columns']):
            if column and column[-1] == 'X':  # Ensure the column is not empty
                column[-1] = columns[j][len(column) - 1]

        # Visualize current game state
        utils.visualize_game_state(game_state['columns'],
                                   game_state['foundations'],
                                   game_state['waste'],
                                   game_state['waste_pile'],
                                   moves_performed,
                                   reshuffles
                                   )
        if (all(not sublist for sublist in game_state['columns'])
                and all(value == 'K' for value in game_state['foundations'].values())
                and not game_state['waste']
                and not game_state['waste_pile']):
            done = True
            wins += 1

        elif not moves_since_reshuffle and not game_state['waste_pile']:
            done = True
            lost += 1

print(f"Out of {number_of_games} games:")
print(f"Wins: {wins}")
print(f"Losses: {lost}")