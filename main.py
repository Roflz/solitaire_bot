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
screenshot = Image.open("game_screenshot_no_waste.bmp")
screen_height, screen_width = screenshot.size

# This one is to run digitally
start_state = game_setup.run_game_setup()

cards = utils.find_cards("game_screenshot_no_waste.bmp",
                         templates_dir,
                         threshold=0.95)
utils.draw_regions("game_screenshot_no_waste.bmp", screen_width, screen_height)
game_state = utils.parse_game_state(cards, screen_width, screen_height)

moves = moves.evaluate_moves(game_state)

for card in cards:
    print(f"Detected {card['card']} at {card['position']}")
