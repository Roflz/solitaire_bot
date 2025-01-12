import os

import pyautogui
import pygetwindow as gw
import cv2
import numpy as np
from PIL import Image
from scripts import utils, evaluate_moves, game_setup

templates_dir = os.path.join("templates", "ranks")

# This one is for debugging
screenshot = Image.open("game_screenshot_no_waste.bmp")
screen_height, screen_width = screenshot.size

cards = utils.find_cards("game_screenshot_no_waste.bmp",
                         templates_dir,
                         threshold=0.95)
utils.draw_regions("game_screenshot_no_waste.bmp", screen_width, screen_height)
game_state = utils.parse_game_state(cards, screen_width, screen_height)

moves = moves.evaluate_moves(game_state)
print('done')
