import pyautogui
import time
from config import CLICK_POINTS, HIDDEN_CARD_HEIGHT, VISIBLE_CARD_HEIGHT
from pyautogui import linear, easeInQuad, easeOutQuad, easeInOutQuad


def perform_move(move):
    """
    Perform a move in the Solitaire game using PyAutoGUI.

    Args:
        move (dict): Dictionary describing the move. Example:
                     {"action": "move_to_foundation", "from_stack": 2}
                     {"action": "move_card", "from_stack": 2, "to_stack": 4}
    """
    if move["action"] == "move_to_foundation":
        click_x = 0
        click_y = 0

        if move["location"] == 'waste':
            click_x = CLICK_POINTS[move["location"]][0][0]
            click_y = CLICK_POINTS[move["location"]][0][1]

        elif 'column' in move["location"]:
            click_x = CLICK_POINTS[move["location"]][0][0]

            hidden_cards = move['column'].count('X')
            visible_cards = len(move['column']) - move['column'].count('X')
            click_y = CLICK_POINTS[move["location"]][0][1] + hidden_cards * HIDDEN_CARD_HEIGHT + visible_cards * VISIBLE_CARD_HEIGHT

        click_position = (click_x, click_y)

        # Move card to foundation
        pyautogui.click(click_position)
        time.sleep(0.1)

    elif move["action"] == "move_to_column":
        dragto_x = CLICK_POINTS[move["to_column"]][0][0]
        hidden_cards = move['to_column_list'].count('X')
        visible_cards = len(move['to_column_list']) - move['to_column_list'].count('X')
        dragto_y = CLICK_POINTS[move["to_column"]][0][
                       1] + hidden_cards * HIDDEN_CARD_HEIGHT + visible_cards * VISIBLE_CARD_HEIGHT

        dragto_position = (dragto_x, dragto_y)

        if move['location'] == "waste":
            pyautogui.moveTo(CLICK_POINTS['waste'][0])
            pyautogui.dragTo(dragto_position, duration=0.45, tween=easeInOutQuad)

        elif "column" in move["location"]:
            moveto_x = CLICK_POINTS[move["location"]][0][0]
            hidden_cards_from_column = move['from_column_list'].count('X')
            moveto_y = CLICK_POINTS[move["location"]][0][
                           1] + hidden_cards_from_column * HIDDEN_CARD_HEIGHT

            moveto_position = (moveto_x, moveto_y)

            pyautogui.moveTo(moveto_position)
            pyautogui.dragTo(dragto_position, duration=0.45, tween=easeInOutQuad)
        time.sleep(0.05)


def shuffle_waste_pile():
    click_position = CLICK_POINTS["waste_shuffle"][0]
    pyautogui.click(click_position)
    time.sleep(0.4)
