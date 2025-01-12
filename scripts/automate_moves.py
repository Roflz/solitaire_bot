import pyautogui
import time

# Example screen coordinates (adjust based on your game setup)
X_OFFSET = 1920 # offset x for using the right screen
STACK_BASE_X = 100  # Starting X-coordinate for the first stack
STACK_BASE_Y = 500  # Y-coordinate for all stacks
STACK_SPACING = 150  # Horizontal spacing between stacks
FOUNDATION_BASE_X = 800  # Starting X-coordinate for the foundations
FOUNDATION_BASE_Y = 100  # Y-coordinate for all foundations
FOUNDATION_SPACING = 150  # Horizontal spacing between foundations


def perform_move(move):
    """
    Perform a move in the Solitaire game using PyAutoGUI.

    Args:
        move (dict): Dictionary describing the move. Example:
                     {"action": "move_to_foundation", "from_stack": 2}
                     {"action": "move_card", "from_stack": 2, "to_stack": 4}
    """
    if move["action"] == "move_to_foundation":
        stack_x, stack_y = get_stack_position(move["from_stack"])
        foundation_x, foundation_y = get_foundation_position()

        # Move card to foundation
        pyautogui.leftClick()

    elif move["action"] == "move_card":
        from_x, from_y = get_stack_position(move["from_stack"])
        to_x, to_y = get_stack_position(move["to_stack"])

        # Drag card from one stack to another
        pyautogui.moveTo(from_x, from_y, duration=0.2)
        pyautogui.dragTo(to_x, to_y, duration=0.2)

    elif move["action"] == "flip_card":
        stack_x, stack_y = get_stack_position(move["stack"])

        # Simulate a click to flip the card
        pyautogui.click(stack_x, stack_y)

    time.sleep(0.5)  # Add a delay between moves for stability


def get_stack_position(stack_index, cards_in_stack):
    """
    Calculate the screen position of a stack based on its index.

    Args:
        stack_index (int): Index of the stack (0-based).

    Returns:
        (int, int): X, Y coordinates of the stack.
    """
    return STACK_BASE_X + stack_index * STACK_SPACING, STACK_BASE_Y


def get_foundation_position(foundation_index=0):
    """
    Calculate the screen position of a foundation based on its index.

    Args:
        foundation_index (int): Index of the foundation (0-based).

    Returns:
        (int, int): X, Y coordinates of the foundation.
    """
    return FOUNDATION_BASE_X + foundation_index * FOUNDATION_SPACING, FOUNDATION_BASE_Y


perform_move()