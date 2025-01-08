import pygetwindow as gw
import mss
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import win32gui
import win32ui
import win32con
from colorama import Fore, Style


def find_window(window_title):
    # Find the game's window
    game_window = None
    for window in gw.getAllTitles():
        if window_title in window:  # Replace "Your Game Title" with the title of your game
            game_window = gw.getWindowsWithTitle(window)[0]
            break

    if game_window:
        # Get the game's window position and size
        x, y = game_window.left, game_window.top
        width, height = game_window.width, game_window.height

        return x, y, width, height
    else:
        print("Game window not found.")


def capture_window(x, y, width, height, z):
    with mss.mss() as sct:
        monitor = {"top": y, "left": x, "width": width, "height": height}
        screenshot = sct.grab(monitor)

        # Save screenshot as a file
        img = mss.tools.to_png(screenshot.rgb, screenshot.size, output=f"game_screenshot_{z}.png")
        print("Screenshot saved!")
        return screenshot


def capture_window_2(window_name):
    # Find the window handle
    hwnd = win32gui.FindWindow(None, window_name)  # Replace with the game's window title
    if not hwnd:
        print("Window not found!")
        return None

    # Get the window dimensions
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top

    # Capture the window
    hdesktop = win32gui.GetDesktopWindow()
    hwndDC = win32gui.GetWindowDC(hdesktop)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(bitmap)

    saveDC.BitBlt((0, 0), (width, height), mfcDC, (left, top), win32con.SRCCOPY)
    bitmap.SaveBitmapFile(saveDC, f"game_screenshot.bmp")

    # Convert to an image
    img = Image.open("game_screenshot.bmp")
    return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)


def find_best_scale(template, screenshot, scales, threshold=0.8):
    """
    Dynamically adjust the template scale to match cards in the screenshot.

    Args:
        template (numpy.ndarray): The template image.
        screenshot (numpy.ndarray): The screenshot image.
        scales (list): List of scaling factors to test.
        threshold (float): Matching confidence threshold.

    Returns:
        float: Best scale value.
        numpy.ndarray: Best resized template.
    """
    best_scale = None
    best_resized_template = None
    best_score = 0

    for scale in scales:
        resized_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(screenshot, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = max_val
            best_scale = scale
            best_resized_template = resized_template

    return best_scale, best_resized_template


def find_cards(screenshot_path, templates_dir, threshold=0.8):
    """
    Detects all cards in a game screenshot using template matching.

    Args:
        screenshot_path (str): Path to the screenshot of the game.
        templates_dir (str): Directory containing card templates.
        threshold (float): Match threshold (default is 0.8).

    Returns:
        List[Dict]: List of detected cards with their positions and names.
    """
    # Load the screenshot
    screenshot = cv2.imread(screenshot_path, cv2.IMREAD_COLOR)
    gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    detected_cards = []

    # # Loop through each card template
    for template_name in os.listdir(templates_dir):
        template_path = os.path.join(templates_dir, template_name)
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)

        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Perform template matching
        result = cv2.matchTemplate(gray_screenshot, gray_template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)

        # Store matches with their positions
        card = template_name.replace(".png", "")  # Remove extension for card name
        card = (card[-1] + card[0]).upper()

        if locations[0].size != 0 and locations[1].size != 0:
            x, y = np.int64(locations[1][0]), np.int64(locations[0][0])  # Switch x, y for OpenCV format
            point = (x, y)  # Top-left corner of the matched card
            detected_cards.append({
                "card": card,
                "position": point,
            })

            # Draw rectangle on screenshot for debugging
            h, w = gray_template.shape
            cv2.rectangle(screenshot, point, (point[0] + w, point[1] + h), (0, 255, 0), 2)

    # Save the screenshot with rectangles for debugging
    cv2.imwrite("debug_detected_cards.png", screenshot)

    return detected_cards


def find_cards_edge_matching(screenshot_path, templates_dir, threshold=0.8, edge_thresholds=(50, 150)):
    """
    Detects all cards in a game screenshot using template matching with edge detection.

    Args:
        screenshot_path (str): Path to the screenshot of the game.
        templates_dir (str): Directory containing card templates.
        threshold (float): Match threshold (default is 0.8).
        edge_thresholds (tuple): Lower and upper thresholds for Canny edge detection.

    Returns:
        List[Dict]: List of detected cards with their positions and names.
    """
    # Load the screenshot and convert to grayscale
    screenshot = cv2.imread(screenshot_path, cv2.IMREAD_COLOR)
    gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # Apply edge detection to the screenshot
    edge_screenshot = cv2.Canny(gray_screenshot, edge_thresholds[0], edge_thresholds[1])

    detected_cards = []

    # Loop through each card template
    for template_name in os.listdir(templates_dir):
        template_path = os.path.join(templates_dir, template_name)
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)

        # Convert template to grayscale and apply edge detection
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        edge_template = cv2.Canny(gray_template, edge_thresholds[0], edge_thresholds[1])

        # Perform template matching on edge-detected images
        result = cv2.matchTemplate(edge_screenshot, edge_template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)

        # Store matches with their positions
        card = template_name.replace(".png", "").replace(".jpg", "")  # Remove file extension for card name
        card = (card[-1] + card[0]).upper()  # Adjust for card naming convention

        if locations[0].size != 0 and locations[1].size != 0:
            x, y = np.int64(locations[1][0]), np.int64(locations[0][0])  # x, y for OpenCV format
            point = (x, y)  # Top-left corner of the matched card
            detected_cards.append({
                "card": card,
                "position": point,
            })

            # Draw rectangle on screenshot for debugging
            h, w = gray_template.shape
            cv2.rectangle(screenshot, point, (point[0] + w, point[1] + h), (0, 255, 0), 2)

    # Save the screenshot with rectangles for debugging
    cv2.imwrite("debug_detected_cards_edges.png", screenshot)

    return detected_cards

def draw_regions(screenshot_path, screen_width, screen_height):
    # Load the screenshot
    screenshot = cv2.imread(screenshot_path)

    # Account for toolbars on the sides of the app
    right_toolbar_width = 33
    top_toolbar_width = 33

    # Define number of columns and calculate column width
    num_columns = 7
    screen_start_x = 656
    screen_end_x = 1264
    column_width = (screen_end_x - screen_start_x) // num_columns

    # Draw column regions
    for i in range(num_columns):
        x1 = i * column_width + screen_start_x
        x2 = x1 + column_width
        y1, y2 = 441, screen_height  # Columns start below waste and foundation regions
        cv2.rectangle(screenshot, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(screenshot, f"Column {i + 1}", (x1 - 10, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Draw foundation region
    foundation_region = (screen_start_x, 235, 1002, 353)
    x1, y1, x2, y2 = foundation_region
    cv2.rectangle(screenshot, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(screenshot, "Foundations", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw Waste region
    waste_region = (1022, 235, 1164, 351)
    x1, y1, x2, y2 = waste_region
    cv2.rectangle(screenshot, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(screenshot, "Waste", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Save the annotated image
    cv2.imwrite("annotated_regions.png", screenshot)
    print("Annotated image saved as 'annotated_regions.png'.")


def parse_start_state(detected_cards, screen_width, screen_height):
    """
    Parse the detected cards into a structured game state.

    Args:
        detected_cards (list): List of detected cards with positions.
        screen_width (int): Width of the gameplay screen.
        screen_height (int): Height of the gameplay screen.

    Returns:
        dict: Parsed game state.
    """
    # Initialize game state
    num_columns = 7
    screen_start_x = 656
    screen_end_x = 1264
    column_width = (screen_end_x - screen_start_x) // num_columns
    game_state = {
        "columns": [[(0, 'X')] * i for i in range(num_columns)],
        "foundations": {"H": [], "S": [], "D": [], "C": []},
        "waste": [],
        "waste_pile": ['X'] * 24
    }

    # Define regions for foundations and waste pile
    foundation_region = (screen_start_x, 235, 1002, 353)
    waste_region = (1022, 235, 1164, 351)

    # Parse each detected card
    for card in detected_cards:
        x, y = card["position"]
        card_name = card["card"]

        # Check if the card is in the foundation region
        if foundation_region[0] <= x <= foundation_region[2] and foundation_region[1] <= y <= foundation_region[3]:
            suit = card_name[1]  # Extract suit
            game_state["foundations"][suit] = card_name[0]  # Assign card value to foundation suit

        # Check if the card is in the waste region
        elif waste_region[0] <= x <= waste_region[2] and waste_region[1] <= y <= waste_region[3]:
            game_state["waste"].append(card_name)

        # Otherwise, assign the card to the nearest column
        else:
            column_index = (x - screen_start_x) // column_width
            game_state["columns"][column_index].append((y, card_name))

    # Sort each column by the y-coordinate
    for column in game_state["columns"]:
        column.sort()  # Sort by y-coordinate
        for i in range(len(column)):
            column[i] = column[i][1]  # Extract card name only

    return game_state


def parse_game_state(game_state, detected_cards, screen_width, screen_height):
    """
    Parse the detected cards into a structured game state.

    Args:
        detected_cards (list): List of detected cards with positions.
        screen_width (int): Width of the gameplay screen.
        screen_height (int): Height of the gameplay screen.

    Returns:
        dict: Parsed game state.
    """
    # Initialize game state
    num_columns = 7
    screen_start_x = 656
    screen_end_x = 1264
    column_width = (screen_end_x - screen_start_x) // num_columns

    # Define regions for foundations and waste pile
    foundation_region = (screen_start_x, 235, 1002, 353)
    waste_region = (1022, 235, 1164, 351)

    # Parse each detected card
    for card in detected_cards:
        x, y = card["position"]
        card_name = card["card"]

        # Check if the card is in the foundation region
        if foundation_region[0] <= x <= foundation_region[2] and foundation_region[1] <= y <= foundation_region[3]:
            suit = card_name[1]  # Extract suit
            game_state["foundations"][suit] = card_name[0]  # Assign card value to foundation suit

        # Check if the card is in the waste region
        elif waste_region[0] <= x <= waste_region[2] and waste_region[1] <= y <= waste_region[3]:
            game_state["waste"][-1] = card_name

        # Otherwise, assign the card to the nearest column
        else:
            column_index = (x - screen_start_x) // column_width
            if game_state["columns"][column_index][-1] == 'X':
                game_state["columns"][column_index][-1] = card_name

    return game_state


def visualize_game_state(tableau, foundation, waste, waste_pile, moves_performed=None, reshuffles=None):
    """
    Visualize the entire game state, including tableau, foundation, and waste piles.
    Optional: moves_performed, reshuffles.
    """
    print(Fore.LIGHTBLUE_EX + "\nGame State:")

    # Print the tableau
    print(Fore.YELLOW + "Tableau:")
    for i, column in enumerate(tableau, start=1):
        if column:
            formatted_column = [
                "X" if idx < len(column) - 1 else card for idx, card in enumerate(column)
            ]
            print(Fore.YELLOW + f"  Column {i}:" + Fore.RED + f" {' '.join(column)}")
        else:
            print(Fore.YELLOW + f"  Column {i}:" + Fore.RED + " Empty")

    # Print the foundation
    print(Fore.YELLOW + "\nFoundation:")
    for suit, cards in foundation.items():
        print(Fore.YELLOW + f"  {suit}: " + Fore.RED + f"{' '.join(cards) if cards else 'Empty'}")

    # Print the waste
    print(Fore.YELLOW + f"\nWaste: " + Fore.LIGHTMAGENTA_EX + f"{' '.join(waste) if waste else 'Empty'}")

    # Print the waste pile
    print(
        Fore.YELLOW + f"Waste Pile: " + Fore.MAGENTA + f"{' '.join(waste_pile) if waste_pile else 'Empty'}" + Style.RESET_ALL)

    # Print moves performed if provided
    if moves_performed is not None:
        print(f"Moves: {moves_performed}")

    # Print reshuffles if provided
    if reshuffles is not None:
        print(f"Reshuffles: {reshuffles}")

    # Add a separator
    print("-" * 40)


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
        print(Fore.GREEN + f"{move['card']} from Column {from_column}, {move['action']} {to_column}" + Style.RESET_ALL)
