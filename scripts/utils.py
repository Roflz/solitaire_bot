import cv2
import numpy as np
import os
import win32gui
import win32ui
import win32con
from config import *
from colorama import Fore, Style
from PIL import Image


def capture_window(window_name):
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


def find_cards(screenshot_path, templates_dir, threshold=0.8):
    """
    Detects all ranks in a game screenshot using template matching.

    Args:
        screenshot_path (str): Path to the screenshot of the game.
        templates_dir (str): Directory containing card templates.
        threshold (float): Match threshold (default is 0.8).

    Returns:
        List[Dict]: List of detected ranks with their positions and names.
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
    cv2.imwrite("debug_detected_suits.png", screenshot)

    return detected_cards


def detect_suit_and_rank(image_path, suit_threshold=0.9, rank_threshold=0.95):
    """
    Detects the suit and rank of ranks in an image using a two-step process.

    Args:
        image_path (str): Path to the card image.

    Returns:
        List[Dict]: Detected ranks with their suits and ranks.
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detected_cards = []
    detected_suits = []

    # Step 1: Detect Suit
    for suit_name in os.listdir(SUIT_TEMPLATES_DIR):
        suit = suit_name.replace('.png', '')
        suit_template_path = os.path.join(SUIT_TEMPLATES_DIR, suit_name)
        suit_template = cv2.imread(suit_template_path, cv2.IMREAD_GRAYSCALE)
        template_input = (suit, suit_template)

        for region, region_area in REGIONS.items():
            region_input = (region, region_area)
            for detected_suit in detect_templates_in_region(gray_image, template_input, region_input, threshold=0.9):
                detected_suits.append(detected_suit)

    # Parse for only the bottom card in each area
    areas = {}
    for card in detected_suits:
        area = card["area"]
        y_coord = card["position"][1]
        if area not in areas or y_coord > areas[area]["position"][1]:
            areas[area] = card

    detected_suits = list(areas.values())

    for suit in detected_suits:
        # Step 2: Detect Rank of cards
        # Build the path to the rank templates directory using os.path.join to
        # avoid platform specific escaping issues.
        rank_templates_dir = os.path.join(RANK_TEMPLATES_DIR, f"{suit['name']}s")
        best_match_score = 0
        best_result = []

        for rank_name in os.listdir(rank_templates_dir):
            rank = rank_name.replace(".png", "")  # Remove extension for card name
            card = (rank[-1] + rank[0]).upper()
            rank_template_path = os.path.join(rank_templates_dir, rank_name)
            rank_template = cv2.imread(rank_template_path, cv2.IMREAD_GRAYSCALE)

            rank_template_input = (card, rank_template)
            region_input = (suit['area'], REGIONS[suit['area']])

            result, match_score = detect_templates_in_region(gray_image, rank_template_input, region_input, threshold=0.75, return_score=True)
            if match_score > best_match_score:
                best_result = result
                best_match_score = match_score
        for detected_card in best_result:
            detected_cards.append(detected_card)

    return detected_cards


def draw_regions(screenshot_path):
    # Load the screenshot
    screenshot = cv2.imread(screenshot_path)

    # Draw column regions
    for i in range(NUM_COLUMNS):
        x1 = i * COLUMN_WIDTH + SCREEN_START_X
        x2 = x1 + COLUMN_WIDTH
        y1, y2 = 441, SCREEN_HEIGHT  # Columns start below waste and foundation regions
        cv2.rectangle(screenshot, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(screenshot, f"Column {i + 1}", (x1 - 10, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Draw foundation region
    x1, y1, x2, y2 = FOUNDATION_REGION
    cv2.rectangle(screenshot, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(screenshot, "Foundations", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw Waste region
    x1, y1, x2, y2 = WASTE_REGION
    cv2.rectangle(screenshot, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(screenshot, "Waste", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Save the annotated image
    cv2.imwrite("annotated_regions.png", screenshot)
    print("Annotated image saved as 'annotated_regions.png'.")


def parse_start_state(detected_cards):
    """
    Parse the detected ranks into a structured game state.

    Args:
        detected_cards (list): List of detected ranks with positions.
        screen_width (int): Width of the gameplay screen.
        SCREEN_HEIGHT (int): Height of the gameplay screen.

    Returns:
        dict: Parsed game state.
    """
    # Initialize game state
    game_state = {
        "columns": [[(0, 'X')] * i for i in range(NUM_COLUMNS)],
        "foundations": {"H": [], "S": [], "D": [], "C": []},
        "waste": [],
        "waste_pile": ['X'] * 24
    }

    # Parse each detected card
    for card in detected_cards:
        x, y = card["position"]
        card_name = card["name"]

        # Check if the card is in the foundation region
        if FOUNDATION_REGION[0] <= x <= FOUNDATION_REGION[2] and FOUNDATION_REGION[1] <= y <= FOUNDATION_REGION[3]:
            suit = card_name[1]  # Extract suit
            game_state["foundations"][suit] = card_name[0]  # Assign card value to foundation suit

        # Check if the card is in the waste region
        elif WASTE_REGION[0] <= x <= WASTE_REGION[2] and WASTE_REGION[1] <= y <= WASTE_REGION[3]:
            game_state["waste"].append(card_name)

        # Otherwise, assign the card to the nearest column
        else:
            column_index = (x - SCREEN_START_X) // COLUMN_WIDTH
            game_state["columns"][column_index].append((y, card_name))

    # Sort each column by the y-coordinate
    for column in game_state["columns"]:
        column.sort()  # Sort by y-coordinate
        for i in range(len(column)):
            column[i] = column[i][1]  # Extract card name only

    return game_state


def parse_game_state(game_state, detected_cards):
    """
    Parse the detected ranks into a structured game state.

    Args:
        detected_cards (list): List of detected ranks with positions.
        screen_width (int): Width of the gameplay screen.
        SCREEN_HEIGHT (int): Height of the gameplay screen.

    Returns:
        dict: Parsed game state.
    """

    # Parse each detected card
    for card in detected_cards:
        x, y = card["position"]
        card_name = card["name"]

        # Check if the card is in the foundation region
        if FOUNDATION_REGION[0] <= x <= FOUNDATION_REGION[2] and FOUNDATION_REGION[1] <= y <= FOUNDATION_REGION[3]:
            suit = card_name[1]  # Extract suit
            game_state["foundations"][suit] = card_name[0]  # Assign card value to foundation suit

        # Check if the card is in the waste region
        elif WASTE_REGION[0] <= x <= WASTE_REGION[2] and WASTE_REGION[1] <= y <= WASTE_REGION[3]:
            game_state["waste"][-1] = card_name

        # Otherwise, assign the card to the nearest column
        else:
            column_index = (x - SCREEN_START_X) // COLUMN_WIDTH
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


def crop_image_ndarray(image_array, crop_area):
    """
    Crops an image represented as a NumPy ndarray.

    Args:
        image_array (ndarray): Input image as a NumPy array.
        crop_area (tuple): The crop area as (left, upper, right, lower).

    Returns:
        ndarray: Cropped image as a NumPy array.
    """
    try:
        # Convert ndarray to a Pillow Image
        image = Image.fromarray(image_array)
        # Crop the image
        cropped_image = image.crop(crop_area)
        # Convert the cropped image back to ndarray
        cropped_array = np.array(cropped_image)

        return cropped_array
    except Exception as e:
        print(f"Error cropping image: {e}")
        return None


def transform_points(coords, region):
    """
    Transforms the x and y coordinates by adding the region's (x1, y1) offset.

    Args:
        coords (tuple): A tuple containing two ndarrays (y_coords, x_coords).
        region (tuple): The region's (x1, y1, x2, y2) coordinates.

    Returns:
        tuple: A tuple of transformed x and y coordinates as ndarrays.
    """
    y_coords, x_coords = coords  # Unpack the input coordinates
    x1, y1, _, _ = region  # Extract region coordinates (x1, y1)

    # Add the offset to each coordinate
    transformed_y = y_coords + y1
    transformed_x = x_coords + x1

    return transformed_y, transformed_x


def detect_templates_in_region(image: np.ndarray, template, region, threshold: float = 0.8, return_score=False):
    """
    Detects the presence of a template image within a specified region of the input image.

    Parameters:
        image (np.ndarray): The input image in which to detect the template.
        template (Tuple[str, np.ndarray]): A tuple containing the template name (str) and the template image (np.ndarray).
        region (Tuple[str, Tuple[int, int, int, int]]): A tuple containing the region name (str) and the region coordinates (a tuple of four
        integers defining the top-left and bottom-right corners).
        threshold (float): The similarity threshold for template matching. Default is 0.8.

    Returns:
        List[Dict[str, object]]: A list of dictionaries, each containing the template name ('suit'),
    the detected position ('position'), and the region ('area').
    """
    detected_templates = []
    locations = ()

    region_image_gray = crop_image_ndarray(image, region[1])
    result = cv2.matchTemplate(region_image_gray, template[1], cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    locations = np.where(result >= threshold)

    # Store matches with their positions
    if locations[0].size != 0 and locations[1].size != 0:
        locations = transform_points(locations, region[1])
        for x, y in zip(locations[1], locations[0]):
            x, y = np.int64(x), np.int64(y)  # Switch x, y for OpenCV format
            point = (x, y)  # Top-left corner of the matched template

            # Check if the point is too close to existing points
            if not any(abs(x - i["position"][0]) <= 10 and abs(y - i["position"][1]) <= 10 for i in
                       detected_templates):
                detected_templates.append({
                    "name": template[0],
                    "position": point,
                    "area": region[0]
                })
    if return_score:
        return detected_templates, max_val
    else:
        return detected_templates
