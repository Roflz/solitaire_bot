from collections import Counter
from scripts import utils
import os


templates_dir = os.path.join("..", "templates", "ranks")
screenshot_path = '/templates/card_mismatch_screenshots/game_screenshot_6S.bmp'
screenshot_width = 1920
screenshot_height = 1080

# Initialize the Counter to keep track of card counts
card_counts = Counter()
for i in range(30):
    cards = utils.find_cards(screenshot_path, templates_dir, threshold=0.99)

    # Assuming `ranks` is a list of dictionaries with a "card" key
    # Update the counts for this iteration
    card_counts.update(card['card'] for card in cards)
    game_state = utils.parse_start_state(cards, screenshot_width, screenshot_height)
    print(i)

# Convert Counter to a dictionary to inspect the results
final_counts = dict(card_counts)
print("Final card counts:", final_counts)
