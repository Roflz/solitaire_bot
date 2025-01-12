from scripts import utils


image_path = "C:\_\solitaire_bot\\templates\\test_screenshots\game_screenshot_9.bmp"
suits_directory = "C:\_\solitaire_bot\\templates\suits"
ranks_directory = "C:\_\solitaire_bot\\templates\supposedly_perfect_matches"

utils.detect_suit_and_rank(image_path, suits_directory, ranks_directory, suit_threshold=0.9, rank_threshold=0.95)