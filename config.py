# config.py
import os


# Spacial Regions
NUM_COLUMNS = 7
SCREEN_START_X = 656
SCREEN_END_X = 1264
COLUMN_WIDTH = (SCREEN_END_X - SCREEN_START_X) // NUM_COLUMNS
FOUNDATION_REGION = (SCREEN_START_X, 235, 1002, 353)
WASTE_REGION = (1022, 235, 1164, 351)
SCREEN_HEIGHT, SCREEN_WIDTH = 1080, 1920
COLUMN_0_REGION = (656, 441, 742, 1080)
COLUMN_1_REGION = (742, 441, 828, 1080)
COLUMN_2_REGION = (828, 441, 914, 1080)
COLUMN_3_REGION = (914, 441, 1000, 1080)
COLUMN_4_REGION = (1000, 441, 1086, 1080)
COLUMN_5_REGION = (1086, 441, 1172, 1080)
COLUMN_6_REGION = (1172, 441, 1258, 1080)

HIDDEN_CARD_HEIGHT = 16
VISIBLE_CARD_HEIGHT = 30

CLICK_POINTS = {
        "waste_shuffle": [(3135, 272)],
        # "waste": [(2978, 267), (3008, 265), (3038, 265)],
        "waste": [(3008, 265)],
        "column_0": [(2615, 450)],
        "column_1": [(2703, 450)],
        "column_2": [(2790, 450)],
        "column_3": [(2877, 450)],
        "column_4": [(2963, 450)],
        "column_5": [(3050, 450)],
        "column_6": [(3135, 450)]
}

REGIONS = {
        "waste": WASTE_REGION,
        "column_0": COLUMN_0_REGION,
        "column_1": COLUMN_1_REGION,
        "column_2": COLUMN_2_REGION,
        "column_3": COLUMN_3_REGION,
        "column_4": COLUMN_4_REGION,
        "column_5": COLUMN_5_REGION,
        "column_6": COLUMN_6_REGION
    }

# Paths
PROJECT_PATH = os.getcwd()
RANK_TEMPLATES_DIR = os.path.join(PROJECT_PATH, "templates", "ranks")
SUIT_TEMPLATES_DIR = os.path.join(PROJECT_PATH, "templates", "suits")
