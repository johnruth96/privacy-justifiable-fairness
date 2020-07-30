import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

ADULT_DATA = os.path.join(DATA_DIR, "adult.data")
ADULT_DATA_TEST = os.path.join(DATA_DIR, "adult.test")

RESULT_DIR = os.path.join(PROJECT_DIR, "results")
