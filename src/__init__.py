import os
from os.path import abspath, dirname

ROOT_PATH = dirname(dirname(abspath(__file__)))

DATA_PATH = f'{ROOT_PATH}/data/train'
MODELS_DIR = f'{ROOT_PATH}/models'

TMP_IMAGES_DIR = f'{ROOT_PATH}/models/tmp_photoshop'
os.makedirs(TMP_IMAGES_DIR, exist_ok=True)

MLFLOW_TRACKING_URI = 'http://localhost:5001'
