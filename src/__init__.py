from os.path import abspath, dirname
import os

ROOT_PATH = dirname(dirname(abspath(__file__)))

XR_HAND_PATH = f'{ROOT_PATH}/data/train/XR_HAND'
XR_HAND_CROPPED_PATH = f'{ROOT_PATH}/data/train/XR_HAND_CROPPED'

MODELS_DIR = f'{ROOT_PATH}/models'

TMP_IMAGES_DIR = f'{ROOT_PATH}/models/tmp'
os.makedirs(TMP_IMAGES_DIR, exist_ok=True)

MLFLOW_TRACKING_URI = 'http://localhost:5001'
