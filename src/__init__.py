import os
from os.path import abspath, dirname

import paramiko

ROOT_PATH = dirname(dirname(abspath(__file__)))

DATA_PATH = f'{ROOT_PATH}/data/train'
MODELS_DIR = f'{ROOT_PATH}/models'

TMP_IMAGES_DIR = f'{ROOT_PATH}/models/tmp'
os.makedirs(TMP_IMAGES_DIR, exist_ok=True)

MLFLOW_TRACKING_URI = 'http://localhost:5001'

TOP_K = 200

SSH = paramiko.SSHClient()
SSH.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
SSH.connect('10.195.1.150', username='ubuntu', )

SFTP = SSH.open_sftp()
