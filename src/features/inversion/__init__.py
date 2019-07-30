import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src import DATA_PATH

# Saves data to the same folder
XR_HAND_CENTRED_PATH = f'{DATA_PATH}/XR_HAND_CENTRED'
path_to_data = f'{XR_HAND_CENTRED_PATH}/*/*/*'
paths = glob.glob(path_to_data)

threshold = 255 / 2

for path in paths:

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Mean color of corners
    color = np.array([img[0:50, 0:50].mean(), img[-50:, -50:].mean(),
                      img[:50, -50:].mean(), img[-50:, :50].mean()]).mean()

    if img.mean() > threshold or color > threshold:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.show()
        cv2.imwrite(path, 255 - img)
