#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 19:03:07 2019

@author: hitansh
"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys
import os

sys.path.append("..")

POSITIVE_IMAGE_PATHS = []
NEGATIVE_IMAGE_PATHS = []

os.chdir(r'../../../data/TensorFlow/workspace/training_demo/images')

for r, d, f in os.walk('showcase_sample/positive/'):
    for file in f:
        if '.png' in file:
            POSITIVE_IMAGE_PATHS.append(os.path.join(r, file))

for r, d, f in os.walk('showcase_sample/negative/'):
    for file in f:
        if '.png' in file:
            NEGATIVE_IMAGE_PATHS.append(os.path.join(r, file))


img = cv.imread(POSITIVE_IMAGE_PATHS[4])
plt.figure(1)
plt.imshow(img)

# It is a mask image where we specify which areas are background, foreground or probable background/foreground etc.
mask = np.zeros(img.shape[:2], np.uint8)

# bdgModel, fgdModel - These are arrays used by the algorithm internally. You just create two np.float64 type zero arrays of size (1,65).
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

'''
iterCount - Number of iterations the algorithm should run.
mode - It should be cv.GC_INIT_WITH_RECT or cv.GC_INIT_WITH_MASK or combined which decides whether we are drawing rectangle or final touchup strokes.
rect - It is the coordinates of a rectangle which includes the foreground object in the format (x,y,w,h)
'''
# x,y,h,w
rect = (25, 25, 400, 290)
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img * mask2[:, :, np.newaxis]
plt.figure(2)
plt.imshow(img), plt.colorbar(), plt.show()
