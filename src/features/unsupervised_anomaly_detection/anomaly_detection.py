#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 18:24:48 2019

@author: hitansh
"""

import numpy as np
import os
import sys
from matplotlib import pyplot as plt
import cv2

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from sklearn import decomposition
# from scipy.signal import find_peaks_cwt
# from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN

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


temp=r'/home/hitansh/TensorFlow/workspace/Screenshot 2019-06-16 at 11.05.14 AM.png'

# grayscale
img = cv2.imread(NEGATIVE_IMAGE_PATHS[24], 0)
# img=cv2.imread(temp,0)
vmax = max(img.max(), -img.min())
# vmax will give maximum value of the color in the image
print(img.shape)
plt.figure(1)
plt.imshow(img)




n_components = 50
estimator = decomposition.NMF(n_components = n_components, init = 'random', tol = 5e-3)    
W = estimator.fit_transform(img)
H = estimator.components_

new_img = np.dot(W,H)
print(new_img.shape)
plt.imshow(new_img)


'''
sc_h = SpectralClustering(4, n_init=10)
sc_h.fit(H.transpose())

sc_w = SpectralClustering(4, n_init=10)
sc_w.fit(W)
print('spectral clustering')
#print(sc_h.labels_)
#print(sc_w.labels_)

hh=[]
pos=-1
for i in sc_h.labels_:
    pos+=1
    if i!=0:
        hh.append(pos)

        
ww=[]
pos=-1
for i in sc_w.labels_:
    pos+=1
    if i!=0:
        ww.append(pos)
        
pprint(hh)
pprint(ww)
'''


plt.figure(2, figsize = (20,40))
plt.imshow(H)

h = H.transpose()
y = []
for i in range(h.shape[0]):
    y.append(sum(h[i][2:48]))

# plt.figure(2)
# plt.plot(y)

'''
y_arr=np.array(y)
x_axis=np.arange(1,len(y))
peaks=find_peaks_cwt(y_arr,x_axis)
x=[y_arr[i] for i in peaks]
plt.plot(peaks,x,'+')
'''

clustering = DBSCAN(eps = 3, min_samples = 2).fit(H.transpose())
labels = clustering.labels_
# pprint(labels)

pos = -1
h_positions = []
for i in labels:
    pos = pos + 1
    if i == -1:
        h_positions.append(pos)
print(h_positions)
h_x = [n_components-2 for i in range(len(h_positions))]
plt.plot(h_positions, h_x, '+')

plt.figure(3, figsize=(20, 40))
plt.imshow(W.transpose())
y = []
for i in range(W.shape[0]):
    y.append(sum(W[i]))

# plt.figure(2)
# plt.plot(y)

'''
y_arr=np.array(y)
x_axis=np.arange(1,len(y))
peaks=find_peaks_cwt(y_arr,x_axis)
x=[y_arr[i] for i in peaks]
plt.plot(peaks,x,'+')
'''

clustering = DBSCAN(eps=3, min_samples=2).fit(W)
labels = clustering.labels_
# pprint(labels)

pos = -1
w_positions=[]
for i in labels:
    pos = pos + 1
    if i == -1:
        w_positions.append(pos)
print(w_positions)
w_x = [n_components-2 for i in range(len(w_positions))]
plt.plot(w_positions, w_x, '+')



x_values = []
y_values = []
for i in h_positions:
    for j in w_positions:
        x_values.append(i)
        img[j][i] = 255
        y_values.append(j)

fig, (ax, ax2) = plt.subplots(ncols = 2)
ax2.imshow(img, cmap = plt.cm.gray, interpolation = 'nearest', vmin = -vmax, vmax = vmax)
ax.imshow(new_img, cmap=plt.cm.gray, interpolation='nearest', vmin =- vmax, vmax = vmax)
plt.show()
