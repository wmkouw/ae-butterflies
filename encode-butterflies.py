#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run an autoencoder on a set of butterfly images. Cluster the embbedings and
decode the cluster centroids.

Author: W.M.Kouw
Date: 28-11-2018
"""
import os.path
import cv2
import numpy as np
from glob import glob

import tensorflow as tf
import keras.backend as K
from keras.callbacks import TensorBoard
from ae import convolutionalAutoEncoder

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt


''' Experimental parameters '''

# Whether to load model
load_model = False

# Number of images
nI = 64

# Number of clusters
nK = 4

# Kernel shape
kernel_shape = (3, 3)

# Pooling shape
pool_shape = (2, 2)

# Batch size
batch_size = 16

# Number of epochs
num_epochs = 32

# Optimizer
opt = 'RMSprop'

# Loss function
loss = 'mean_absolute_error'

''' Read data '''

# Input shape
input_shape = (600, 900, 3)

# Directory
data_dir = '../data/specAarhus/'

# Check if images are already converted to numpy arrays
if os.path.isfile('specAarhus_tops.npy'):

    X = np.load('specAarhus_tops.npy')

else:

    # Check for top images only
    tops = np.sort(glob(data_dir + '*top*'))

    # Preallocate array
    X = np.zeros((nI, *input_shape))

    for i in range(nI):

        # Read image and store in array
        im = cv2.imread(tops[i])

        # Swap color channels and normalize
        X[i, :, :, :] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) / 255.

    # Store read images as numpy array
    np.save('specAarhus_tops.npy', X)

for i in range(3):

    plt.figure()
    plt.imshow(X[i, :, :, :], vmin=0.0, vmax=1.0)
    plt.savefig('viz/X' + str(i) + '.png',
                bbox_inches='tight',
                padding='tight')


''' Start training '''

# Define model
model = convolutionalAutoEncoder(input_shape=input_shape,
                                 kernel_shape=kernel_shape,
                                 pool_shape=pool_shape,
                                 num_epochs=num_epochs,
                                 batch_size=batch_size,
                                 loss=loss,
                                 opt=opt)

if load_model:

    # Load model from file
    model.load('models/CAE001')

else:
    # Call training method
    model.train(X, callback=True)

    # Save model
    model.save('models/CAE001')

''' Encode '''

H = model.encode(X)

for i in range(3):

    plt.figure()
    plt.imshow(H[i, :, :, :], vmin=0.0, vmax=1.0)
    plt.savefig('viz/H' + str(i) + '.png',
                bbox_inches='tight',
                padding='tight')

''' Decode '''

# Preallocate decoded image array
Z = model.decode(H)

for i in range(3):

    plt.figure()
    plt.imshow(Z[i, :, :, :], vmin=0.0, vmax=1.0)
    plt.savefig('viz/Z' + str(i) + '.png',
                bbox_inches='tight',
                padding='tight')

''' Cluster and decode centroids'''

# Fit a k-means algorithm
kM = KMeans(n_clusters=nK)
kM.fit(H.reshape((nI, -1)))

# Extract cluster centroids
centroids = kM.cluster_centers_.reshape((nK, *H.shape[1:]))

for k in range(nK):

    plt.figure()
    plt.imshow(centroids[k, :, :, :], vmin=0.0, vmax=1.0)
    plt.savefig('viz/C' + str(k) + '.png',
                bbox_inches='tight',
                padding='tight')

# Preallocate decoded image array
D = model.decode(centroids)

for k in range(nK):

    plt.figure()
    plt.imshow(D[k, :, :, :], vmin=0.0, vmax=1.0)
    plt.savefig('viz/D' + str(k) + '.png',
                bbox_inches='tight',
                padding='tight')

plt.show()
