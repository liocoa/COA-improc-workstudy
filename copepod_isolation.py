# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 14:46:29 2020

@author: Lio

Building the process of measuring copepods.
"""

import numpy as np
from skimage import morphology, filters
import copeproc as cp
import random as rand
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

#%% HELPERS

# Erode

# Remove edge regions

# (Show the image)

def show(image):
    fig,ax=plt.subplots()
    ax.show(image)



#%% Get the image


#%% Downsample to 1/2 size


# (Show the image)


#%% Triangle filter


# (Show the image)


#%% Erode 10 times


#%% Label the blobs

# Image to int from bool

# Label regions

# (Show the image)

#%% Look at each labeled blob in turn

# Remove edge ones

# Keep biggest one

# Dilate 10 times

# Measure the blob

