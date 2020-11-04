# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 14:46:29 2020

@author: Lio

Building the process of measuring copepods.
"""

import numpy as np
from skimage import morphology, filters, color, io
import copeproc as cp
import random as rand
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import glob

#%% HELPERS

# Erode
def erode(image):
    # Pick the element for the erosion to be
    # [[1,1,1]
    #  [1,1,1]
    #  [1,1,1]]
    element = morphology.square(width = 3)
    image = morphology.erosion(image,element)
    return image

def dilate(image):
    # Pick the element for the dilation to be
    # [[1,1,1]
    #  [1,1,1]
    #  [1,1,1]]
    element = morphology.square(width = 3)
    image = morphology.dilation(image,element)
    return image
    

# Remove edge regions

# Show the image
def show(image,cmap = 'gray'):
    fig,ax=plt.subplots()
    ax.imshow(image, cmap = cmap)





#%% Get the image
    
# Get the image paths

file_path = "C:/Users/Emily/Desktop/Image Processing/select_copepods/"

img_paths = glob.glob(file_path+'*.jpg')
    
# Select one at random

path = rand.choice(img_paths)
    
# Get the image

image = color.rgb2gray(io.imread(path))


#%% Downsample to 1/2 size
image = image[::2,::2]

# # (Show the image)
# show(image)


#%% Triangle filter

thresh = filters.threshold_triangle(image)
image = image < thresh

# # (Show the image)
# show(image)

#%% Erode 10 times

for n in range(10):
    image = erode(image)
    
# # (Show the image)
# show(image)


#%% Label the blobs

# Image to int from bool
image = image.astype(int)

# Label regions
labels, count = ndi.label(image)
print(f"{count} regions found")
image[:] = labels

# (Show the image)
show(image, cmap = "inferno")

#%% Look at each labeled blob in turn

########################################################
#EXPLORATION

# See what find_objects does
objects = ndi.find_objects(image)
print("objects is")
print(objects)
for obj in objects:
    if obj != None:
        print("obj is")
        print(obj)
        show(image[obj])


########################################################
# Remove edge ones


# Keep biggest one

# Dilate 10 times
# for n in range(10):
#     image = dilate(image)
    
# (Show the image)
# show(image)

# Measure the blob

