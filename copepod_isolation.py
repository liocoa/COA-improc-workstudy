# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 14:46:29 2020

@author: Lio

Building the process of measuring copepods.
"""

import numpy as np
from skimage import morphology, filters, color, io, measure
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

# (Show the image)
show(image)


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
# show(image, cmap = "inferno")

#%% Look at each labeled blob in turn

# Get clear on what the edges of the main image are
print(np.shape(image))
n_rows = np.shape(image)[0]
n_cols = np.shape(image)[1]

# Get all the individual blobs in the image, encoded as lists of slice objects
objects = ndi.find_objects(image)

# Track labels with a counter
label = 0

# We'll want to keep the label and rough size of each non-edge blob.
kept_labels = []
kept_areas = []

# Look at each blob
for blob in objects:
    # Each blob is a tuple of slices like 
    # (slice(start,stop,step), slice(start,stop,step))
    if blob != None: # Except sometimes they're nones, so skip those
        # Increment label
        label += 1
        # The first slice is the rows (vertical)
        # The second slice is the columns (horizontal)
        row_slice = blob[0]
        col_slice = blob[1]
        # I can't use these objects directly.
        # I can access the (start, stop, step) values as a tuple using the 
        # indices() function of slice objects.
        # That function takes the length of the object you're slicing
        # as its single (required) argument.
        row_slice_indices = row_slice.indices(n_rows)
        col_slice_indices = col_slice.indices(n_cols)
        # These objects are tuples of the (start, stop, step) of the boxes
        # that find_objects has cropped around the blobs.
        # Step is inconsequential here, because a cropped box always has a 
        # step of 1. What we're really interested in is start and stop.
        # If any of the start/stop values in in these tuples are edge values,
        # meaning they're equal to zero or the image size, then the blob is
        # on the edge and can be discarded.
        row_nums_to_check = [row_slice_indices[0],
                             row_slice_indices[1]]
        row_edge_vals = [0, n_rows]
        
        col_nums_to_check = [col_slice_indices[0],
                             col_slice_indices[1]]
        
        col_edge_vals = [0, n_cols]
        
        check_row =  any(item in row_nums_to_check for item in row_edge_vals)
        check_col =  any(item in col_nums_to_check for item in col_edge_vals)
        
        
        # show(image[blob])
         
        if check_row or check_col:   
            # We can discard this blob.
            # HOW DO I DO THAT
            # label is label of this blob
            # Make everything with that label that's within this rectangle a 0
            image = np.ma.where(image==label, np.zeros_like(image), image)
        else :
            # We want to keep this blob.
            # We need to know which blob is the biggest blob. 
            # We can do that roughly for now.
            # Let's just calculate the area of the minimal rectangle.
            area = (row_slice_indices[1] - row_slice_indices[0]) * (col_slice_indices[1] - col_slice_indices[0])
            
            # Keep this blob.
            kept_labels.append(label)
            kept_areas.append(area)
            
        # show(image[obj])


# Keep the label that's at the same index as the max kept area
largest_label = kept_labels[kept_areas.index(max(kept_areas))]
print(largest_label)

# Now we can get rid of everything that's not that blob
image = np.ma.where(image!=largest_label, np.zeros_like(image), image)
 
################
# NEED SOME ASSURANCE THAT THE SELECTED BLOB IS ACTUALLY A COPEPOD
################

# (Show the image)
# show(image)

# Dilate 10 times
for n in range(10):
    image = dilate(image)
    
# (Show the image)
show(image)


# Measure the blob

import math


regions = measure.regionprops(image)


fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)

for props in regions:
    y0, x0 = props.centroid
    orientation = props.orientation
    
    # Minor axis
    x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
    y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
    x2 = x0 - math.cos(orientation) * 0.5 * props.minor_axis_length
    y2 = y0 + math.sin(orientation) * 0.5 * props.minor_axis_length
    
    
    # Major axis
    x3 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
    y3 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length
    x4 = x0 + math.sin(orientation) * 0.5 * props.major_axis_length
    y4 = y0 + math.cos(orientation) * 0.5 * props.major_axis_length
    
    
    #########
    
    print(props.major_axis_length)
    
    #########  
    

    ax.plot((x2, x1), (y2, y1), '-r', linewidth=2.5) # Minor axis
    ax.plot((x3, x4), (y3, y4), '-r', linewidth=2.5) # Major axis
    ax.plot(x0, y0, '.g', markersize=15) # The centroid




plt.show()








