# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 20:25:54 2020

@author: Lio
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage import measure
import math

# Make a test image

image = np.array(((0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,1,1,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1),
                  (1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,1,1),
                  (1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1),
                  (1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0),
                  (0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0)))

# Show the image
def show(image,cmap = 'gray'):
    fig,ax=plt.subplots()
    ax.imshow(image, cmap = cmap)

# show(image)



# Label regions
labels, count = ndi.label(image)
print(f"{count} regions found")
image[:] = labels

# (Show the image)
# show(image, cmap = "inferno")

# Get clear on what the edges are
print(np.shape(image))
n_rows = np.shape(image)[0]
n_cols = np.shape(image)[1]



# Get all the individual blobs in the image, encoded as lists of slice objects
objects = ndi.find_objects(image)

# Track labels with a counter
label = 0

# We'll want to keep the label and rough size of each non-edge blob.
kept_blobs = []
kept_labels = []
kept_areas = []

# Look at each blob
for obj in objects:
    # Each blob is a list of slices like 
    # (slice(start,stop,step), slice(start,stop,step))
    if obj != None: # Except sometimes they're nones, so skip those
        # Increment label
        label += 1
        # The first slice is the rows (vertical)
        # The second slice is the columns (horizontal)
        row_slice = obj[0]
        col_slice = obj[1]
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
        
        
        # show(image[obj])
         
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
 
# (Show the image)
show(image)

#%% Measuring the image

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skimage.draw import ellipse_perimeter
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate


# image = np.zeros((600, 600))

# rr, cc = ellipse(300, 350, 100, 220)
# image[rr, cc] = 1

# image = rotate(image, angle=15, order=0)

# rr, cc = ellipse(100, 100, 60, 50)
# image[rr, cc] = 1

label_img = label(image)
regions = regionprops(label_img)


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