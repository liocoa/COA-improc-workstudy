# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 20:25:54 2020

@author: Lio
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

# Make a test image

image = np.array(((0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,1,1),
                  (1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1),
                  (1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1),
                  (1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0),
                  (0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                  (0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0)))

# Show the image
def show(image,cmap = 'gray'):
    fig,ax=plt.subplots()
    ax.imshow(image, cmap = cmap)

show(image)



# Label regions
labels, count = ndi.label(image)
print(f"{count} regions found")
image[:] = labels

# (Show the image)
show(image, cmap = "inferno")

# Get clear on what the edges are
print(np.shape(image))
n_rows = np.shape(image)[0]
n_cols = np.shape(image)[1]



# Get all the individual blobs in the image, encoded as lists of slice objects
objects = ndi.find_objects(image)

# Look at each blob
for obj in objects:
    # Each blob is a list of slices like 
    # (slice(start,stop,step), slice(start,stop,step))
    if obj != None: # Except sometimes they're nones, so skip those
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
        nums_to_check = [row_slice_indices[0],
                         row_slice_indices[1],
                         col_slice_indices[0],
                         col_slice_indices[1]]
        edge_vals = [0, n_rows, n_cols]
        
        check =  any(item in nums_to_check for item in edge_vals)
         
        if check is True:
            print(f"The list {nums_to_check} contains some elements of the list {edge_vals}")    
            # We can discard this blob.
            # HOW DO I DO THAT
        else :
            print("No, nums_to_check doesn't have any elements of the edge_vals.")
            # We want to keep this blob.
        # show(image[obj])
pass
 
show(image)           
            
# Lingering questions:
# How to ID the blob in each slice and discard it?

