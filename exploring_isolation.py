# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:50:31 2020

@author: Lio

This code contains some first steps at an attempt to isolate copepods.
"""

import numpy as np
from skimage import morphology, filters
import copeproc as cp
import random as rand
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

# Get all the images

wd = 'C:/Users/Emily/Desktop/Image Processing/select_copepods/'

images, imgnames = cp.get_images(wd)


# Downsample to 1/2 size
for image in images:
    image = image[::2][::2]

QUANTITY = len(images)

# Select a single image at random, in case you want to spot test something
# on just one image
imgpick = rand.randint(0,QUANTITY-1)
testimg, testimgname = images[imgpick], imgnames[imgpick]


# Toggle which line is commented to run either on the selected image or the 
# whole batch

test = [testimg]
#test = images


# A helper to erode all images in a list
def erode(images):
    # Pick the element for the erosion to be
    # [[1,1,1]
    #  [1,1,1]
    #  [1,1,1]]
    element = morphology.square(width = 3)
    eroded = []
    for image in images:
        # Erode each image
        eroded.append(morphology.erosion(image,element))
    return eroded

# Helper to remove all regions that touch the edges of labeled image
def remove_edge_regions(labeled_image):
    # Get all the labels present in the top row
    toplabels = [l for l in labeled_image[0] if l != 0]
    # Get all the labels present in the bottom row
    bottomlabels = [l for l in labeled_image[-1] if l != 0]
    # Get all the labels present in the left column
    leftlabels = [l[0] for l in labeled_image if l[0] != 0]
    # Get all the labels present in the right column
    rightlabels = [l[-1] for l in labeled_image if l[-1] != 0]
    
    # Put all of those labels into a list and keep only the unique values
    labels = []
    for labs in (toplabels, bottomlabels, leftlabels, rightlabels):
        labels.extend(labs)
    labels = np.unique(labels)
    
    # Iterate through all pixels in the image
    # THIS IS WHY IT'S SLOW
    for r in range(labeled_image.shape[0]):
        for c in range(labeled_image.shape[1]):
            # If this pixel contains a label that we found on an edge...
            if labeled_image[r][c] in labels:
                # ... then we want to region to go away, so set this pixel = 0
                labeled_image[r][c] = 0
    # Now the image should contain only regions that don't touch the edge.
    return labeled_image




# First show all of our images, to see what we're working with.
cp.show_all(test)

# Do a triangle filter to every image, and show the result.
for n in range(len(test)):
  thresh = filters.threshold_triangle(test[n])
  test[n] = test[n] < thresh
cp.show_all(test)

# Now that the images are binary, erode them all 10 times to get rid of the
# gridlines.
for i in range(10):
  test = erode(test)
  
# Now we should have only blobs. Use scipy.ndimage.label to identify the blobs.
# First we need to change the images from boolean to integer 0's and 1's. 
for n in range(len(test)):
  test[n] = test[n].astype(int)

# Now we can label the regions
for img in test:
    labels, count = ndi.label(img)
    print(f"{count} regions found")
    img[:] = labels

# Show each region in a different color, to make it clear that they're labeled
cp.show_all(test, cmap = 'inferno')

# Now get rid of the regions touching the edges of the images
edited = []
for img in test:
    edited.append(remove_edge_regions(img))
    
# And show the result.
cp.show_all(edited,cmap = 'inferno')

# That's as far as I got with this, but the idea is to select the largest
# remaining region and assume it's the copepod. Then you can get rid of
# everything else and dilate 10 times to get the copepod back to its original
# dimensions. Once it's the only thing in the image, it should be pretty easy 
# to measure.

#%% And we're back


for img in test:
    # See what find_objects does
    objects = ndi.find_objects(img)
    print("objects is")
    print(objects)
    for obj in objects:
        if obj != None:
            print("obj is")
            print(obj)
            fig,ax=plt.subplots()
            ax.imshow(img[obj])
    
    # Select largest remaining region
    
    
    # Get rid of all other regions
    
    
    
    # Dilate 10 times
    
    
    # Measure the major axis of the copepod






















