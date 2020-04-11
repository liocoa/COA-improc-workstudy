# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 15:09:05 2020

@author: Lio

Lecture 2 code
"""

#%%
from __future__ import division, print_function

#%%
#Turn off pixel interpolation? What's that???
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'none'

#%%
import skdemo
from skimage import data
# Rename module so we don't shadow the builtin function
import skimage.filters as filters

image = data.camera()
pixelated = image[::10, ::10]
gradient = filters.sobel(pixelated)
skdemo.imshow_all(pixelated, gradient)
#%%
skdemo.imshow_all(gradient, gradient > 0.3)

#%%
#Steps of canny edge detection
#Step 1: Gaussian filter
from skimage import img_as_float

sigma = 1  # Standard-deviation of Gaussian; larger smooths more.
pixelated_float = img_as_float(pixelated)
pixelated_float = pixelated
smooth = filters.gaussian(pixelated_float, sigma)
skdemo.imshow_all(pixelated_float, smooth)
#%%
#Step 2: Sobel filter
gradient_magnitude = filters.sobel(smooth)
skdemo.imshow_all(smooth, gradient_magnitude)
#%%
#Step 3: Non-maximal suppression
#Goal: suppress gradients that aren't on an edge.
zoomed_grad = gradient_magnitude[15:25, 5:15]
maximal_mask = np.zeros_like(zoomed_grad)
# This mask is made up for demo purposes
maximal_mask[range(10), (7, 6, 5, 4, 3, 2, 2, 2, 3, 3)] = 1
grad_along_edge = maximal_mask * zoomed_grad
skdemo.imshow_all(zoomed_grad, grad_along_edge, limits='dtype')
#The result of the filter is that an edge is only possible if there are no 
#better edges near it.
#%%
#Step 4: Hystersis thresholding
#Goal: Prefer pixels that are connected to edges
from skimage import color

low_threshold = 0.2
high_threshold = 0.3
label_image = np.zeros_like(pixelated)
# This uses `gradient_magnitude` which has NOT gone through non-maximal-suppression.
label_image[gradient_magnitude > low_threshold] = 1
label_image[gradient_magnitude > high_threshold] = 2
demo_image = color.label2rgb(label_image, gradient_magnitude,
                             bg_label=0, colors=('yellow', 'red'))
plt.imshow(demo_image)

#%%
#Actual canny edge detection
from skimage import feature

image = data.coins()

def canny_demo(**kwargs):
    edges = feature.canny(image, **kwargs)
    plt.imshow(edges)
    plt.show()

canny_demo()
























