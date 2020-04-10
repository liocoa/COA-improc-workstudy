# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 20:07:46 2020

@author: Lio

This is the example code from the lecture on color and exposure
"""


from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt


from skimage import data

color_image = data.chelsea()

print(color_image.shape)
plt.imshow(color_image);

red_channel = color_image[:, :, 0]  # or color_image[..., 0]

plt.imshow(red_channel);

red_channel.shape



#%%

#Exercise: Split image into component colors and display separately

import skdemo
from skimage import io
color_image = io.imread('C:/Users/Emily/skimage-tutorials/images/balloon.jpg')

# This code is just a template to get you started.
red_image = [1,0,0] * color_image
green_image = [0,1,0] * color_image
blue_image = [0,0,1] * color_image

skdemo.imshow_all(color_image, red_image, green_image, blue_image)


#%%
color_patches = color_image.copy()
# Remove green (1) & blue (2) from top-left corner.
color_patches[:100, :100, 1:] = 0
# Remove red (0) & blue (2) from bottom-right corner.
color_patches[-100:, -100:, (0, 2)] = 0
plt.imshow(color_patches);


#%%
image = data.camera()
skdemo.imshow_with_histogram(image);

#%%
cat = data.chelsea()
skdemo.imshow_with_histogram(cat);

#%%
from skimage import exposure
high_contrast = exposure.rescale_intensity(image, in_range=(10, 180))
skdemo.imshow_with_histogram(high_contrast);

#%%
ax_image, ax_hist = skdemo.imshow_with_histogram(image)
skdemo.plot_cdf(image, ax=ax_hist.twinx())

equalized = exposure.equalize_hist(image)
ax_image, ax_hist = skdemo.imshow_with_histogram(equalized)
skdemo.plot_cdf(equalized, ax=ax_hist.twinx())

print(equalized.dtype)

#%%
equalized = exposure.equalize_adapthist(image)
ax_image, ax_hist = skdemo.imshow_with_histogram(equalized)
skdemo.plot_cdf(equalized, ax=ax_hist.twinx())

#%%
skdemo.imshow_with_histogram(image);
threshold = 50
ax_image, ax_hist = skdemo.imshow_with_histogram(image)
# This is a bit of a hack that plots the thresholded image over the original.
# This just allows us to reuse the layout defined in `plot_image_with_histogram`.
ax_image.imshow(image > threshold)
ax_hist.axvline(threshold, color='red');

#%%
# Rename module so we don't shadow the builtin function
import skimage.filters as fltr
threshold = fltr.threshold_otsu(image)
print(threshold)
plt.imshow(image > threshold);

#%%
from skimage import color
#color.rgb2  # <TAB>
plt.imshow(color_image);
lab_image = color.rgb2lab(color_image)
lab_image.shape

plt.imshow(lab_image);
skdemo.imshow_all(lab_image[..., 0], lab_image[..., 1], lab_image[..., 2],
                 titles=['L', 'a', 'b'])


#%%
#Exercise: Green screen

from skimage import io
greenscreen = io.imread('C:/Users/Emily/skimage-tutorials/images/greenscreen.jpg')
forest = io.imread('C:/Users/Emily/skimage-tutorials/images/forest.jpg')

#Experiment with extracting actors from background
lab_greenscreen = color.rgb2lab(greenscreen)
lab_forest = color.rgb2lab(forest)
skdemo.imshow_all(lab_greenscreen[..., 0], lab_greenscreen[..., 1], 
                 lab_greenscreen[..., 2],
                 titles=['L', 'a', 'b'])
skdemo.imshow_with_histogram(lab_greenscreen[..., 1])


    
#This code is from the solutions.
#What is happening in this tuple assignment?
luminance, a, b = np.rollaxis(lab_greenscreen, axis=-1)
titles = ['luminance', 'a-component', 'b-component']
skdemo.imshow_all(luminance, a, b, titles=titles)




#%%
#I know I want to threshold with a < -50
#But how?
#I can show the image of the people with just the bits I want,
#but I'm not sure how to apply that to the forest image.


#Come back to this problem in meeting tomorrow

#plt.imshow(a > -50)

newforest = forest[:375,:500,:]
lab_newforest = color.rgb2lab(newforest)

composite = np.ma.where(lab_greenscreen[:,:,[1]] > -30, lab_greenscreen, lab_newforest)
plt.imshow(color.lab2rgb(composite))


#%%

#Ok! We've gotten a function to do the thing! Now I need to write a loop
#to do the same thing.
#Could use nditer to iterate over the image
#But I'm gonna try a regular loop first.

#Step 1: Make a couple of images
redsquare = np.zeros((7,7,3),dtype=int)
redsquare[...,:] = [255,0,0]
plt.imshow(redsquare)

masksquare = np.zeros_like(redsquare,dtype=int)
masksquare[...,:] = [0,255,0]
masksquare[2:5,2:5,:] = [0,0,255]
plt.imshow(masksquare)

#Step 2: figure out how to iterate through both images at once
target = np.zeros_like(redsquare,dtype=int)
for row in range(target.shape[0]):
    for col in range(target.shape[1]):
        if masksquare[row,col,[1]] > 0:    
            target[row,col,:] = redsquare[row,col,:]
        else:
            target[row,col,:] = masksquare[row,col,:]
plt.imshow(target)
        















