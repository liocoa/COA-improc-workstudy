# -*- coding: utf-8 -*-
"""
Created on Sun May 24 10:51:55 2020

@author: Lio

This code explores the exposure module in skimage, particularly contrast 
stretching using skimage.exposure.rescale_intensity.
"""
import copeproc as cp
import matplotlib.pyplot as plt
from skimage import exposure,feature

import numpy as np


# Get some images
wd = 'C:/Users/Emily/Desktop/Image Processing/select_copepods/'

images, imgnames = cp.get_images(wd)



cp.show_all(images, "Raw")

# #%%
# #################
# #GAMMA TRANSFORM#
# #################
# gamma = []
# gammas = [3,2,1/2,1/4]
# for g in gammas:
#     for image in images:
#         gamma.append(exposure.adjust_gamma(image,g))
#     cp.show_all(gamma, f"Gamma transform g = {g}")
#     gamma = []
    
# #%%
# ###############
# #LOG TRANSFORM#
# ###############
# log = []
# gains = [3,2,1/2,1/4]
# for g in gains:
#     for image in images:
#         log.append(exposure.adjust_log(image,g,True))
#     cp.show_all(log, f"Log transform gain = {g}")
#     log = []   


# #%%
# ###################
# #SIGMOID TRANSFORM#
# ###################
# sigmoid = []
# gains = [0]
# for g in gains:
#     for image in images:
#         sigmoid.append(exposure.adjust_sigmoid(image,inv=True))
#     cp.show_all(sigmoid, f"Sigmoid transform")
#     sigmoid = []   


###################
#RESCALE INTENSITY#
###################

# imgs = []
# ranges = [(2,80)]
# for r in ranges:
#     for img in images:
#         p2, p98 = np.percentile(img, r)
#         img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
#         imgs.append(img_rescale)
#     cp.show_all(imgs, f"Rescaled to {r}")
#     imgs = []




#RESCALE SINGLE IMAGE MANY TIMES#
    
img = images[9]
lows = [0,2,8,10,13,15,20,30]
highs = [100,98,92,90,87,85,80,70]
out = []
captions = []
for l in lows:
    for h in highs:
        pl, ph = np.percentile(img, (l,h))
        out.append(exposure.rescale_intensity(img, in_range=(pl,ph)))
        captions.append(f"{l,h}")
quantity = len(out)
if quantity > 0:
    rows = len(lows)
    cols = len(highs)
    
    fig, axes = plt.subplots(rows,cols, figsize = (rows*7, cols*7))
    try:
      ax = axes.ravel()
    except:
      ax = [axes]
    for n in range(quantity):
        ax[n].imshow(out[n], cmap = 'gray')
        ax[n].set_title(captions[n])
    for a in ax:
        a.set_axis_off()
    
    plt.show()

#Now do canny to them
canny = []
for img in out:
    canny.append(feature.canny(img))
quantity = len(out)
if quantity > 0:
    rows = len(lows)
    cols = len(highs)
    
    fig, axes = plt.subplots(rows,cols, figsize = (rows*7, cols*7))
    try:
      ax = axes.ravel()
    except:
      ax = [axes]
    for n in range(quantity):
        ax[n].imshow(canny[n])
        ax[n].set_title(captions[n])
    for a in ax:
        a.set_axis_off()
    
    plt.show()