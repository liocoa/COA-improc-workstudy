# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:13:07 2020

@author: Lio

Developing the bits of the copepod test that work
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import img_as_float
import skimage.filters as filters
from skimage import feature
from skimage import color
from skimage import transform
from skimage import exposure
import time
from matplotlib import cm
from skimage.transform import hough_line_peaks


image = io.imread('C:/Users/Emily/Desktop/Image Processing/select_copepods/test_copepod.jpg')
image = color.rgb2gray(image)

#%%
#Get line-detectable edges

sigma = 2
canny = feature.canny(image,sigma)
plt.imshow(canny)

#Detect gridlines
#Straight line hough
h, theta, d = transform.hough_line(canny)

#I wanna see the result so now I get to learn about generating figures
fig, axes = plt.subplots(1,2)
ax = axes.ravel() #What?

ax[0].imshow(canny)
ax[0].set_title("Input image")
ax[0].set_axis_off()

ax[1].imshow(canny, cmap=cm.gray)
origin = np.array((0,canny.shape[1]))
for _, angle, dist in zip(*hough_line_peaks(h,theta,d)):
    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    ax[1].plot(origin,(y0,y1), '-r')
ax[1].set_xlim(origin)
ax[1].set_ylim((canny.shape[0],0))
ax[1].set_axis_off()
ax[1].set_title("Detected lines")

plt.tight_layout()
plt.show()    

#%%

#Get edges for axis detection

do = exposure.rescale_intensity(image, in_range = (0.3,0.45))

fig, (ax0, ax1) = plt.subplots(1,2)
ax0.imshow(image, cmap = 'gray')
ax0.set_title("Original grayscale")
ax1.imshow(do, cmap = 'gray')
ax1.set_title("Rescaled")
plt.show()
do_edges = feature.canny(do, 10)
edges_clipped = do_edges[125:300, 150:500]
plt.imshow(edges_clipped)





#%%
#Find the major axis of the copepod

starttime = time.time()
copepod = edges_clipped

rows = range(copepod.shape[0])
cols = range(copepod.shape[1])

#checkiter = np.zeros_like(copepod)

fig, ax = plt.subplots(1,1)

distances = []
terminals = []
#Find each edge pixel
for row in rows:
    for col in cols:
        if copepod[row, col]:
            #Get the distance from that edge pixel to every other edge pixel
            here = (row, col)
            for r in rows:
                for c in cols:
                    if copepod[r,c]:
                        there = (r,c)
                        terminals.append(((row,col),(r,c)))
                        distance = np.sqrt((row - r)**2 + (col - c)**2)
                        distances.append(distance)
                    
#Find and draw the longest line
major = distances.index(max(distances))
ends = terminals[major]

xs = [ends[0][1],ends[1][1]]
ys = [ends[0][0],ends[1][0]]

ax.plot(xs,ys)
ax.plot(129,310)
ax.imshow(copepod)
plt.show()

            
elapsedtime = time.time() - starttime
    
print(f"Finding the copepod axis took {elapsedtime:.2f} seconds.")


#%%
#Now I have to figure out how to measure the grid.
#So how do we do that?
#I've got to figure out how to identify locations in the the hough output.
#I... don't understand the hough output.



