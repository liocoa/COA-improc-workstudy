# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:16:40 2020

@author: Lio

Can we get our copepod info working???
"""


import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import feature
from skimage import color
from skimage import transform
from skimage import exposure
import time
from matplotlib import cm
from skimage.transform import hough_line_peaks



imgname = 'test_copepod.jpg'

image = io.imread('C:/Users/Emily/Desktop/Image Processing/select_copepods/' + imgname)
image = color.rgb2gray(image)

#%%
#Get line-detectable edges

sigma = 2
canny = feature.canny(image,sigma)

#Detect gridlines
#Straight line hough
h, theta, d = transform.hough_line(canny)

#I'd like to save the lines so I'll make a place to put them
lines = []

#I wanna see the result so now I get to learn about generating figures
fig, axes = plt.subplots(1,2)
ax = axes.ravel() #Flattens to a 1D array

ax[0].imshow(canny)
ax[0].set_title("Input image")

ax[1].imshow(canny, cmap=cm.gray)
origin = (0,canny.shape[1])
for _, angle, dist in zip(*hough_line_peaks(h,theta,d)):
    y0 = (dist - origin[0] * np.cos(angle)) / np.sin(angle)
    y1 = (dist - origin[1] * np.cos(angle)) / np.sin(angle)
    #Save the line and plot it
    #We know the lines are stored correctly because we can visually check
    lines.append((origin,(y0,y1)))
    ax[1].plot(origin,(y0,y1), '-r')
ax[1].set_xlim(origin)
ax[1].set_ylim((canny.shape[0],0))
ax[1].set_title("Detected lines")

#Now I want to find the intersections
x_is = []
y_is = []
#And I'll want the slopes later
slopes = []

for line in lines:
    y_a1 = line[1][0]
    y_a2 = line[1][1]
    x_a1 = line[0][0]
    x_a2 = line[0][1]
    m_a  = (y_a1 - y_a2)/(x_a1 - x_a2)
    slopes.append(m_a)
    others = lines
    others.remove(line)
    for other in others:
        y_b1 = other[1][0]
        y_b2 = other[1][1]
        x_b1 = other[0][0]
        x_b2 = other[0][1]
        m_b  = (y_b1 - y_b2)/(x_b1 - x_b2)
        if m_b != m_a:
            x_i = (y_b1 - y_a1 + m_a*x_a1 - m_b*x_b1)/(m_a - m_b)
            y_i = m_a * (x_i - x_a1) + y_a1
            if 0 < x_i < canny.shape[1] and 0 < y_i < canny.shape[0]: #Check it's within the image
                x_is.append(x_i)
                y_is.append(y_i)
                
#Select a single intersection
choice = 5
home_x = x_is.pop(choice)
home_y = y_is.pop(choice)
#Get the distance from home point to each other point in the image
distances = []
for x, y in zip(x_is, y_is):
    dist = np.sqrt((x-home_x)**2 + (y-home_y)**2)
    m = (y - home_y)/(x - home_x)
    print(m)
    if m in slopes:
        distances.append(dist)
print(distances)
px_per_unit = np.mean(distances)




ax[1].plot(x_is, y_is, 'go')
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
major_loc = distances.index(max(distances))
ends = terminals[major_loc]

xs = [ends[0][1],ends[1][1]]
ys = [ends[0][0],ends[1][0]]

copepod_length = np.sqrt((xs[0]-xs[1])**2 + (ys[0]-ys[1])**2)

ax.plot(xs,ys)
ax.plot(129,310)
ax.imshow(copepod)
plt.show()

            
elapsedtime = time.time() - starttime
    
print(f"Finding the copepod axis took {elapsedtime:.2f} seconds.")



#%%


#RESULTS!

output = open(r"copepod_data.txt", "a")
length = copepod_length/px_per_unit
print(f"The detected copepod is {length} units long. \nCongratulations!")
output.write(f"{imgname}: {px_per_unit:.2f} pixels per mm, copepod length {length:.2f} mm\n")


output.close()







