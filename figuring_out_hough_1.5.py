# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:25:51 2020

@author: Lio

Trying again in the hough department
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import feature
from skimage import color
from skimage import transform
from matplotlib import cm
from skimage.transform import hough_line_peaks


image = io.imread('C:/Users/Emily/Desktop/Image Processing/select_copepods/test_copepod.jpg')
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
origin = np.array((0,canny.shape[1]))
for _, angle, dist in zip(*hough_line_peaks(h,theta,d)):
    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
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

for line in lines:
    y_a1 = line[1][0]
    print(f"Y IS {y_a1}")
    y_a2 = line[1][1]
    x_a1 = line[0][0]
    print(f"X IS {x_a1}")
    x_a2 = line[0][1]
    m_a  = (y_a1 - y_a2)/(x_a1 - x_a2)
    others = lines
    others.remove(line)
    for other in others:
        print(line)
        print(other)
        y_b1 = line[1][0]
        y_b2 = line[1][1]
        x_b1 = line[0][0]
        x_b2 = line[0][1]
        m_b  = (y_b1 - y_b2)/(x_b1 - x_b2)
        x_i = (y_b1 - y_a1 + m_a*x_a1 - m_b*x_b1)/(m_a - m_b)
        y_i = m_a * (x_i - x_a1) + y_a1

        x_is.append(x_i)
        y_is.append(y_i)

print(x_is)
print(y_is)

plt.tight_layout()
plt.show()  

#Why isn't this working?!


