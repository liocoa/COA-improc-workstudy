# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:21:22 2020

@author: Lio
"""

import time
import copeproc as cp
from skimage import feature,transform
from skimage.transform import hough_line_peaks
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.filters import threshold_triangle
from more_itertools import distinct_combinations


#%%


starttime = time.time()

#Get the pictures

wd = 'C:/Users/Emily/Desktop/Image Processing/select_copepods/'

images, imgnames = cp.get_images(wd)

for image, name in zip(images, imgnames):
    #Do a real in-depth look at every image.
    
    fig, axes = plt.subplots(2,3)
    ax = axes.ravel()
    
    ax[0].imshow(image, cmap = 'gray')
    
    
    thresh = threshold_triangle(image)
    binary = image < thresh
    
    ax[1].imshow(binary)

    #First is canny

    
    canny = feature.canny(binary)
    ax[2].imshow(canny)
    
    #Next we get the lines
    h, theta, d = transform.hough_line(canny)
    lines = []
    origin = (0,canny.shape[1])
    for _, angle, dist in zip(*hough_line_peaks(h,theta,d)):
        y0 = (dist - origin[0] * np.cos(angle)) / np.sin(angle)
        y1 = (dist - origin[1] * np.cos(angle)) / np.sin(angle)
        lines.append((origin,(y0,y1)))
    
    
    #Show the lines
    ax[3].imshow(image, cmap='gray')
    for line in lines:
        ax[3].plot(line[0],line[1], '-r')
         
    
    #Once we have the lines, we can find the intersections
    x_is = []
    y_is = []
    slopes = []
    
    comparisons = list(distinct_combinations(lines,2))
    for comp in comparisons:
        # Line format is ((x0,x1),(y0,y1))
        line_a = comp[0]
        y_a1 = line_a[1][0]
        y_a2 = line_a[1][1]
        x_a1 = line_a[0][0]
        x_a2 = line_a[0][1]
        m_a  = (y_a1 - y_a2)/(x_a1 - x_a2)
        slopes.append(m_a)
        line_b = comp[1]
        y_b1 = line_b[1][0]
        y_b2 = line_b[1][1]
        x_b1 = line_b[0][0]
        x_b2 = line_b[0][1]
        m_b  = (y_b1 - y_b2)/(x_b1 - x_b2)
        if m_b != m_a:
            x_i = (y_b1 - y_a1 + m_a*x_a1 - m_b*x_b1)/(m_a - m_b)
            y_i = m_a * (x_i - x_a1) + y_a1
            if 0 < x_i < image.shape[1] and 0 < y_i < image.shape[0]: #Check it's within the image
                x_is.append(x_i)
                y_is.append(y_i)

                    
    #Show the intersections
    ax[4].imshow(image, cmap='gray')
    for line in lines:
        ax[4].plot(line[0],line[1], '-r')        
    ax[4].plot(x_is, y_is, 'go')                
        
    
    #Select a single intersection
    choice = random.choice(np.arange(1,len(x_is))-1)
    home_x = x_is.pop(choice)
    home_y = y_is.pop(choice)
    
    #Get the distance from home point to each other intersection in the image
    distances = []
    for x, y in zip(x_is, y_is):
        dist = np.sqrt((x-home_x)**2 + (y-home_y)**2)
        m = (y - home_y)/(x - home_x)
        if cp.near_in(m,slopes):
            distances.append(dist)
#####
    px_per_unit = np.mean(distances)
#####
    
    #Show the selected intersection and the found grid radius
    point = (home_x, home_y)
    ax[5].imshow(image, cmap='gray')
    for line in lines:
        ax[5].plot(line[0],line[1], '-r')        
    ax[5].plot(x_is, y_is, 'go')  
    ax[5].plot(point[0],point[1], 'bo')
    circle = mpatches.Circle((point), px_per_unit, ec = 'b', fc = 'none')
    ax[5].add_patch(circle)
    
    
    for a in ax:
        a.set_axis_off()
        #a.set_xlim(line[0])
        a.set_ylim((image.shape[0],0))
    plt.show()
    fig.savefig("line_outputs/"+name)
    plt.close()

