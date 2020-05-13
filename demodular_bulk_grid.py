# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:21:22 2020

@author: Lio
"""

import time
import copeproc as cp
from skimage import feature,transform
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



starttime = time.time()

#Get the pictures

wd = 'C:/Users/Emily/Desktop/Image Processing/img_cache/'

images, imgnames = cp.get_images(wd)

for image, name in zip(images, imgnames):
    #Do a real in-depth look at every image.
    
    # try:
    fig, axes = plt.subplots(2,3)
    ax = axes.ravel()
    
    ax[0].imshow(image, cmap = 'gray')
    

    #First is canny
    canny = feature.canny(image)
    ax[1].imshow(canny)
    
    #Next we get the lines (cp.detect_gridlines())
    h, theta, d = transform.hough_line(canny)
    lines = []
    origin = (0,canny.shape[1])
    for _, angle, dist in zip(*transform.hough_line_peaks(h,theta,d)):
        y0 = (dist - origin[0] * np.cos(angle)) / np.sin(angle)
        y1 = (dist - origin[1] * np.cos(angle)) / np.sin(angle)
        lines.append((origin,(y0,y1)))
    
    
    #Show the lines
    ax[2].imshow(image, cmap='gray')
    for line in lines:
        ax[2].plot(line[0],line[1], '-r')
    
    #Once we have the lines, we can find the intersections
    x_is = []
    y_is = []
    slopes = []
    
    for line in lines:
        y_a1 = line[1][0]
        y_a2 = line[1][1]
        x_a1 = line[0][0]
        x_a2 = line[0][1]
        m_a  = (y_a1 - y_a2)/(x_a1 - x_a2)
        slopes.append(m_a)
        others = copy.deepcopy(lines)
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
                if 0 < x_i < image.shape[1] and 0 < y_i < image.shape[0]: #Check it's within the image
                    x_is.append(x_i)
                    y_is.append(y_i)
                    
    #Show the intersections
    ax[3].imshow(image, cmap='gray')
    for line in lines:
        ax[3].plot(line[0],line[1], '-r')        
    ax[3].plot(x_is, y_is, 'go')                
        
    
    #Select a single intersection
    choice = random.randint(0,len(x_is)-1)
    home_x = x_is.pop(choice)
    home_y = y_is.pop(choice)
    #Get the distance from home point to each other point in the image
    distances = []
    for x, y in zip(x_is, y_is):
        dist = np.sqrt((x-home_x)**2 + (y-home_y)**2)
        m = (y - home_y)/(x - home_x)
        if cp.near_in(m,slopes,0.01):
            distances.append(dist)
#####
    px_per_unit = np.mean(distances)
#####
    
    #Show the selected intersection and the found grid radius
    point = (home_x, home_y)
    ax[4].imshow(image, cmap='gray')
    for line in lines:
        ax[4].plot(line[0],line[1], '-r')        
    ax[4].plot(x_is, y_is, 'go')  
    ax[4].plot(point[0],point[1], 'bo')
    circle = mpatches.Circle((point), px_per_unit, ec = 'b', fc = 'none')
    ax[4].add_patch(circle)
        
    # except:
    #     print("oops")
        
    # finally:
        
    for a in ax:
        a.set_axis_off()
        a.set_xlim(line[0])
        a.set_ylim((image.shape[0],0))
    plt.show()
    fig.savefig("line_outputs/"+name)
    plt.close()

