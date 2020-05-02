# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:33:37 2020

@author: Lio

Trying to make a pipeline just for detecting the gridlines
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
from skimage import feature
from skimage import transform
from skimage import exposure
from skimage import io
from skimage import color
from skimage.transform import hough_line_peaks
import random
import math
#import skdemo

##################
#HELPER FUNCTIONS#
##################
#(These are from copeproc, copied here in case I need to make changes)


def detect_gridlines(canny):
    """Return a list of the gridlines in an image.
    
    Assumes an image that has already been processed with canny.
    Assumes gridlines are present.
    Returns list of lines in the format ((x0,x1),(y0,y1)).
    """
    h, theta, d = transform.hough_line(canny)
    lines = []
    origin = (0,canny.shape[1])
    for _, angle, dist in zip(*hough_line_peaks(h,theta,d)):
        y0 = (dist - origin[0] * np.cos(angle)) / np.sin(angle)
        y1 = (dist - origin[1] * np.cos(angle)) / np.sin(angle)
        lines.append((origin,(y0,y1)))
    return lines


def find_grid_size(lines, imgshape):
    """Return the side length of a grid defined by lines.
    
    Lines should be a list of iterables in the format ((x0,x1),(y0,y1)).
    Shape should be an iterable in the format (rows, cols).
    Assume there are intersections.
    """
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
                if 0 < x_i < imgshape[1] and 0 < y_i < imgshape[0]: #Check it's within the image
                    x_is.append(x_i)
                    y_is.append(y_i)
                    
    #Select a single intersection
    choice = random.choice(range(len(lines)))
    home_x = x_is.pop(choice)
    home_y = y_is.pop(choice)
    #Get the distance from home point to each other point in the image
    distances = []
    for x, y in zip(x_is, y_is):
        dist = np.sqrt((x-home_x)**2 + (y-home_y)**2)
        m = (y - home_y)/(x - home_x)
        if near_in(m, slopes, 0.1):
            distances.append(dist)
    px_per_unit = np.mean(distances)
    
    return px_per_unit

def near_in(value, alist, threshold):
    for item in alist:
        if abs(item - value) < threshold:
            #print(f"Yay, {value} is close to {item}")
            return True
    return False

##############################################################################

# Get the images

imgnames = []
images = []

for img in glob.glob('C:/Users/Emily/Desktop/Image Processing/img_cache/*.jpg'):
    imgnames.append(img[50:])
    image = io.imread(img)
    images.append(color.rgb2gray(image))
    
fails = []
failnames = []
nans = []
nannames = []
suspicious = []
suspiciousnames = []

# Do Canny to all of them

edges = []

for img in images:
    edges.append(feature.canny(img))
    
# Get the gridlines and output
    
output = open(r"grid_sizes.txt", "a")

for img, name in zip(edges,imgnames):
    try:
        lines = detect_gridlines(img)
        scale = find_grid_size(lines,img.shape)
        
        if math.isnan(scale):
            nans.append(img)
            nannames.append(name)
            print("We've got a nan.")
        elif scale < 100 or scale > 650:
            print(f"This one looks funny: {scale:.2f}")
            suspicious.append(img)
            suspiciousnames.append(name)
        else:
            output.write(f"{name}, {scale:.2f} ppm\n")
            print(f"{name}, {scale:.2f}")
        
    except IndexError:
        print("No lines in that one.")
        fails.append(img)
        failnames.append(name)
    except:
        print("unknown error")
        fails.append(img)
        failnames.append(name)
    
nimages = len(images)
nfails = len(fails)
nnans = len(nans)
nsusp = len(suspicious)
successrate = ((nimages - (nfails + nnans + nsusp)) / nimages) * 100
    
    
summary = f"SUMMARY:\nOut of {nimages} total images, {nfails} threw errors and {nnans} returned nan.\n{nsusp} returned sketchy numbers.\nThat's a {successrate:.2f}% success rate.\n"
print(summary)
output.write(summary)
output.close()
    

# skdemo.imshow_all(fails)
# skdemo.imshow_all(nans)





