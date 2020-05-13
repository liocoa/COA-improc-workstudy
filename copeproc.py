# -*- coding: utf-8 -*-
"""
A module containing functions for detecting and measuring Andrea's copepod images.

"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, io, color
from skimage import transform
from skimage import exposure
from skimage.transform import hough_line_peaks
import glob
import math
import copy
import random


__all__ = ['show_all','get_images','plot_with_gridlines','detect_gridlines',
           'get_intersections','find_grid_size','get_major_axis',
           'get_copepod_length','length_from_axis']


def near_in(value, alist, threshold):
    """ 
    Iterate through elements in alist and check if |element - value| < threshold for any element in alist.
    
    """
    for item in alist:
        if abs(item - value) < threshold:
            return True
    return False



def show_all(images, title = None, cmap = 'gray'):
    """
    Show all images in a roughly square-ish grid.
    
    Can handle lists of any size including 0.

    Parameters
    ----------
    images : list
        list of images to show
    title : string, optional
        A title for the gris of images. The default is None.
    cmap : string, optional
        A colormap for the images. The default is 'gray'.

    """
    quantity = len(images)
    if quantity > 0:
        rows = int(math.ceil(np.sqrt(quantity)))
        cols = int(math.ceil(quantity/rows))
        
        fig, axes = plt.subplots(rows,cols)
        try:
          ax = axes.ravel()
        except:
          ax = [axes]
        
        fig.suptitle(title)
        
        for n in range(quantity):
            ax[n].imshow(images[n], cmap = cmap)
        for a in ax:
            a.set_axis_off()
        
        plt.show()




def get_images(directory):
    """
    Get all jpg images from a directory and store the images in one list and the images names in another.

    Parameters
    ----------
    directory : str
        The directory name containing the images.

    Returns
    -------
    images : list
        A list of the all the images in directory.
    imgnames : list
        A list of the image names of all the imported images in order.

    """
    imgnames = []
    images = []
    
    for img in glob.glob(directory+'*.jpg'):
        imgnames.append(img[(len(directory)):])
        image = io.imread(img)
        images.append(color.rgb2gray(image))
    
    return images, imgnames



def plot_with_gridlines(image,name,lines):
    """
    Plot a copepod image overlaid with its gridlines and intersections.
    Save image as name; don't show it.


    Parameters
    ----------
    image : array
        copepod image
    name : str
        The name of the image when saved. Can be just the filename to save
        in the current working directory, or can include some filepath
        to save it in a subfolder.
    lines : array
        The lines in the copepod image, found by detect_gridlines().
    """
    xs, ys = get_intersections(lines,image.shape)
    rad, center = find_grid_size(lines,image.shape)
    fig, ax = plt.subplots(1,1)
    ax.imshow(image, cmap='gray')
    for line in lines:
        ax.plot(line[0],line[1], '-r')
    ax.plot(xs, ys, 'go')
    ax.plot()
    ax.set_xlim(line[0])
    ax.set_ylim((image.shape[0],0))
    fig.savefig(name)
    plt.close()



def success(lines):
    """
    I need to know if line detection has given me a good output.
    How shall I do it?
    
    What makes a good output?
        Lines are about perpendicular
        There are at least two intersections among at least three lines
        More than 8 lines is suspect

    """
    #First, check that there are enough lines
    if len(lines) < 3:
        return False
    if len(lines) > 10:
        return False
    
    
    
    return True
    















#%%
    
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

def get_intersections(lines, imgshape):
    x_is = []
    y_is = []
    
    for line in lines:
        y_a1 = line[1][0]
        y_a2 = line[1][1]
        x_a1 = line[0][0]
        x_a2 = line[0][1]
        m_a  = (y_a1 - y_a2)/(x_a1 - x_a2)
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
                if 0 < x_i < imgshape[1] and 0 < y_i < imgshape[0]: #Check it's within the image
                    x_is.append(x_i)
                    y_is.append(y_i)
    return x_is, y_is







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
                if 0 < x_i < imgshape[1] and 0 < y_i < imgshape[0]: #Check it's within the image
                    x_is.append(x_i)
                    y_is.append(y_i)
                    
    #Select a single intersection
    choice = random.choice(np.arange(len(x_is)))
    home_x = x_is.pop(choice)
    home_y = y_is.pop(choice)
    #Get the distance from home point to each other point in the image
    distances = []
    for x, y in zip(x_is, y_is):
        dist = np.sqrt((x-home_x)**2 + (y-home_y)**2)
        m = (y - home_y)/(x - home_x)
        if near_in(m,slopes,0.01):
            distances.append(dist)
    px_per_unit = np.mean(distances)
    
    return px_per_unit, (home_x,home_y)


#%%
    
def get_major_axis(copepod):
    """
    Return the major axis of a copepod.

    Parameters
    ----------
    copepod : 2D image
        Image, preprocessed in canny, outlining copepod.
        Edges must outline only one copepod.

    Returns
    -------
    ends : Tuple
        A tuple containing the coordinates of the endpoints of the major axis of the copepod.
        Format ((y0, x0),(y1, x1)) (yes it's a quirk)

    """
    
    rows = range(copepod.shape[0])
    cols = range(copepod.shape[1])
    
    
    distances = []
    terminals = []
    #Find each edge pixel
    for row in rows:
        for col in cols:
            if copepod[row, col]:
                #Get the distance from that edge pixel to every other edge pixel
                for r in rows:
                    for c in cols:
                        if copepod[r,c]:
                            terminals.append(((row,col),(r,c)))
                            distance = np.sqrt((row - r)**2 + (col - c)**2)
                            distances.append(distance)
                        
    #Find and draw the longest line
    major_loc = distances.index(max(distances))
    ends = terminals[major_loc]
    
    return ends
    
def get_copepod_length(copepod):
    ends = get_major_axis(copepod)
    xs = [ends[0][1],ends[1][1]]
    ys = [ends[0][0],ends[1][0]]
    copepod_length = np.sqrt((xs[0]-xs[1])**2 + (ys[0]-ys[1])**2)
    return copepod_length

def length_from_axis(ends):
    xs = [ends[0][1],ends[1][1]]
    ys = [ends[0][0],ends[1][0]]
    copepod_length = np.sqrt((xs[0]-xs[1])**2 + (ys[0]-ys[1])**2)
    return copepod_length







