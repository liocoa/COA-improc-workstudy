# -*- coding: utf-8 -*-
"""
A module containing functions for detecting and measuring Andrea's copepod images.

"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from more_itertools import distinct_combinations


def almostEqual(a,b,EPSILON=1e-2):
    """ Return whether a and b are roughly equal, +- EPSILON. """
    return abs(a - b) < EPSILON

def near_in(value, alist):
    """ 
    Iterate through elements in alist and check if |element - value| < threshold for any element in alist.
    """
    for item in alist:
        if almostEqual(value, item):
            return True
    return False

def erode(image):
    """
    Erode a binary image with the element 
    [[1,1,1]
    [1,1,1]
    [1,1,1]]
    
    Returns the eroded image.
    """
    element = morphology.square(width = 3)
    image = morphology.erosion(image,element)
    return image

def dilate(image):
    """
    Dilate a binary image with the element
    [[1,1,1]
    [1,1,1]
    [1,1,1]]
    
    Returns the dilated image.
    """
    element = morphology.square(width = 3)
    image = morphology.dilation(image,element)
    return image


def show(image,cmap = 'gray'):
    """
    Show an image using matplotlib.pyplot.
    
    Parameters
    ----------
    image : array
        An image
    cmap : str, optional
        The color map to use in the display. The default is 'gray'.

    Returns
    -------
    None.

    """
    fig,ax=plt.subplots()
    ax.imshow(image, cmap = cmap)
    plt.show()
    plt.close()


def is_square(combination, epsilon=1):
    """ Return whether or not a set of four lines makes a square, and the side length.
    Each line should be in polar coordinates in the format (angle, dist).
    
    Epsilon here is the amount of difference allowed in the angle of the lines
    to decide it they're parallel, measured in degrees.
    
    
    Return tuple (square, size,)
    square = bool whether or not the lines form a square.
    size = the side length of the square. 0 if not a square.
    
    """
    # Compare the four lines pairwise.
    line_comps = list(distinct_combinations(combination,2))
    n_parallel = 0
    n_perpendicular = 0
    dists = []
    for comp in line_comps:
        # Get the angle between the lines.
        
        # First check the signs on the angles to the origin.       
        # If the sign is the same, take the abs of the difference.
        if np.sign(comp[0][0]) == np.sign(comp[1][0]):
            angle = abs(comp[0][0] - comp[1][0])
        # If the sign is different, sum the abs vals.
        else:
            angle = abs(comp[0][0]) + abs(comp[1][0])

        # We want to know if they're perpendicular or parallel.
        if almostEqual(comp[0][0], comp[1][0], epsilon):
            # They're parallel!
            n_parallel += 1
            # We want to know the distance between these lines.
            dists.append(np.sqrt((comp[0][1] - (comp[1][1]))**2))
        elif almostEqual(angle,np.pi/2, epsilon):
            # They're perpendicular!
            n_perpendicular += 1
        else:
            # These are neither parallel nor perpendicular.
            # This whole combination can be discarded.
            return False, 0

    if n_parallel == 2 and n_perpendicular == 4 and len(dists) == 2:
        # This is a rectangle.
        # Check if it's a square. Allow +/- 5%
        if abs(dists[0] - dists[1]) <= min(dists)*0.05:
            # This is a square. Return True and the size.
            return True, np.mean((dists[0],dists[1]))
    return False, 0


def removeOutliers(x, outlierConstant):
    """
    From https://www.dasca.org/world-of-big-data/article/identifying-and-removing-outliers-using-python-packages
    """

    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []
    for y in a.tolist():
        if y >= quartileSet[0] and y <= quartileSet[1]:
            resultList.append(y)
    return resultList


















