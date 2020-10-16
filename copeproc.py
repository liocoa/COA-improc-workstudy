# -*- coding: utf-8 -*-
"""
A module containing functions for detecting and measuring Andrea's copepod images.

"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform
from skimage.transform import hough_line_peaks
import glob
import math
import copy
import random
from more_itertools import distinct_combinations


####################
# ACTIVE FUNCTIONS #
####################



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
        
        fig, axes = plt.subplots(rows,cols, figsize = (rows*7, cols*7))
        try:
          ax = axes.ravel()
        except:
          ax = [axes]
        
        fig.suptitle(title,fontsize=rows*20)
        
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
        The directory name containing the images. Should end with a slash.

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












######################
# ARCHIVAL FUNCTIONS #
######################
# Traverse at your own risk.
    

def hough_grid_peaks(h,theta,ro,min_distance=5):
    """
    An attempt to make a version of hough_line_peaks that detects only lines
    in a grid. Actually detects all lines that are either parallel or
    perpendicular to any other line.

    Parameters
    ----------
    h : array
        output from hough_line
    theta : array
        output from hough_line
    ro : array
        output from hough_line
    min_distance : TYPE, optional
        The minimum number of pixels between lines. The default is 5.

    Returns
    -------
    _h : TYPE
        the same h and above
    angles
        angles for the detected lines
    distnaces
        ro values for the detected lines

    """
    #Get the regular peaks
    _h, angle, dist = hough_line_peaks(h,theta,ro,min_distance=min_distance)
    #Get these points associated
    points = [(angle[n], dist[n]) for n in range(len(angle))]
    points.sort()
    #Pick points to try to fit
    candidates = []
    for point in points:
      others = copy.copy(points)
      others.remove(point)
      for other in others:
        diff = abs(point[0]) - abs(other[0])
        if almostEqual(diff,0) or almostEqual(diff,np.pi / 2):
          candidates.append(point)
          break
    return (_h,[p[0] for p in candidates],[p[1] for p in candidates])






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
    

def get_intersections_2(lines, imgshape):
    x_is = []
    y_is = []
    
    comparisons = list(distinct_combinations(lines,2))
    for comp in comparisons:
        # Line format is ((x0,x1),(y0,y1))
        line_a = comp[0]
        y_a1 = line_a[1][0]
        y_a2 = line_a[1][1]
        x_a1 = line_a[0][0]
        x_a2 = line_a[0][1]
        m_a  = (y_a1 - y_a2)/(x_a1 - x_a2)
        line_b = comp[1]
        y_b1 = line_b[1][0]
        y_b2 = line_b[1][1]
        x_b1 = line_b[0][0]
        x_b2 = line_b[0][1]
        m_b  = (y_b1 - y_b2)/(x_b1 - x_b2)
        if m_b != m_a:
            x_i = (y_b1 - y_a1 + m_a*x_a1 - m_b*x_b1)/(m_a - m_b)
            y_i = m_a * (x_i - x_a1) + y_a1
            if 0 < x_i < imgshape[1] and 0 < y_i < imgshape[0]: #Check it's within the image
                x_is.append(x_i)
                y_is.append(y_i)
    
    
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







