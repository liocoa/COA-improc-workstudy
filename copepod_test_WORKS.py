# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:16:40 2020

@author: Lio

This script finds and measures the copepod in the image 'test01.jpg.' 
It uses a slow algorithm that is difficult to generalize... but it does find
and measure the copepod.
"""

##IMPORTS##
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
from more_itertools import distinct_combinations



# Import the image to be processed
# Store the image name so it can be referenced in the output.
imgname = 'test01.jpg'
image = io.imread('C:/Users/Emily/Desktop/Image Processing/select_copepods/' + imgname)

# Make the image grayscale. Color is unnecessary and difficult or impossible 
# to include in future computations.
image = color.rgb2gray(image)

# Downsample the image to 1/2 size; no necessary detail is lost and this helps
# the code run much more quickly.
image = image[::2,::2]

# First, we find and measure the grid.

# Step 1: Use canny edge detection to get an image that can be processed by
# the Hough transform. The default settings for canny work fine for this.
canny = feature.canny(image)

# Step 2: Detect the gridlines using the Hough transform.

# Do the straight line Hough transform
h, theta, d = transform.hough_line(canny)

# I'd like to save the lines so I'll make a place to put them
lines = []

# This step is complex enough that it helps to get a visual output. 
# We'll start producing a plot now and add to it as we go along.

# The plot will contain two subplots, the first simply the image with the
# lines we're trying to detect, and the second the detected lines and
# intersections overlaid on the input image.

fig, axes = plt.subplots(1,2)
ax = axes.ravel() #Flattens to a 1D array

# Show the canny image in which we're detecting lines.
ax[0].imshow(canny)
ax[0].set_title("Input image")

# Show the canny image in the second subplot as a background for the lines we 
# detect.
ax[1].imshow(canny, cmap=cm.gray)

# Ok so I don't actually fully understand the geometry here, but it works.
# I think this code is based on the example code in the skimage docs. 
# I will come back to trying to explain this later if I have the hours left.
# Point is, origin holds the x0 and x1 of the lines, and the math calculates
# the y0 and y1.
origin = (0,canny.shape[1])
for _, angle, dist in zip(*hough_line_peaks(h,theta,d)):
    y0 = (dist - origin[0] * np.cos(angle)) / np.sin(angle)
    y1 = (dist - origin[1] * np.cos(angle)) / np.sin(angle)
    # Save the line and plot it
    # We know the lines are stored correctly because we can visually check
    # Note plot takes lines in the form (x0,x1,...), (y0,y1,...).
    lines.append((origin,(y0,y1)))
    ax[1].plot(origin,(y0,y1), '-r')
# Plot size and title
ax[1].set_xlim(origin)
ax[1].set_ylim((canny.shape[0],0))
ax[1].set_title("Detected lines")

# Now I want to find the intersections
x_is = []
y_is = []
# And I'll want the slopes later
slopes = []

# Note: This is updated intersection logic. I'll explain it here and use it
# more later.

# Here's the idea: look at all the pairs of lines and check if they intersect.
# If they're not parallel, they must intersect somewhere, so first we check that.
# Then, if they have an instesection, we calculate its location. If the coords
# of the intersection are within the area of the image, then we store the point.

# First make a list containing all the distinct combinations of lines.
# This is a handy function from the more_itertools package.
comparisons = list(distinct_combinations(lines,2))
# Iterate through the comparisons
for comp in comparisons:
    # Line format is ((x0,x1),(y0,y1))
    # Get our first line and name the points that define it so we can use them more easily
    line_a = comp[0]
    y_a1 = line_a[1][0]
    y_a2 = line_a[1][1]
    x_a1 = line_a[0][0]
    x_a2 = line_a[0][1]
    # Get the slope of the first line.
    m_a  = (y_a1 - y_a2)/(x_a1 - x_a2)
    # Store the slope of the first line. We'll use the slopes in the grid measurement logic...
    # But note that that logic is not robust and I wouldn't encourage using it in general.
    slopes.append(m_a)
    # Get the deets on the other line
    line_b = comp[1]
    y_b1 = line_b[1][0]
    y_b2 = line_b[1][1]
    x_b1 = line_b[0][0]
    x_b2 = line_b[0][1]
    m_b  = (y_b1 - y_b2)/(x_b1 - x_b2)
    # Check if the lines are parallel, i.e. their slopes are equal.
    # If they're not parallel we can calculate an intersection without dividing
    # by zero.
    if m_b != m_a:
        # This math is from just from solving the equation for each line for y,
        # setting them equal to each other, and solving for x. Kinda messy.
        x_i = (y_b1 - y_a1 + m_a*x_a1 - m_b*x_b1)/(m_a - m_b)
        y_i = m_a * (x_i - x_a1) + y_a1
        # Check if the intersection is within the image
        if 0 < x_i < image.shape[1] and 0 < y_i < image.shape[0]:
            # If it is, store it!
            x_is.append(x_i)
            y_is.append(y_i)
                
# Select a single intersection - Doesn't really matter which one but this one is
# hardcoded
choice = 5
# Get the x and y of that selected intersection while removing them from the 
# list so we can compare
home_x = x_is.pop(choice)
home_y = y_is.pop(choice)
# Get the distance from home point to each other intersection in the image
# We'll store all the ones we care about
distances = []
# For each intersection...
for x, y in zip(x_is, y_is):
    # Get the distance between it and our chosen intersection.
    dist = np.sqrt((x-home_x)**2 + (y-home_y)**2)
    # Imagine a line between it and our chosen intersection and get its slope
    m = (y - home_y)/(x - home_x)
    # If the line between these two intersections is parallel to one of the gridlines...
    if m in slopes:
        # Then assume the two points are adjacent intersections and the distance
        # between them is the side length of a grid square.
        # I DO NOT ENDORSE THIS LOGIC.
        distances.append(dist)
# The grid size is the mean of all the sizes we stored.
px_per_unit = np.mean(distances)


# Show the lines we found.

ax[1].plot(x_is, y_is, 'go')
plt.tight_layout()
plt.show()  

#%%

# Now it's time for axis detection. The canny image that we made earlier is 
# just too noisy for isolating the copepod, so first we'll rescale the image
# so the copepod is basically the only thing visible.

# I found these thresholds manually. They don't generalize to other images and
# I don't have a way to work them out automatically.
do = exposure.rescale_intensity(image, in_range = (0.3,0.45))

# Make an image showing what the rescale looks like.
fig, (ax0, ax1) = plt.subplots(1,2)
ax0.imshow(image, cmap = 'gray')
ax0.set_title("Original grayscale")
ax1.imshow(do, cmap = 'gray')
ax1.set_title("Rescaled")
plt.show()

# Now do canny on the rescaled image and show the new canny image.
do_edges = feature.canny(do, 10)
plt.imshow(do_edges)



#%%

# Ok so here's the main event. We're gonna find the major axis of the copepod.

# It's nice to know how long this takes.
starttime = time.time()

# Rename that canny image from above
copepod = do_edges

# Name the ranges we need for iterating through the image, just for readability
rows = range(copepod.shape[0])
cols = range(copepod.shape[1])

# Gonna want to plot this
fig, ax = plt.subplots(1,1)

# Game plan: way too many nested for loops.
# We're gonna iterate through all the pixels in the image, and if we get to a
# pixel that is on the edge of the copepod (is True) the we'll iterate through
# all the pixels AGAIN and get the distance from that pixel to each of the other
# True pixels. Then once we've done that a bajillion times we'll just pick the
# longest distance and call that the major axis.

distances = []
terminals = []
# For every pixel in the image
for row in rows:
    for col in cols:
        # If it's on the copepod edge (remember the copepod edge image is boolean)
        if copepod[row, col]:
            #Get the distance from that edge pixel to every other edge pixel
            here = (row, col)
            for r in rows:
                for c in cols:
                    if copepod[r,c]:
                        there = (r,c)
                        # Save the ends of the line segment connecting these points
                        terminals.append(((row,col),(r,c)))
                        # and the distance between them.
                        distance = np.sqrt((row - r)**2 + (col - c)**2)
                        distances.append(distance)
                    
# Find and draw the longest line we measured.
# Get the index of the biggest distance
major_loc = distances.index(max(distances))
# Get the inds of that line from the terminals list
ends = terminals[major_loc]

# Calculate the distance between these end points for some reason?! 
# I legit have no idea what I was thinking with this. Ought to just grab the 
# distance from the distances list!
xs = [ends[0][1],ends[1][1]]
ys = [ends[0][0],ends[1][0]]
copepod_length = np.sqrt((xs[0]-xs[1])**2 + (ys[0]-ys[1])**2)

# Anyway, plot that.
ax.plot(xs,ys)
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







