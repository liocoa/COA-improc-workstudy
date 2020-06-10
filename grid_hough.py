# -*- coding: utf-8 -*-
"""
Attempting to isolate a grid from detected lines in copepod images by selecting
points from the hough_line output that resemble a grid pattern.
"""

#########
#IMPORTS#
#########
import numpy as np
import copy

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage import io,color

import matplotlib.pyplot as plt
from matplotlib import cm

from skimage.draw import line
import copeproc as cp


##################
#HELPER FUNCTIONS#
##################


def hough_grid_peaks(h,angle,dist):
    """ This is the same as in copeproc. I'll explain the logic here. """
    # The angles and distances defining each line are currently in different 
    # lists, like this: [angle0, angle1, ...], [dist0, dist1, ...], but I want
    # to have them associated with each other by line instead, like this:
    # [(angle0, dist0), (angle1, dist1), ...] in order to compare them more
    # intuitively. So we'll do that first.
    points = [(angle[n], dist[n]) for n in range(len(angle))]
    points.sort()
    # Pick points to try to fit
    candidates = []
    # Go through all the points in Hough space...
    for point in points:
        others = copy.copy(points)
        others.remove(point)
        # And compare them to all the other points in Hough space...
        # (Note I'm not using distinct_combinations() here - that's because I 
        # wrote this code before discovering that gem.)
        for other in others:
            # What is the difference between the angles
            diff = abs(point[0]) - abs(other[0])
            # If the lines to which these points refer are parallel or perpendicular...
            if cp.almostEqual(diff,0) or cp.almostEqual(diff,np.pi / 2):
                # Keep this line and move on.
                candidates.append(point)
                break
            return (h,[p[0] for p in candidates],[p[1] for p in candidates])

# First create an artificial grid with some extraneous lines as proof of concept.
images = []
tags = []

tag = "rotated_grid_with_intruders"
image = np.zeros((200, 200))
#Vertical lines
rr, cc = line(0, 30, 199, 70)
image[rr, cc] = 1
rr, cc = line(0, 130, 199, 170)
image[rr,cc] = 1
rr, cc = line(0, 80, 199, 120)
image[rr, cc] = 1
#Horizontal lines
rr, cc = line(70,0,30,199)
image[rr, cc] = 1
rr, cc = line(170,0,130,199)
image[rr,cc] = 1
rr, cc = line(120,0,80,199)
image[rr,cc] = 1
#Intruders
rr,cc = line(0,154,199,12)
image[rr,cc] = 1
rr,cc = line(16,0,108,199)
image[rr,cc] = 1

images.append(image)
tags.append(tag)

#############################################


# Snag a bunch of real images.

imgs, tgs = cp.get_images('C:/Users/Emily/Desktop/Image Processing/select_copepods/')
for image in imgs:
    images.append(canny(color.rgb2gray(image)))
for tg in tgs:
    tags.append(tg)


#%%


#This bit plots all the tests and their hough outputs


for image, tag in zip(images,tags):

  # Classic straight-line Hough transform
  # Set a precision of 0.5 degree.
  tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
  h, theta, d = hough_line(image, theta=tested_angles)

  # Create a plot to show the image, the lines we find, and the Hough space
  fig, axes = plt.subplots(1, 3, figsize=(15, 6))
  ax = axes.ravel()

  # Here's the input image
  ax[0].imshow(image, cmap=cm.gray)
  ax[0].set_title(f'Input image: {tag}')
  ax[0].set_axis_off()

  # Here's the Hough space - don't ask me how this works, I got it from the 
  # skimage docs.
  ax[1].imshow(np.log(1 + h),
              extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
              cmap=cm.gray, aspect=1/1.5)
  ax[1].set_title('Hough transform')
  ax[1].set_xlabel('Angles (degrees)')
  ax[1].set_ylabel('Distance (pixels)')
  ax[1].axis('image')

  # And here's the detected lines. Same code for this as always.
  ax[2].imshow(image, cmap=cm.gray)
  origin = np.array((0, image.shape[1]))
  x,y,z = hough_line_peaks(h, theta, d)
  for _, angle, dist in zip(*hough_grid_peaks(x,y,z)):
      y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
      ax[2].plot(origin, (y0, y1), '-r')
  ax[2].set_xlim(origin)
  ax[2].set_ylim((image.shape[0], 0))
  ax[2].set_axis_off()
  ax[2].set_title('Detected lines')

  # Show it
  plt.tight_layout()
  plt.savefig(f"{tag}.jpg")
  plt.show()

# Maybe instead of seeing the detected lines we want to see the points
# detected in Hough space. The below code does that.

for image, tag in zip(images,tags):
  #Do Hough to it
  tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
  h, theta, ro = hough_line(image, theta=tested_angles)

  #Get the peaks as selected by hough_line_peaks
  intensity, angle, dist = hough_line_peaks(h,theta,ro)
  
  # Get the peaks as selected by hough_grid_peaks
  _, xgrid, ygrid = hough_grid_peaks(intensity, angle, dist)

  #Plot the points found and the points that grid_peaks will select
  fig,ax = plt.subplots(1,2,figsize=(11,4))
  ax[0].scatter(angle,dist)
  ax[0].set_xlim(-np.pi/2 - 0.1,np.pi/2 + 0.1)
  ax[0].set_ylim(-200,200)
  ax[0].set_title(f"{tag} raw")
  ax[1].scatter(xgrid,ygrid)
  ax[1].set_xlim(-np.pi/2 - 0.1,np.pi/2 + 0.1)
  ax[1].set_ylim(-200,200)
  ax[1].set_title(f"{tag} select")