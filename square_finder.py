# -*- coding: utf-8 -*-
"""
Created on Sat May 30 13:54:39 2020

@author: Lio

Another attempt at finding the lines. This time I'm planning to repeat the
process I used on the test images in my colab notebook to look at the hough
output visually and see if I can think of anything intelligent to do with it.
I'll also be downsampling the images first, like I totally should have from 
the beginning.
"""

#%% Imports and outline

#Imports
import copeproc as cp
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.feature import canny
import more_itertools
import time
from skimage import io, color
import glob
from scipy import stats


"""
Here's the basic outline.

# Get the image paths
# For each path:
    # Get the image and its tag
    # Downsample
    # Canny
    # Hough lines
    # Count the lines
    # If the count is good:
        # Check for squares
        # If there are squares:
            # Remove outliers
            # Get square size
    # Generate a tracking figure


"""






#%% Get the image paths
file_path = "C:/Users/Emily/Desktop/Image Processing/img_cache/"

img_paths = glob.glob(file_path+'*.jpg')

successes = 0
total = len(img_paths)
starttime = time.time()


# A place to store image paths that worked
successful_images = []
# The square sizes found in the successful images, associated by order
successful_squares = []

#%% For each path
for path in img_paths:
    #%%% Get the image and its tag
    image = color.rgb2gray(io.imread(path))
    tag = path[(len(file_path)):]

    #%%%Downsample and canny

    image = canny(image[::2,::2])
        
    

    #Here's the code from colab - mostly copied from the skimage docs actually.

    
    #%%% Hough
    # Classic straight-line Hough transform
    # Set a precision of 0.5 degree.
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = hough_line(image, theta=tested_angles)
    # Get the peaks from the Hough
    x,angle,ro = hough_line_peaks(h,theta,d)
    
    # Start data string
    data_string = ""
    scale = 1
    
    # Let's start doing something with these lines.
    
    #%%% Count the lines
    # Quantity matters. Let's count them and decide what to do with the image
    # based on the number of lines.
    n_lines = len(angle)
    count_ok = 3 < n_lines < 9
    data_string += (f"lines found: {n_lines}\n")
    data_string += (f"count_ok = {count_ok}\n")
    
    #%%% If the count is good...
    # Now, if we have a decent shot at success, we want to try to find a square.
    if count_ok:
        # Associate angle and distance in a way that will be easier to iterate
        lines = [(a,r) for a,r in zip(angle,ro)]
        #%%%% Check for squares
        # Check each distinct combination of 4 lines to see if they make a square
        combs = list(more_itertools.distinct_combinations(lines, 4))
        data_string += f"{len(combs)} combinations of lines\n"
        # Ok here we go.
        outputs = []
        square_combs = []
        for comb in combs:
            square, size = cp.is_square(comb)
            outputs.append((square, size))
            if square:
                square_combs.append(comb)
       
        #%%%% If there are squares...
        # If any of those combinations made a square, we'll get a True in the output.
        if any([out[0] for out in outputs]):
            
            #%%%%% Measure them
            square_sizes = [out[1] for out in outputs if out[0]]
            size_strs = [f"{x:.1f}" for x in square_sizes]
            data_string += f"Square sizes are {size_strs}\n"
            
            #%%%%% Remove outliers
            # Now it would be good to get rid of outliers, in case of detecting big squares.
            square_sizes = cp.removeOutliers(square_sizes, 1)
            size_strs = [f"{x:.1f}" for x in square_sizes]
            data_string += f"removeOutlier results are {size_strs}\n"
        
        
            #%%%%% Get square size
            # If these sizes don't vary widely, we can use the mean value as our scale!
            if np.var(square_sizes) < 10:
                scale = np.mean(square_sizes)
                data_string += f"scale = {scale:.2f} pixels/mm"
                successes += 1
                
                # If we got this far, we can save the image path and square size
                successful_images.append(path)
                successful_squares.append(scale)
        
        




    
    #%%% Plot the result
    
    
    # # Generate a figure to show what happened
    # fig, axes = plt.subplots()

    # # Plot the detected lines on the canny image
    # axes.imshow(image, cmap=cm.gray)
    # origin = np.array((0, image.shape[1]))
    
    # for _, angle, dist in zip(*(x,angle,ro)):
    #     y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    #     axes.plot(origin, (y0, y1), '-r')
    # axes.set_xlim(origin)
    # axes.set_ylim((image.shape[0], 0))
    # axes.set_axis_off()
    # axes.set_title(f'Detected lines: {tag}')
    
    # try:
    #     square_lines = []
    #     for comb in square_combs:
    #         for line in comb:
    #             square_lines.append(line)
    #     square_angles = [line[0] for line in square_lines]
    #     square_dists = [line[1] for line in square_lines]
    #     for q,p in zip(square_angles, square_dists):
    #         y0, y1 = (p - origin * np.cos(q)) / np.sin(q)
    #         axes.plot(origin, (y0, y1), '-b')
    # except:
    #     pass
    # finally:
    #     square_lines = []
    #     square_combs = []
    
    # axes.text(25,image.shape[1]+30,data_string)
    
    # # Show the image
    # plt.show()
    

#%% Clean the data
    
# I think at this point it's reasonable to remove outliers again...
# Have to make sure the associated paths go too.
    # But for now, let's just remove outliers to see the stats.

successful_squares = cp.removeOutliers(successful_squares, 2)


y = successful_squares
# x = list(range(len(y)))
plt.hist(y,bins=10)
plt.show()

successes = len(successful_squares)

elapsed = time.time() - starttime
success_rate = successes/total*100
print(f"Out of {total} images tested, {successes} found usable squares.\nThat's a {success_rate:.2f}% success rate.\nThe program took {elapsed:.2f} seconds to run, which is an average of {elapsed/total:.3f} seconds per image.")
print(stats.describe(successful_squares))


# The million dollar question now...
# Are these square sizes uniform enough to apply the mean square size to grids
# I can't measure and call it good??
