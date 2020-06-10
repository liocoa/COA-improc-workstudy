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

#Imports
import copeproc as cp
import numpy as np
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.feature import canny
import more_itertools
import time


#Get the images I want to use
file_path = "C:/Users/Emily/Desktop/Image Processing/select_copepods/"
images, imgnames = cp.get_images(file_path)

#Downsample to 1/2 size and canny
for n in range(len(images)):
    images[n] = canny(images[n][::2,::2])
    
successes = 0
total = len(images)
starttime = time.time()

#Here's the code from colab - mostly copied from the skimage docs actually.
for image, tag in zip(images,imgnames):
    
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
    
    # Quantity matters. Let's count them and decide what to do with the image
    # based on the number of lines.
    n_lines = len(angle)
    count_ok = 3 < n_lines < 9
    data_string += (f"lines found: {n_lines}\n")
    data_string += (f"count_ok = {count_ok}\n")
    
    # Now, if we have a decent shot at success, we want to try to find a square.
    if count_ok:
        # Associate angle and distance in a way that will be easier to iterate
        lines = [(a,r) for a,r in zip(angle,ro)]
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
        # If any of those combinations made a square, we'll get a True in the output.
        # Let's keep them.
        square_sizes = [out[1] for out in outputs if out[0]]
        data_string += f"Square sizes are {square_sizes}\n"
        
        # If these sizes don't vary widely, we can use the mean value as our scale!
        if np.var(square_sizes) < 10:
            scale = np.mean(square_sizes)
            data_string += f"scale = {scale:.2f} pixels/mm"
            successes += 1
        
            
    
    
    
    
    # Generate a figure to show what happened
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()
    
    # Plot the canny image  
    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title(f'Input image: {tag}')
    ax[0].set_axis_off()
    
    # Plot the hough output and detected points
    ax[1].imshow(np.log(1 + h),
                extent=[np.rad2deg(theta[-1 ]), np.rad2deg(theta[0]), d[-1], d[0]],
                cmap=cm.gray)
    ax[1].scatter(np.rad2deg(angle)*(-1),ro,c='r')
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')
    
    # Plot the detected lines on the canny image
    ax[2].imshow(image, cmap=cm.gray)
    origin = np.array((0, image.shape[1]))
    
    for _, angle, dist in zip(*(x,angle,ro)):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        ax[2].plot(origin, (y0, y1), '-r')
    ax[2].set_xlim(origin)
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')
    
    try:
        square_lines = []
        for comb in square_combs:
            for line in comb:
                square_lines.append(line)
        square_angles = [line[0] for line in square_lines]
        square_dists = [line[1] for line in square_lines]
        for q,p in zip(square_angles, square_dists):
            y0, y1 = (p - origin * np.cos(q)) / np.sin(q)
            ax[2].plot(origin, (y0, y1), '-b')
    except:
        pass
    finally:
        square_lines = []
        square_combs = []
    
    ax[2].text(25,image.shape[1]+30,data_string)
    
    # Save it and show it
    plt.savefig(f"{tag}.jpg")
    plt.show()
    
    print("DONE WITH THIS ONE")

elapsed = time.time() - starttime
success_rate = successes/total*100
print(f"Out of {total} images tested, {successes} found usable squares.\nThat's a {success_rate:.2f}% success rate.\nThe program took {elapsed:.2f} seconds to run, which is an average of {elapsed/total:.3f} seconds per image.")
  
#%%
""" 
Let's start to imagine some kind of algorithm.
  
    1. Check if it's good.
        - It's good if you can find a perfect square AND there's not an absurd
            number of lines.
            CHECKING FOR GOOD
            - Count the lines
                - There should be at least 4, and no more than like 6 or 8.
            
            
        - Good ones don't need any more pre-processing.
        - Not good ones get set aside to be dealt with later.
    2. Handle the ones that need help.
        WAYS TO NOT HAVE A SQUARE
        - Too many lines
            - Might be blurry or need rescaling
        - Too few lines
            - Still might be blurry or need rescaling
        - Crucial line obstructed somehow
            - This will take some other kind of logic...
            - Maybe check for right angles.
    

"""
#%%

# Let's start on some methods to identify success.
