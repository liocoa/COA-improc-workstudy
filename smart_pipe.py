# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:29:40 2020

@author: Lio

copepod pipe with image recognition
"""
#%% Options
##############################################################################
# INPUT AND OUTPUT DIRECTORIES

# Directory for input images (jpgs)
input_dir = "C:/Users/Emily/coding/COA-improc-workstudy/test_images/"

# Directory for output images
img_out_dir = "C:/Users/Emily/coding/COA-improc-workstudy/images_out/smart_test/"

# Directory for output summary table
table_out_dir = "C:/Users/Emily/coding/COA-improc-workstudy/table_out/"

##############################################################################
# OTHER OPTIONS

# Path to model to use for object detection
model_path = "copepod_model.pth"

# Filename for output table
table_name = "smart_copepod_data.csv"

# Show plots
show = True

# Print status
verbose = True

# Number of sample images to select at random (a number or None to run all)
sample_size = 1

##############################################################################
##############################################################################

#%% Imports

import numpy as np
from skimage import filters, color, measure, util
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import cm
import glob
import copy
import random as rand
import more_itertools
import copeproc as cp
import math
from detecto import core, utils, visualize
import time

#%% Set-up

# Get start time
starttime = time.time()

# Get the image paths
img_paths = glob.glob(input_dir+'*.jpg')

# Select a subset of images, if sample_size isn't None
if sample_size:
    img_paths = rand.sample(img_paths,sample_size)

# Open a file for outputting data
output = open(f"{table_out_dir}{table_name}", "w")

# Write headers in the output file
output.write("image name,copepod length (px),copepod width (px),scale (px/mm),comments,confidence\n")

# Load object detection model
model = core.Model.load(model_path,['copepod'])

if verbose:
    print("Successfully loaded model.")
    print(f"Selected images {img_paths}")



# Open the loop
for img_path in img_paths:
    
    # Load in the image
    raw_image = utils.read_image(img_path)
    
    # Get the image name
    img_name = img_path[(len(input_dir)):]
    
    # Zero out our tracking variables
    square_found = False
    scale = "NA"
    copepod_measured = False
    copepod_length = "NA"
    copepod_width = "NA"
    comments = ""
    
    #%% Detect grid
    
    # Get the grid image
    grid = copy.copy(color.rgb2gray(raw_image)) # (depricate after retraining model)
    
    
    # Canny
    
    grid = canny(grid)
    
    # Hough
    
    # Classic straight-line Hough transform
    # Set a precision of 0.5 degree.
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = hough_line(grid, theta=tested_angles)
    
    # Get the peaks from the Hough
    x,angle,ro = hough_line_peaks(h,theta,d)
    
    # Let's start doing something with these lines.
    
    # Count the lines
    
    # Quantity matters. Let's count them and decide what to do with the image
    # based on the number of lines.
    n_lines = len(angle)
    count_ok = 3 < n_lines < 9

    # Check for squares and get the scale
    
    # Now, if we have a decent shot at success, we want to try to find a square.
    if count_ok:
        
        # Associate angle and distance in a way that will be easier to iterate
        lines = [(a,r) for a,r in zip(angle,ro)]
    
        
        # Check each distinct combination of 4 lines to see if they make a square
        combs = list(more_itertools.distinct_combinations(lines, 4))

        # Ok here we go.
        outputs = []
        square_combs = []
        for comb in combs:
            square, size = cp.is_square(comb)
            outputs.append((square, size))
            if square:
                square_combs.append(comb)
                
        # If any of those combinations made a square, we'll get a True in the output.
        if any([out[0] for out in outputs]):
            
            # Measure them
            square_sizes = [out[1] for out in outputs if out[0]]

            # Remove outliers, in case of detecting big squares.
            square_sizes = cp.removeOutliers(square_sizes, 1)

            # Get the scale
            if np.var(square_sizes) < 10:
                
                scale = np.mean(square_sizes)
                square_found = True
                
                
    
    #%% Detect copepod
    
    if verbose:
        print("Detecting copepod...")
    
    # Detect the copepod(s)
    labels, boxes, scores = model.predict_top(raw_image)
    
    if show:
        visualize.show_labeled_image(raw_image, boxes, labels)
        
    if verbose:
        print("Done.")
        print(f"Found {len(boxes)} copepods.")
    
    # Get the detected bounding box(es) of the detected copepod(s) as a numpy array
    boxes = np.rint(boxes.numpy()).astype(int)
    
    # Open boxes loop in case of more than one copepod
    for box in boxes:
        
        #%% Process/measure copepod
        
        # Get the edges of our found box explicitly
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]
        
        # Get copepod (and get it greyscale) (depricate after retraining model)
        b_box = color.rgb2gray(raw_image[ymin:ymax,xmin:xmax])
        
        # denoise (depricate after retraining model)
        denoised = ndi.median_filter(util.img_as_float(b_box), size=5)

        copepod = denoised
        
        # Show
        if show:
            cp.show(copepod)
    
        

        # Run triangle filter
        thresh = filters.threshold_triangle(copepod)
        copepod = copepod < thresh
        
        
        # Trim the edges so erode will play nicer
        # Top and bottom
        copepod[0:1] = 0
        copepod[-1:] = 0
        # Left and right
        copepod[:,0] = 0
        copepod[:,-1] = 0
        
        
        
        # Show
        if show:
            cp.show(copepod)
    

        # Erode
        
        # Erode/dilate count - 1/10 of the length of the shortest rectangle dimension
        er_dil = math.ceil(min(ymax-ymin, xmax-xmin)/10)
        
        if verbose:
            print(f"erode/dilate count = {er_dil}")
        
        for n in range(er_dil):
            copepod = cp.erode(copepod)
        
        # Show
        if show:
            cp.show(copepod)
        

        # Now we label the regions        

        # Image to int from bool
        copepod = copepod.astype(int)
        
        # Label regions
        labels, count = ndi.label(copepod)
        copepod[:] = labels

        
        # Get clear on what the edges of the main image are
        n_rows = np.shape(copepod)[0]
        n_cols = np.shape(copepod)[1]

        
        try:

            # Dilate the appropriate number of times
            for n in range(er_dil):
                copepod = cp.dilate(copepod)
                
            # Show
            if show:
                cp.show(copepod)
            
            # Measure the blob
            
            regions = measure.regionprops(copepod)
            
            # There's only one set of properties in this list, by construction
            if len(regions) == 1:
                props = regions[0]
            else:
                raise ValueError("Too many regions")
            
            copepod_length = props.major_axis_length
            copepod_width = props.minor_axis_length
            
            # Report copepod find
            copepod_measured = True
            
        except ValueError:
            pass
        
        
        
        #%% Make qa/qc image
        # Generate a figure to show what happened
        fig, axes = plt.subplots()
        
        # Show the original image
        axes.imshow(raw_image, cmap=cm.gray)
        
        # Overlay the detected gridlines
        origin = np.array((0, raw_image.shape[1]))
        
        for _, angle, dist in zip(*(x,angle,ro)):
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            axes.plot(origin, (y0, y1), '-r')
        axes.set_xlim(origin)
        axes.set_ylim((grid.shape[0], 0))
        axes.set_axis_off()
        axes.set_title(f'QA/QC: {img_name}')
        
        try:
            square_lines = []
            for comb in square_combs:
                for line in comb:
                    square_lines.append(line)
            square_angles = [line[0] for line in square_lines]
            square_dists = [line[1] for line in square_lines]
            for q,p in zip(square_angles, square_dists):
                y0, y1 = (p - origin * np.cos(q)) / np.sin(q)
                axes.plot(origin, (y0, y1), '-b')
        except:
            pass
        finally:
            square_lines = []
            square_combs = []
            
        # Overlay the copepod measurement
    
        if copepod_measured:
            y0, x0 = props.centroid
            orientation = props.orientation
            
            ellipse = patches.Ellipse((x0+xmin,y0+ymin), 
                                      props.minor_axis_length, 
                                      props.major_axis_length,
                                      -np.rad2deg(orientation),
                                      fill = False,
                                      linestyle = "-",
                                      edgecolor = "green",
                                      linewidth = 3)
              
            axes.add_artist(ellipse)
        
        # Show the image
        # plt.savefig(f"{img_out_dir}qaqc{img_name}")
    
        ####################
        # Show for debugging
        plt.show()
        ####################    
        
        plt.close(fig)
        
        #%% Write results to file

        # Generate comments and turn measured values to strings
        if square_found:
            scale = f"{scale:.1f}"
        else:
            comments = comments+"no grid "
            
        if copepod_measured:

            copepod_length = f"{copepod_length:.1f}"
            copepod_width = f"{copepod_width:.1f}"
        else:
            comments = comments+"no copepod"

            
        if square_found and copepod_measured:
            comments = "NA"
            
        output.write(f"{img_name},{copepod_length},{copepod_width},{scale},{comments}\n")
    
        ######################################
        # Print for debugging
        print(f"{img_name},{copepod_length},{copepod_width},{scale},{comments}")
        ######################################
    
    
