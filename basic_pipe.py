# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 07:29:21 2020

@author: Lio

Full copepod detection loop
"""
#%% I/O directories
##############################################################################
# INPUT AND OUTPUT DIRECTORIES
##############################################################################
# Directory for input images (jpgs)
input_dir = "C:/Users/Emily/coding/COA-improc-workstudy/img_cache/"

# Directory for output images
img_out_dir = "C:/Users/Emily/coding/COA-improc-workstudy/images_out/"

# Directory for output summary table
table_out_dir = "C:/Users/Emily/coding/COA-improc-workstudy/table_out/"

# Filename for output table
table_name = "copepod_data.csv"

##############################################################################
#                                                                            #
# PLEASE RUN IMAGES THROUGH PRE_PIPE_COMPRESSION.PY BEFORE RUNNING THIS CODE #
#                                                                            #
##############################################################################

#%% Imports

import numpy as np
from skimage import filters, io, measure
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import cm
import glob
import copy
import more_itertools
import copeproc as cp

import time

#%% Pre-loop stuff
    
starttime = time.time()

# Get the image paths
img_paths = glob.glob(input_dir+'*.jpg')

# Open a file for outputting data
output = open(f"{table_out_dir}{table_name}", "w")


# Write headers in the output file
output.write("image name,copepod length (px),copepod width (px),scale (px/mm),comments,accuracy guess\n")


#%% Open the loop
    
for path in img_paths:

    # Get the image
    image = io.imread(path)
        
    # Get the image name
    img_name = path[(len(input_dir)):]

    
    # This is the last time we do something to "image". Everything else is a copy.

    # Zero out our tracking variables
    square_found = False
    scale = "NA"
    copepod_measured = False
    copepod_length = "NA"
    copepod_width = "NA"
    comments = ""
    

    #%% GRID PIPELINE
    
    # Copy the image so I can return to the original later
    grid = copy.copy(image)
    
    # Canny
    grid = canny(grid)
    
    #%%% Hough
    
    # Classic straight-line Hough transform
    # Set a precision of 0.5 degree.
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = hough_line(grid, theta=tested_angles)
    
    # Get the peaks from the Hough
    x,angle,ro = hough_line_peaks(h,theta,d)
    
    # Let's start doing something with these lines.
    
    #%%% Count the lines
    
    # Quantity matters. Let's count them and decide what to do with the image
    # based on the number of lines.
    n_lines = len(angle)
    count_ok = 3 < n_lines < 9

    #%%% Check for squares and get the scale
    
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


    #%% COPEPOD MEASURMENT

    copepod = copy.copy(image)
    
    #%%% Triangle filter
    
    thresh = filters.threshold_triangle(copepod)
    copepod = copepod < thresh

    
    #%%% Erode 10 times
    
    for n in range(10):
        copepod = cp.erode(copepod)
        
    
    
    #%%% Label the blobs
    
    # Image to int from bool
    copepod = copepod.astype(int)
    
    # Label regions
    labels, count = ndi.label(copepod)
    copepod[:] = labels
    
    
    #%%% Look at each labeled blob in turn
    
    # Get clear on what the edges of the main image are
    n_rows = np.shape(copepod)[0]
    n_cols = np.shape(copepod)[1]
    
    # Get all the individual blobs in the image, encoded as lists of slice objects
    objects = ndi.find_objects(copepod)
    
    # Track labels with a counter
    label = 0
    
    # We'll want to keep the label and rough size of each non-edge blob.
    kept_labels = []
    kept_areas = []
    
    # Look at each blob
    for blob in objects:
        # Each blob is a tuple of slices like 
        # (slice(start,stop,step), slice(start,stop,step))
        if blob != None: # Except sometimes they're nones, so skip those
            # Increment label
            label += 1
            # The first slice is the rows (vertical)
            # The second slice is the columns (horizontal)
            row_slice = blob[0]
            col_slice = blob[1]
            # I can't use these objects directly.
            # I can access the (start, stop, step) values as a tuple using the 
            # indices() function of slice objects.
            # That function takes the length of the object you're slicing
            # as its single (required) argument.
            row_slice_indices = row_slice.indices(n_rows)
            col_slice_indices = col_slice.indices(n_cols)
            # These objects are tuples of the (start, stop, step) of the boxes
            # that find_objects has cropped around the blobs.
            # Step is inconsequential here, because a cropped box always has a 
            # step of 1. What we're really interested in is start and stop.
            # If any of the start/stop values in in these tuples are edge values,
            # meaning they're equal to zero or the image size, then the blob is
            # on the edge and can be discarded.
            row_nums_to_check = [row_slice_indices[0],
                                 row_slice_indices[1]]
            row_edge_vals = [0, n_rows]
            
            col_nums_to_check = [col_slice_indices[0],
                                 col_slice_indices[1]]
            
            col_edge_vals = [0, n_cols]
            
            check_row =  any(item in row_nums_to_check for item in row_edge_vals)
            check_col =  any(item in col_nums_to_check for item in col_edge_vals)
            

            if check_row or check_col:   
                # We can discard this blob.
                copepod = np.ma.where(copepod==label, np.zeros_like(copepod), copepod)
            else :
                # We want to keep this blob.
                
                # We need to know which blob is the biggest blob. 
                # We can do that roughly for now.
                # Let's just calculate the area of the minimal rectangle.
                area = (row_slice_indices[1] - row_slice_indices[0]) * (col_slice_indices[1] - col_slice_indices[0])
                
                # Keep this blob.
                kept_labels.append(label)
                kept_areas.append(area)
    
    # Keep the label that's at the same index as the max kept area
    # If we get an error here, then we didn't find a copepod.
    try:
        largest_label = kept_labels[kept_areas.index(max(kept_areas))]
    
        # Now we can get rid of everything that's not that blob
        copepod = np.ma.where(copepod!=largest_label, np.zeros_like(copepod), copepod)
         
        ####################################################################
        # NEED SOME ASSURANCE THAT THE SELECTED BLOB IS ACTUALLY A COPEPOD #
        ####################################################################
        
        # Dilate 10 times
        for n in range(10):
            copepod = cp.dilate(copepod)
            
        
        # Measure the blob
        
        regions = measure.regionprops(copepod)
        
        # There's only one set of properties in this list, by construction
        props = regions[0]
        
        copepod_length = props.major_axis_length
        copepod_width = props.minor_axis_length
        
        # Report copepod find
        copepod_measured = True
        
    except ValueError:
        pass
    


    #%% Write results to file

    
        
    # Generate comments and turn measured values to strings
    if square_found:
        scale = f"{scale:.1f}"
    else:
        comments = comments+"no grid "
        
    if copepod_measured:
        
        # Make a guess about copepod accuracy
    
        # Copepod prediction rules:
        predict_min_length = 75
        predict_max_length = 475
        predict_min_aspect = 1.5
        predict_max_aspect = 4.3
        
        # Calculate copepod aspect ratio
        measured_aspect = copepod_length/copepod_width
        
        # Make checks
        length_ok = predict_min_length <= copepod_length <= predict_max_length
        aspect_ok = predict_min_aspect <= measured_aspect <= predict_max_aspect
        
        # Make our guess
        if length_ok and aspect_ok:
            accuracy = "good"
        else:
            accuracy = "fail"
        
        
        copepod_length = f"{copepod_length:.1f}"
        copepod_width = f"{copepod_width:.1f}"
    else:
        comments = comments+"no copepod"
        accuracy = "miss"
        
    if square_found and copepod_measured:
        comments = "NA"
        
        
    output.write(f"{img_name},{copepod_length},{copepod_width},{scale},{comments},{accuracy}\n")

    ######################################
    # Print for debugging
    # print(f"{img_name},{copepod_length},{copepod_width},{scale},{comments},{accuracy}")
    ######################################

    #%% Create a qa/qc image
                
    # Generate a figure to show what happened
    fig, axes = plt.subplots()
    
    # Show the original image
    axes.imshow(image, cmap=cm.gray)
    
    # Overlay the detected gridlines
    origin = np.array((0, image.shape[1]))
    
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
        
        ellipse = patches.Ellipse((x0,y0), 
                                  props.minor_axis_length, 
                                  props.major_axis_length,
                                  -np.rad2deg(orientation),
                                  fill = False,
                                  linestyle = "-",
                                  edgecolor = "green",
                                  linewidth = 3)
          
        axes.add_artist(ellipse)
    
    # Show the image
    plt.savefig(f"{img_out_dir}qaqc{img_name}")

    ####################
    # Show for debugging
    # plt.show()
    ####################    
    
    plt.close(fig)

#%% Close up shop
    
output.close()

endtime = time.time()
duration = endtime-starttime
n_imgs = len(img_paths)
secs_per_img = duration/n_imgs
tenthou_guess = secs_per_img * 10000


print(f"It took {duration:.3f} seconds to process {n_imgs} images.")
print(f"That's {secs_per_img:.3f} seconds per image.")
print(f"At this pace, it would take {tenthou_guess:.3f} seconds to process 10,000 images.")
