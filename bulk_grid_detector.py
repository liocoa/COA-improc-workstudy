# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:33:37 2020

@author: Lio

Trying to make a pipeline just for detecting the gridlines.

NOTE end-of-term: I'm leaving this code in since it's a useful framework, but
keep in mind that this code does not have robust logic for deciding whether
the grid has been successfully measured. So it really has no idea what it's 
doing. Also it might use old functions that don't work anymore.
"""


import matplotlib.pyplot as plt
from skimage import feature
import math
import time
import copeproc as cp


import os
import glob

sort = ['C:/Users/Emily/Desktop/Image Processing/img_cache/successes/*','C:/Users/Emily/Desktop/Image Processing/img_cache/failures/*']


for direct in sort:
    files = glob.glob(direct)
    for f in files:
        os.remove(f)



starttime = time.time()
# Get the images


images, imgnames = cp.get_images('C:/Users/Emily/Desktop/Image Processing/img_cache/')
    
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
imgfolder = "line_outputs/"
good = "line_outputs/successes/"
bad = "line_outputs/failures/"
success = 0
fail = 0

for img, name in zip(edges,imgnames):
  
    orig = images[imgnames.index(name)]
    
    try:
        lines = cp.detect_gridlines(img)
        scale = cp.find_grid_size(lines,img.shape)
        if math.isnan(scale):
            nans.append(orig)
            nannames.append(name)
            print("We've got a nan.")
        elif scale < 100 or scale > 650:
            print(f"This one looks funny: {scale:.2f}")
            suspicious.append(orig)
            suspiciousnames.append(name)
        else:
            output.write(f"{name}, {scale:.2f} ppm\n")
            print(f"{name}, {scale:.2f}")

        if cp.success(lines):
            cp.plot_with_gridlines(orig, good+name, lines)
            success += 1
        elif not cp.success(lines):
            cp.plot_with_gridlines(orig, bad+name, lines)
            fail += 1
        
    except IndexError:
        print("No lines in that one.")
        fails.append(orig)
        failnames.append(name)

    
nimages = len(images)
nfails = len(fails)
nnans = len(nans)
nsusp = len(suspicious)
successrate = ((nimages - (nfails + nnans + nsusp)) / nimages) * 100
    

time = time.time() - starttime
summary = f"SUMMARY:\nOut of {nimages} total images, {nfails} threw errors and {nnans} returned nan.\n{nsusp} returned sketchy numbers.\nThat's a {successrate:.2f}% success rate."
time_report = f"The program took {time:.2f} seconds to run. That's approximately {time/nimages:.2f} seconds per image."
sort = f"{success} images were sorted as successes. {fail} images were sorted as failures."
print(summary)
print(time_report)
print(sort)
output.write(summary)
output.close()
#%%
cp.show_all(fails,"Fails")
cp.show_all(nans,"NANs")
cp.show_all(suspicious,"Suspicious")





