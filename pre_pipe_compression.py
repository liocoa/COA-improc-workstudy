# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:34:36 2020

@author: Lio

Loop to desaturate and compress a large number of images
"""

input_dir = "C:/Users/Emily/coding/COA-improc-workstudy/images/img_cache/raw/"
output_dir = "C:/Users/Emily/coding/COA-improc-workstudy/images/img_cache/prepped/"

from skimage import color, io, util
import scipy.ndimage as ndi
import glob

# Get the image paths
img_paths = glob.glob(input_dir+'*.jpg')

print("working...")

for path in img_paths:

    # Get the image
    image = io.imread(path)
        
    # Get the image name
    img_name = path[(len(input_dir)):]
    
    # Desaturate the image
    image = color.rgb2gray(image)
    
    # Denoise the image a little
    image = ndi.median_filter(util.img_as_float(image), size=5)
    
    # Downsample the image to 1/2 size
    image = image[::2,::2]
    
    # Cast image type
    image = util.img_as_ubyte(image)
    
    # Save the image
    io.imsave(f"{output_dir}{img_name}",image)
    
print("done.")
