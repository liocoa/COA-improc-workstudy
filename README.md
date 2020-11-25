# Copepod processing

This repository contains two pipelines for detecting and measuring copepods in images taken at The Rock. Both pipelines are buggy and neither should be considered complete, but they both successfully detect and measure copepods at least sometimes.

## File Description

`copeproc.py` 
A module containing helper functions for the other two modules.

`pre_pipe_compression.py`
A module which desaturates and downsamples to 1/2 size an entire folder of jpeg images. Input and output directories are set within the script. 

`basic_pipe.py`
A program which runs grid detection and naive copepod detection on a folder of images. Input and output directories are set within the script, including QA/QC output images and a .csv file containing the the outputs. This pipeline runs on images that have been processed using `pre_pipe_compression.py`.

`smart_pipe.py`
A program which runs grid detecction and smart copepod detection on a folder of images using an object detection model trained in Detecto. This program currently uses uncompressed, color images. The plan is to retrain the object detection on compressed greyscale images, but the training code keeps breaking in very weird ways so that hasn't happened yet.

## Use Guide

### Installation and preparation

Clone the repo and download your target image set. Set up whatever directory tree you'd like, within these guidelines:
	You must have all the images you want to process in the same folder
	You must have raw (color) images and prepped (compressed) images in different folders
	You must not have any .jpg files in an input folder except those you want to process

### Basic pipeline

In `pre_pipe_compression.py`, set the `input_dir` variable to the directory containing all the raw (color) images you want to process. Set the `output_dir` variable to the directory in which you want to store the compressed images (must be a different directory). Run the code. It will take a little while and there is no progress bar. 

Next, open `basic_pipe.py`. Set the `input_dir` variable to the directory containing the compressed images generated in the previous step. Set the `img_out_dir` variable to wherever you want to store the generated QA/QC images, and the `table_out_dir` variable to wherever you want to store the generated .csv file. Set the `table_name` variable to whatever you want that .csv to be called. Run the code. On my computer this takes about .3 seconds per image. There is no progress bar, but you will get a little printout at the end telling you how long it took and how long it might take to run 10,000 images.

Once the code is done, have a look at your output images and output table. If a copepod was measured on an image, the program makes a little guess about whether the measurement was good, either "fail" or "good". You can compare visually against the QA/QC images to see whether the guesses are ok.

### Smart pipeline

Open `smart_pipe.py`. For now, use raw images for this pipeline, not compressed ones.

For setup, set `input_dir`, `img_out_dir`, and `table_out_dir` variables. Set `model_path` to the path to the .pth file to use for the model. This file can be found on Lio's Drive. I'll share it with you, Dan. It's too big to upload here. Set `table_name.py`. 

There are some extra options at the top of this file, mostly for debugging purposes. You can set `show = True` to get some images to show how the algorithm is going. You can set `verbose = True` to get some printouts, but please don't expect much from them. You can set `sample_size` to `None` to run all the images in your input directory, or an integer smaller than your total number of images to run the code on a random sample of that many images. I don't recommend running it on more than a few images, because I expect it to break.

Then you can run the code and see what happens. Maybe it'll work! Maybe it won't.
