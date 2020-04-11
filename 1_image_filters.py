# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:24:20 2020

@author: Lio

Lecture code from image filtering
"""


import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['image.cmap'] = 'gray'

step_signal = np.zeros(100)
step_signal[50:] = 1
fig, ax = plt.subplots()
ax.plot(step_signal)
ax.margins(y=0.1)
#%%
# Just to make sure we all see the same results
np.random.seed(0)


noisy_signal = (step_signal
                + np.random.normal(0, 0.35, step_signal.shape))
fig, ax = plt.subplots()
ax.plot(noisy_signal);
#%%
# Take the mean of neighboring pixels
smooth_signal = (noisy_signal[:-1] + noisy_signal[1:]) / 2.0
fig, ax = plt.subplots()
ax.plot(smooth_signal);
#%%
smooth_signal3 = (noisy_signal[:-2] + noisy_signal[1:-1]
                  + noisy_signal[2:]) / 3
fig, ax = plt.subplots()
ax.plot(smooth_signal, label='mean of 2')
ax.plot(smooth_signal3, label='mean of 3')
ax.legend(loc='upper left');
#%%
# Same as above, using a convolution kernel
# Neighboring pixels multiplied by 1/3 and summed
mean_kernel3 = np.full((3,), 1/3)
smooth_signal3p = np.convolve(noisy_signal, mean_kernel3,
                              mode='valid')
fig, ax = plt.subplots()
ax.plot(smooth_signal3p)

print('smooth_signal3 and smooth_signal3p are equal:',
      np.allclose(smooth_signal3, smooth_signal3p))
#%%
def convolve_demo(signal, kernel):
    ksize = len(kernel)
    convolved = np.correlate(signal, kernel)
    def filter_step(i):
        fig, ax = plt.subplots()
        ax.plot(signal, label='signal')
        ax.plot(convolved[:i+1], label='convolved')
        ax.legend()
        ax.scatter(np.arange(i, i+ksize),
                   signal[i : i+ksize])
        ax.scatter(i, convolved[i])
    return filter_step

from ipywidgets import interact, widgets

i_slider = widgets.IntSlider(min=0, max=len(noisy_signal) - 3,
                             value=0)

interact(convolve_demo(noisy_signal, mean_kernel3),
         i=i_slider);
#the ipython notebook isn't doing the thing, and I don't know it well enough
#to make it do the thing, so I'm just gonna leave this be.
#%%
#Let's try to get the widgets to work?


#%%
mean_kernel11 = np.full((11,), 1/11)
smooth_signal11 = np.convolve(noisy_signal, mean_kernel11,
                              mode='valid')
fig, ax = plt.subplots()
ax.plot(smooth_signal11);
#%%
smooth_signal3same = np.convolve(noisy_signal, mean_kernel3,
                                 mode='same')
smooth_signal11same = np.convolve(noisy_signal, mean_kernel11,
                                  mode='same')

fig, ax = plt.subplots(1, 2)
ax[0].plot(smooth_signal3p)
ax[0].plot(smooth_signal11)
ax[0].set_title('mode=valid')
ax[1].plot(smooth_signal3same)
ax[1].plot(smooth_signal11same)
ax[1].set_title('mode=same');

#%%
fig, ax = plt.subplots()
ax.plot(step_signal)
ax.margins(y=0.1) 

#stop and think...

result_corr = np.correlate(step_signal, np.array([-1, 0, 1]),
                           mode='valid')
result_conv = np.convolve(step_signal, np.array([-1, 0, 1]),
                          mode='valid')
fig, ax = plt.subplots()
ax.plot(step_signal, label='signal')
ax.plot(result_conv, linestyle='dashed', label='convolved')
ax.plot(result_corr, linestyle='dashed', label='correlated',
        color='C3')
ax.legend(loc='upper left')
ax.margins(y=0.1) 

#so we can use this "spike" to find the edges of things

#%%
noisy_change = np.correlate(noisy_signal, np.array([-1, 0, 1]))
fig, ax = plt.subplots()
ax.plot(noisy_signal, label='signal')
ax.plot(noisy_change, linestyle='dashed', label='change')
ax.legend(loc='upper left')
ax.margins(0.1)

#%%
mean_diff = np.correlate([-1, 0, 1], [1/3, 1/3, 1/3], mode='full')
print(mean_diff)

smooth_change = np.correlate(noisy_signal, mean_diff,
                             mode='same')
fig, ax = plt.subplots()
ax.plot(noisy_signal, label='signal')
ax.plot(smooth_change, linestyle='dashed', label='change')
ax.margins(0.1)
ax.hlines([-0.5, 0.5], 0, 100, linewidth=0.5, color='gray');

#This finds the edge in a noisy signal
#Dude, this is a lot
#I am moderately overwhelmed
#What do you do with all of this
#Like, I get it conceptually but I am not following the code
#And I can't envision how to apply it

#%%
#Exercise: Doing that same thing to the Gaussian filter
#You know, I don't think I understand the question.
#What exactly is a filter? It's an operation you do to the value of each
#pixel based on the values of the pixels around it.
#How do you filter a filter???
#Ok, I think I'm starting to get it
xi = np.arange(9)
x0 = 9 // 2  # 4
x = xi - x0
sigma = 1
# complete this code

def gaussian(xi_arr, x0_int, sigma):
    x = xi - x0
    kernel = (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-(x**2/(2*(sigma**2))))
    return kernel


#Actually this filter isn't even a filter
    #I'm so confused
#Ok so I think I wrote the filter ok, but I'll admit I'm having a hard time
#understanding the question.

#%%
from scipy import ndimage

gauskernal = gaussian(xi, x0, 1)

attempt = ndimage.convolve(noisy_signal,gauskernal,mode="mirror")

fig, ax = plt.subplots()
ax.plot(noisy_signal)
ax.plot(attempt)
gaus_diff = np.convolve(gauskernal,[-1,0,1],mode="same")
smoothdiff = np.convolve(noisy_signal,gaus_diff,mode="same")
ax.plot(smoothdiff)
#Hmmm!
ax.hlines([-0.5, 0.5], 0, 100, linewidth=0.5, color='gray');



#%%
#Local filtering of images
#I imagine I'll dive into this and then need to back up a bunch
import numpy as np

bright_square = np.zeros((7, 7), dtype=float)
bright_square[2:5, 2:5] = 1
fig, ax = plt.subplots()
ax.imshow(bright_square);
#%%
mean_kernel = np.full((3, 3), 1/9)
print(mean_kernel)

import scipy.ndimage as ndi


print(bright_square)
a = ndi.correlate(bright_square, mean_kernel)
print(a)
ax.imshow(a)


#%%
#A real image

from skimage import data

image = data.camera()
pixelated = image[::10, ::10]
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
ax0.imshow(image)
ax1.imshow(pixelated) ;

from skimage import img_as_float

#Helper function provided
def imshow_all(*images, titles=None):
    images = [img_as_float(img) for img in images]

    if titles is None:
        titles = [''] * len(images)
    vmin = min(map(np.min, images))
    vmax = max(map(np.max, images))
    ncols = len(images)
    height = 5
    width = height * len(images)
    fig, axes = plt.subplots(nrows=1, ncols=ncols,
                             figsize=(width, height))
    for ax, img, label in zip(axes.ravel(), images, titles):
        ax.imshow(img, vmin=vmin, vmax=vmax)
        ax.set_title(label)

filtered = ndi.correlate(pixelated, mean_kernel)
imshow_all(pixelated, filtered, titles=['pixelated', 'mean filtered'])

#%%

#Exercise: (Chapter 0 reminder!) Plot the profile of the gaussian kernel at 
#its midpoint, i.e. the values under the line shown here:
from skimage import filters
sidelen = 45
sigma = (sidelen - 1) // 2 // 4
spot = np.zeros((sidelen, sidelen), dtype=float)
spot[sidelen // 2, sidelen // 2] = 1
kernel = filters.gaussian(spot, sigma=sigma)

imshow_all(spot, kernel / np.max(kernel))

fig, ax = plt.subplots()

ax.imshow(kernel, cmap='inferno')
ax.vlines(22, -100, 100, color='C9')
ax.set_ylim((sidelen - 1, 0))
#%%

relevant = kernel[:,[22]]
print(relevant)
relevant = np.reshape(relevant,(1,len(relevant)))
print(relevant)

fig, ax1 = plt.subplots()
ax1.plot(relevant[0])
#Got it.
#%%
#Edge detection
vertical_kernel = np.array([
    [-1],
    [ 0],
    [ 1],
])

gradient_vertical = ndi.correlate(pixelated.astype(float),
                                  vertical_kernel)
fig, ax = plt.subplots()
ax.imshow(gradient_vertical);
#%%
#Exercise:
#Add a horizontal kernel to the above example to also compute the 
    #horizontal gradient,  𝑔𝑦
#Compute the magnitude of the image gradient at each point:  
    #|𝑔|=√(𝑔^2𝑥+𝑔^2𝑦)
#The idea here is that we're using a 1D difference filter to approximate
    #the gradient
#What is the gradient?

image = pixelated.astype(float)

vertical_kernel = np.array([
    [-1],
    [ 0],
    [ 1],
])
horizontal_kernel = np.array([[-1, 0, 1]])

gradient_vertical = ndi.correlate(image, vertical_kernel)
gradient_horizontal = ndi.correlate(image, horizontal_kernel)

gradient_magnitude = np.zeros_like(image)
for row in range(np.shape(image)[0]):
    for col in range(np.shape(image)[1]):
        gradient_magnitude[row,col] = np.sqrt(gradient_vertical[row,col]**2 +
                                              gradient_horizontal[row,col]**2)

imshow_all(gradient_vertical,gradient_horizontal, gradient_magnitude,
           titles=('vertical','horizontal','magnitude'));

#I did it!






































