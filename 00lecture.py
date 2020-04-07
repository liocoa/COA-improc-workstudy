"""
This is the example code from the first doc in the tutorial series.

"""


import numpy as np
from matplotlib import pyplot as plt

random_image = np.random.random([500,500])

plt.imshow(random_image, cmap='gray')
plt.colorbar();

from skimage import data
coins = data.coins()
print("Type:", type(coins))
print("dtype:", coins.dtype)
print("shape:", coins.shape)

plt.imshow(coins, cmap="gray")

cat = data.chelsea()
print('Shape:', cat.shape)
print('Values min/max:', cat.min(),cat.max())

plt.imshow(cat);

cat[10:110, 10:110, :] = [255,0,0]
plt.imshow(cat);


img0 = data.chelsea()
img1 = data.rocket()

f, (ax0,ax1) = plt.subplots(1,2,figsize=(20,10))

ax0.imshow(img0)
ax0.set_title('Cat', fontsize=18)
ax0.axis('off')

ax1.imshow(img1)
ax1.set_title("Rocket", fontsize=18)
ax1.set_xlabel(r'Launching position $\alpha=320$')

ax1.vlines([202, 300], 0, img1.shape[0], colors='magenta', linewidth=3, label='Side tower position')
ax1.plot([168, 190, 200], [400, 200, 300], color='white', linestyle='--', label='Side angle')

ax1.legend();

#%%
#Exercise: draw the letter H
def draw_H(image, coords, color=(0, 255, 0)):
    out = image.copy()
    row = coords[0]
    col = coords[1]
    out[col:col+24, row:row+3, :] = color
    out[col:col+24, row+17:row+20, :] = color
    out[col+11:col+14, row:row+20, :] = color
    
    
    return out 



#%%
#Exercise: visualize RGB color channels
# --- read in the image ---

image = plt.imread('C:/Users/Emily/skimage-tutorials/images/Bells-Beach.jpg')

# --- assign each color channel to a different variable ---


r = [1,0,0] * image
g = [0,1,0] * image
b = [0,0,1] * image


# --- display the image and r, g, b channels ---

f, axes = plt.subplots(1, 4, figsize=(16, 5))

for ax in axes:
    ax.axis('off')

(ax_r, ax_g, ax_b, ax_color) = axes
    
ax_r.imshow(r, cmap='gray')
ax_r.set_title('red channel')

ax_g.imshow(g, cmap='gray')
ax_g.set_title('green channel')

ax_b.imshow(b, cmap='gray')
ax_b.set_title('blue channel')

# --- Here, we stack the R, G, and B layers again
#     to form a color image ---
ax_color.imshow(np.stack([r, g, b], axis=2))
ax_color.set_title('all channels');


#%%
#Exercise: Convert to grayscale

from skimage import color, img_as_float, io

image = img_as_float(io.imread('C:/Users/Emily/skimage-tutorials/images/balloon.jpg'))

gray = color.rgb2gray(image)
my_gray = 

# --- display the results ---

f, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 6))

ax0.imshow(gray, cmap='gray')
ax0.set_title('skimage.color.rgb2gray')

ax1.imshow(my_gray, cmap='gray')
ax1.set_title('my rgb2gray')







