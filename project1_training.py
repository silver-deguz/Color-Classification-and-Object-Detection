#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: jdeguzman
"""

import numpy as np
import cv2
import os
from skimage.measure import label, regionprops
from roipoly import RoiPoly
import pylab as pl
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.color import label2rgb
from scipy import ndimage

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

folder = '/Users/jdeguzman/Desktop/ECE276A_HW1/trainset'
data = load_images_from_folder(folder)
trainset = data[0:40]
valset = data[40:]

# Load color class training data
filename = open('/Users/jdeguzman/Desktop/ECE276A_HW1/colormodels.pkl', 'rb')
K1 = pickle.load(filename)
K2 = pickle.load(filename)
K3 = pickle.load(filename)
K4 = pickle.load(filename)
filename.close()

# Load barrel blue pixel training data
filename = open('/Users/jdeguzman/Desktop/ECE276A_HW1/barrelblue.pkl', 'rb')
K_blue = pickle.load(filename)
filename.close()

# Load non-barrel blue pixel training data
filename = open('/Users/jdeguzman/Desktop/ECE276A_HW1/nonbarrelblue.pkl', 'rb')
K_nbb = pickle.load(filename)
filename.close()

K1 = K_blue.astype(np.uint8)
K2 = K2.astype(np.uint8)
K3 = K3.astype(np.uint8)
K4 = K4.astype(np.uint8)
K5 = K_nbb.astype(np.uint8)

def rgb_to_yuv(X):
    yuv_val = cv2.cvtColor(np.uint8([X]),cv2.COLOR_RGB2YUV) # returns nested array
    return yuv_val[0] # unpack the array

K_blue = rgb_to_yuv(K1)
K_green = rgb_to_yuv(K2)
K_red = rgb_to_yuv(K3)
K_dark = rgb_to_yuv(K4)
K_nbb = rgb_to_yuv(K5)

#%% Single Gaussian model
def compute_gaussian_params(X, diagonal=False):
    mu = np.sum(X, axis=0) / len(X)
    diff = X - mu
    if diagonal:
        covar = np.zeros((3,3))
        for i in range(len(diff)):
            covar += np.diag(diff[i])**2
        covar /= len(X)
    else:
        covar = (np.transpose(diff) @ diff) / len(X)
    return mu, covar

def train(K1, K2, K3, K4, K5):
    mu_blue, covar_blue = compute_gaussian_params(K1, False)
    mu_green, covar_green = compute_gaussian_params(K2, False)
    mu_red, covar_red = compute_gaussian_params(K3, False)
    mu_dark, covar_dark = compute_gaussian_params(K4, False)
    mu_nbb, covar_nbb = compute_gaussian_params(K5, False)

    mu = [mu_blue, mu_green, mu_red, mu_dark, mu_nbb]
    covar = [covar_blue, covar_green, covar_red, covar_dark, covar_nbb]

    a_blue = np.log( 1 / np.sqrt( ((2*np.pi)**3) * np.linalg.det(covar[0]) ) )
    a_green = np.log( 1 / np.sqrt( ((2*np.pi)**3) * np.linalg.det(covar[1]) ) )
    a_red = np.log( 1 / np.sqrt( ((2*np.pi)**3) * np.linalg.det(covar[2]) ) )
    a_dark = np.log( 1 / np.sqrt( ((2*np.pi)**3) * np.linalg.det(covar[3]) ) )
    a_nbb = np.log( 1 / np.sqrt( ((2*np.pi)**3) * np.linalg.det(covar[4]) ) )
    a = [a_blue, a_green, a_red, a_dark, a_nbb]
    return mu, covar, a

def test(mu, covar, a, img):
    # reshape input image into M*Nx3 2D-array
    X = img.reshape((img.shape[0]*img.shape[1]), img.shape[2])
    probs = np.ones((len(X),5))*-100

    diff0 = X - mu[0]
    diff1 = X - mu[1]
    diff2 = X - mu[2]
    diff3 = X - mu[3]
    diff4 = X - mu[4]

    diff0T = np.transpose(diff0)
    diff1T = np.transpose(diff1)
    diff2T = np.transpose(diff2)
    diff3T = np.transpose(diff3)
    diff4T = np.transpose(diff4)

    inv_cov0 = np.linalg.inv(covar[0])
    inv_cov1 = np.linalg.inv(covar[1])
    inv_cov2 = np.linalg.inv(covar[2])
    inv_cov3 = np.linalg.inv(covar[3])
    inv_cov4 = np.linalg.inv(covar[4])

    for i in range(len(X)):
        probs[i,0] = (-0.5 * (diff0[i,:] @ inv_cov0 @ diff0T[:,i])) + a[0]
        probs[i,1] = (-0.5 * (diff1[i,:] @ inv_cov1 @ diff1T[:,i])) + a[1]
        probs[i,2] = (-0.5 * (diff2[i,:] @ inv_cov2 @ diff2T[:,i])) + a[2]
        probs[i,3] = (-0.5 * (diff3[i,:] @ inv_cov3 @ diff3T[:,i])) + a[3]
        probs[i,4] = (-0.5 * (diff4[i,:] @ inv_cov4 @ diff4T[:,i])) + a[4]

    y_hat = np.argmax(probs, axis=1)
    y_hat[y_hat!=0] = -1
    y_hat += 1
    return y_hat


#%% Training!
#mu, covar, a = train(K_blue, K_green, K_red, K_dark, K_nbb)

# Save trained parameters of gaussian classifier
#filename = open('/Users/jdeguzman/Desktop/ECE276A_HW1/gaussian_params2.pkl', 'wb')
#pickle.dump(mu, filename)
#pickle.dump(covar, filename)
#pickle.dump(a, filename)
#filename.close()

#blue_im = np.ones((20,20,3)).astype(np.uint8)
#blue_yuv = blue_im * mu[0]
#blue_yuv = blue_yuv.astype(np.uint8)
#plt.imshow(blue_yuv)
#plt.show()
#plt.imsave('/Users/jdeguzman/Desktop/ECE276A_HW1/mean_img_colors/blue_yuv.png', blue_yuv)
#blue_rgb = cv2.cvtColor(blue_yuv, cv2.COLOR_YUV2RGB)
#plt.imshow(blue_rgb)
#plt.show()
#plt.imsave('/Users/jdeguzman/Desktop/ECE276A_HW1/mean_img_colors/blue_rgb.png', blue_rgb)
#
#green_im = np.ones((20,20,3)).astype(np.uint8)
#green_yuv = green_im * mu[1]
#green_yuv = green_yuv.astype(np.uint8)
#plt.imshow(green_yuv)
#plt.show()
#plt.imsave('/Users/jdeguzman/Desktop/ECE276A_HW1/mean_img_colors/green_yuv.png', green_yuv)
#green_rgb = cv2.cvtColor(green_yuv, cv2.COLOR_YUV2RGB)
#plt.imshow(green_rgb)
#plt.show()
#plt.imsave('/Users/jdeguzman/Desktop/ECE276A_HW1/mean_img_colors/green_rgb.png', green_rgb)
#
#red_im = np.ones((20,20,3)).astype(np.uint8)
#red_yuv = red_im * mu[2]
#red_yuv = red_yuv.astype(np.uint8)
#plt.imshow(red_yuv)
#plt.show()
#plt.imsave('/Users/jdeguzman/Desktop/ECE276A_HW1/mean_img_colors/red_yuv.png', red_yuv)
#red_rgb = cv2.cvtColor(red_yuv, cv2.COLOR_YUV2RGB)
#plt.imshow(red_rgb)
#plt.show()
#plt.imsave('/Users/jdeguzman/Desktop/ECE276A_HW1/mean_img_colors/red_rgb.png', red_rgb)
#
#dark_im = np.ones((20,20,3)).astype(np.uint8)
#dark_yuv = dark_im * mu[3]
#dark_yuv = dark_yuv.astype(np.uint8)
#plt.imshow(dark_yuv)
#plt.show()
#plt.imsave('/Users/jdeguzman/Desktop/ECE276A_HW1/mean_img_colors/dark_yuv.png', dark_yuv)
#dark_rgb = cv2.cvtColor(dark_yuv, cv2.COLOR_YUV2RGB)
#plt.imshow(dark_rgb)
#plt.show()
#plt.imsave('/Users/jdeguzman/Desktop/ECE276A_HW1/mean_img_colors/dark_rgb.png', dark_rgb)
#
#nbb_im = np.ones((20,20,3)).astype(np.uint8)
#nbb_yuv = nbb_im * mu[4]
#nbb_yuv = nbb_yuv.astype(np.uint8)
#plt.imshow(nbb_yuv)
#plt.show()
#plt.imsave('/Users/jdeguzman/Desktop/ECE276A_HW1/mean_img_colors/nbb_yuv.png', nbb_yuv)
#nbb_rgb = cv2.cvtColor(nbb_yuv, cv2.COLOR_YUV2RGB)
#plt.imshow(nbb_rgb)
#plt.show()
#plt.imsave('/Users/jdeguzman/Desktop/ECE276A_HW1/mean_img_colors/nbb_rgb.png', nbb_rgb)



#%% Segment test image
# Segment the image into K=4 different colors
# trainset #11, 33
# valset #5
num = 3
test_img = trainset[num]
x_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2YUV)

tic = time.time()
y_test = test(mu, covar, a, x_test)
toc = time.time()
print(toc-tic)

# Create binary mask
y_mask = y_test.reshape((x_test.shape[0], x_test.shape[1]))
y_mask = y_mask.astype(np.uint8)*255 #Binary image mask

# close then open
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

mask = cv2.morphologyEx(y_mask, cv2.MORPH_CLOSE, kernel1)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)
binary_img = y_mask * mask

# Display binary mask pre-processing
fig, ax = plt.subplots(figsize=(8,6))
ax.imshow(y_mask, cmap='gray')
#plt.imshow(y_mask, cmap='gray')
plt.tight_layout()
plt.title('Segmentation Image')
plt.show()

#%%
# Display binary mask post-processing
fig, ax = plt.subplots(figsize=(8,6))
#ax.imshow(y_mask, cmap='gray')
plt.imshow(binary_img, cmap='gray')
plt.tight_layout()
plt.title('Binary Mask Image')
plt.show()

# Display original test image in RGB color space
x_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
#fig, ax = plt.subplots(figsize=(8,6))
#plt.imshow(x_test)
#plt.tight_layout()
#plt.title('Original Image')
#plt.show()

fig, ax = plt.subplots(figsize=(8,8))
_, contours, hierarchy = cv2.findContours(binary_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True) # reorder contours from large to small

boxes = []

for contour in contours:
#    print(cv2.contourArea(contour))
    if cv2.contourArea(contour) >= 1500 and cv2.contourArea(contour) <= 45000:
#        print(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour) #[r1, c1, r2, c2] = [y1, x1, y2, x2]
        aspect_ratio = float(h)/w
        print(aspect_ratio)
        if aspect_ratio >= 1.1:
            rect = mpatches.Rectangle((x,y), w, h,fill=False,edgecolor='green',linewidth=3)
            ax.add_patch(rect)
            boxes.append([x, y, x+w, y+h])

if not boxes:
    print('empty')
    x, y, w, h = cv2.boundingRect(contours[0])
    boxes.append([x, y, x+w, y+h])

plt.imshow(x_test)
plt.tight_layout()
ax.set_axis_off()
plt.show()

#%% Bounding box

#plt.imshow(binary_img, cmap='gray')
#plt.show()
#
##img_fill_holes = ndimage.binary_fill_holes(binary_img).astype(np.uint8)
##plt.imshow(img_fill_holes, cmap='gray')
##plt.show()
#
#_, contours, hierarchy = cv2.findContours(binary_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
##img, contours, hierarchy = cv2.findContours(img_fill_holes,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#labels = label(binary_img)
#
#fig, ax = plt.subplots(figsize=(8,6))
#ax.imshow(binary_img, cmap='gray')
##ax.imshow(img_fill_holes, cmap='gray')
#
#boxes = []
#for region in regionprops(labels):
#    if region.area >= 1500:
#        print(region.area)
#        y1, x1, y2, x2 = region.bbox # [r1, c1, r2, c2] = [y1, x1, y2, x2]
#        print(y1,x1,y2,x2)
#        rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1,fill=False,edgecolor='green',linewidth=3)
#        ax.add_patch(rect)
#        boxes.append([x1, y1, x2, y2])
#
#plt.title('Region Props')
#plt.show()

#%% Barrel detection function

def barrel_detect(binary_img, i, rgb_img):
    fig, ax = plt.subplots(figsize=(8,8))
    _, contours, hierarchy = cv2.findContours(binary_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True) # reorder contours from large to small

    boxes = []

    for contour in contours:
        A = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(h)/w
        if aspect_ratio >= 1.1:
            print(aspect_ratio)
            if cv2.contourArea(contour) >= 1000 and cv2.contourArea(contour) <= 50000:
                print(A)
                rect = mpatches.Rectangle((x,y), w, h,fill=False,edgecolor='red',linewidth=4)
                ax.add_patch(rect)
                boxes.append([x, y, x+w, y+h])

#        if cv2.contourArea(contour) >= 1000 and cv2.contourArea(contour) <= 45000:
#            print(cv2.contourArea(contour))
#            x, y, w, h = cv2.boundingRect(contour) #[r1, c1, r2, c2] = [y1, x1, y2, x2]
#            aspect_ratio = float(h)/w
#            print(aspect_ratio)
#            if aspect_ratio >= 0.9:
#                rect = mpatches.Rectangle((x,y), w, h,fill=False,edgecolor='green',linewidth=3)
#                ax.add_patch(rect)
#                boxes.append([x, y, x+w, y+h])

    if not boxes:
        print('empty')
        x, y, w, h = cv2.boundingRect(contours[0])
        boxes.append([x, y, x+w, y+h])

    plt.imshow(rgb_img)
#    plt.imshow(binary_img, cmap='gray')
    plt.tight_layout()
    ax.set_axis_off()
    plt.show()
    fig.savefig('/Users/jdeguzman/Desktop/ECE276A_HW1/detections/trainset%s_detect.png' %i, bbox_inches='tight')

#%%
#-------------------------#
# Training Set Evaluation #
#-------------------------#

kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

tic = time.time()

for i in range(len(trainset)):
    print('trainset #%d' %i)
    img = cv2.cvtColor(trainset[i], cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
#    plt.imsave('/Users/jdeguzman/Desktop/ECE276A_HW1/RGB/trainset%s_rgb.png' %i, img)

    # Convert to YUV space
    x_test = cv2.cvtColor(trainset[i], cv2.COLOR_BGR2YUV)
#    plt.imsave('/Users/jdeguzman/Desktop/ECE276A_HW1/YUV/trainset%s_yuv.png' %i, x_test)
    y_test = test(mu, covar, a, x_test)

    # Create binary mask
    y_mask = y_test.reshape((x_test.shape[0], x_test.shape[1]))
    y_mask = y_mask.astype(np.uint8)*255 #Binary image mask
    plt.imshow(y_mask, cmap='gray')
    plt.show()
#    plt.imsave('/Users/jdeguzman/Desktop/ECE276A_HW1/binary_masks/trainset%s_bm.png' %i, y_mask, cmap='gray')

    # Morphological post-processing
    mask = cv2.morphologyEx(y_mask, cv2.MORPH_CLOSE, kernel2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
    binary_img = y_mask * mask
    plt.imshow(binary_img, cmap='gray')
    plt.show()
#    plt.imsave('/Users/jdeguzman/Desktop/ECE276A_HW1/postprocessed_masks/trainset%s_ppm.png' %i, binary_img, cmap='gray')

    # Barrel detection, bounding box
    barrel_detect(binary_img, i, img)

toc = time.time()
print(toc-tic)

#%%
#---------------------------#
# Validation Set Evaluation #
#---------------------------#

kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

for i in range(len(valset)):
    img = cv2.cvtColor(valset[i], cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    plt.imsave('/Users/jdeguzman/Desktop/ECE276A_HW1/RGB/valset%s_rgb.png' %i, img)

    # Convert to YUV space
    x_test = cv2.cvtColor(valset[i], cv2.COLOR_BGR2YUV)
    plt.imsave('/Users/jdeguzman/Desktop/ECE276A_HW1/YUV/valset%s_yuv.png' %i, x_test)
    y_test = test(mu, covar, a, x_test)

    # Create binary mask
    y_mask = y_test.reshape((x_test.shape[0], x_test.shape[1]))
    y_mask = y_mask.astype(np.uint8)*255 #Binary image mask
    plt.imshow(y_mask, cmap='gray')
    plt.show()
    plt.imsave('/Users/jdeguzman/Desktop/ECE276A_HW1/binary_masks/valset%s_bm.png' %i, y_mask, cmap='gray')

    # Morphological post-processing
    mask = cv2.morphologyEx(y_mask, cv2.MORPH_CLOSE, kernel2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
    binary_img = y_mask * mask
    plt.imshow(binary_img, cmap='gray')
    plt.show()
    plt.imsave('/Users/jdeguzman/Desktop/ECE276A_HW1/postprocessed_masks/valset%s_ppm.png' %i, binary_img, cmap='gray')

    # Barrel detection, bounding box
    barrel_detect(binary_img, i, img)
