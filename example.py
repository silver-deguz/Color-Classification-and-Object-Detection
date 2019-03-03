#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 13:26:33 2019

@author: jdeguzman
"""

import logging

import numpy as np
import pickle
from matplotlib import pyplot as plt
from roipoly import RoiPoly
from PIL import Image


logger = logging.getLogger(__name__)

logging.basicConfig(format='%(levelname)s ''%(processName)-10s : %(asctime)s '
                           '%(module)s.%(funcName)s:%(lineno)s %(message)s',
                    level=logging.INFO)

# Implements RoiPoly to get color class labels for the first 40 images in the
# training dataset. The color class pixel data is saved in a pickle file.
#
# K1 = np.empty((0,3), int)
# K2 = np.empty((0,3), int)
# K3 = np.empty((0,3), int)
# K4 = np.empty((0,3), int)
K5 = np.empty((0,3), int)

for i in range(1,41):
    I1 = np.array(Image.open('./trainset/%d.png' %i), dtype='int')
    gray = I1[:,:,0]

    # Show the image
    fig = plt.figure()
    plt.imshow(I1, interpolation='nearest')
    plt.title('Image %i, draw first ROI' %i)
    plt.show(block=False)

    # Let user draw first ROI
    roi1 = RoiPoly(color='r', fig=fig)

    # Show the image
    fig = plt.figure()
    plt.imshow(I1, interpolation='nearest')
    plt.title('Image %i, draw second ROI' %i)
    plt.show(block=False)

    # # Let user draw second ROI
    # roi2 = RoiPoly(color='b', fig=fig)
    #
    # # Show the image
    # fig = plt.figure()
    # plt.imshow(I1, interpolation='nearest')
    # plt.title('Image %i, draw third ROI' %i)
    # plt.show(block=False)

    # # Let user draw third ROI
    # roi3 = RoiPoly(color='r', fig=fig)
    #
    # # Show the image
    # fig = plt.figure()
    # plt.imshow(I1, interpolation='nearest')
    # plt.title('Image %i, draw fourth ROI' %i)
    # plt.show(block=False)

    # # Let user draw fourth ROI
    # roi4 = RoiPoly(color='b', fig=fig)
    #
    # # Show the image with both ROIs
    # plt.imshow(I1, interpolation='nearest')
    # [x.display_roi() for x in [roi1, roi2, roi3, roi4]]
    roi1.display_roi()
    plt.title('All ROIs')
    plt.show()

    mask1 = roi1.get_mask(gray)
    # mask2 = roi2.get_mask(gray)
    # mask3 = roi3.get_mask(gray)
    # mask4 = roi4.get_mask(gray)

    # Show ROI masks
    plt.imshow(mask1, interpolation='nearest', cmap='Greys')
    plt.title('ROI mask')
    plt.show()
    # plt.imshow(mask1 + mask2 + mask3 + mask4, \
    #            interpolation='nearest', cmap="Greys")
    # plt.title('ROI masks of the ROIs')
    # plt.show()

    r1,c1 = np.where(mask1 == True)
    # r2,c2 = np.where(mask2 == True)
    # r3,c3 = np.where(mask3 == True)
    # r4,c4 = np.where(mask4 == True)

    # Four color classes for barrel_blue, green, red, dark
    # K1 = np.append(K1, I1[r1,c1,:], axis=0)
    # K2 = np.append(K2, I1[r2,c2,:], axis=0)
    # K3 = np.append(K3, I1[r3,c3,:], axis=0)
    # K4 = np.append(K4, I1[r4,c4,:], axis=0)
    K5 = np.append(K5, I1[r1,c1,:], axis=0)



filename = open('nonbarrelblue.pkl', 'wb')
pickle.dump(K5, filename)
filename.close()
