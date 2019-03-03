'''
ECE276A WI19 HW1
Blue Barrel Detector
'''

import os, cv2
from skimage.measure import label, regionprops
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pylab as pl
import time
from scipy import ndimage

class BarrelDetector():
	def __init__(self):
		'''
			Initilize your blue barrel detector with the attributes you need
			eg. parameters of your classifier
		'''
		# raise NotImplementedError
		filename = open('gaussian_params2.pkl', 'rb')
		self.mu = pickle.load(filename)
		self.covar = pickle.load(filename)
		self.a = pickle.load(filename)
		filename.close()

	def segment_image(self, img):
		'''
			Calculate the segmented image using a classifier
			eg. Single Gaussian, Gaussian Mixture, or Logistic Regression
			call other functions in this class if needed

			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
		'''
		# raise NotImplementedError
		test_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
		X = test_img.reshape((img.shape[0]*img.shape[1]), img.shape[2])

		# Color classification
		# probs = np.zeros((len(X),4)) # using 4 color classes
		probs = np.zeros((len(X),5)) # using 5 color classes

		diff0 = X - self.mu[0]
		diff1 = X - self.mu[1]
		diff2 = X - self.mu[2]
		diff3 = X - self.mu[3]
		diff4 = X - self.mu[4]

		diff0T = np.transpose(diff0)
		diff1T = np.transpose(diff1)
		diff2T = np.transpose(diff2)
		diff3T = np.transpose(diff3)
		diff4T = np.transpose(diff4)

		inv_cov0 = np.linalg.inv(self.covar[0])
		inv_cov1 = np.linalg.inv(self.covar[1])
		inv_cov2 = np.linalg.inv(self.covar[2])
		inv_cov3 = np.linalg.inv(self.covar[3])
		inv_cov4 = np.linalg.inv(self.covar[4])

		for i in range(len(X)):
			probs[i,0] = (-0.5 * (diff0[i,:] @ inv_cov0 @ diff0T[:,i])) + self.a[0]
			probs[i,1] = (-0.5 * (diff1[i,:] @ inv_cov1 @ diff1T[:,i])) + self.a[1]
			probs[i,2] = (-0.5 * (diff2[i,:] @ inv_cov2 @ diff2T[:,i])) + self.a[2]
			probs[i,3] = (-0.5 * (diff3[i,:] @ inv_cov3 @ diff3T[:,i])) + self.a[3]
			probs[i,4] = (-0.5 * (diff4[i,:] @ inv_cov4 @ diff4T[:,i])) + self.a[4]

		y_hat = np.argmax(probs, axis=1)
		y_hat[y_hat != 0] = -1
		y_hat += 1

		# Create binary mask
		y_mask = y_hat.reshape((img.shape[0], img.shape[1]))
		y_mask = y_mask.astype(np.uint8)*255

		# Post-processing to remove noise
		# kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
		# kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(11,11))
		# mask = cv2.morphologyEx(y_mask, cv2.MORPH_CLOSE, kernel1)
		# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)
		# out = y_mask * mask

		## Display binary mask pre-processing
		# plt.imshow(y_mask, cmap='gray')
		# plt.show()
		## Display binary mask post-processing
		# plt.imshow(out, cmap='gray')
		# plt.show()
		## Display original test image in RGB color space
		# x_test = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# plt.imshow(x_test)
		# plt.show()
		# out = np.ones((img.shape[0], img.shape[1]))
		# mask_img = out.astype(np.uint8)*255

		mask_img = y_mask
		return mask_img

	def get_bounding_box(self, img):
		'''
			Find the bounding box of the blue barrel
			call other functions in this class if needed

			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2]
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
				is from left to right in the image.

			Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
		'''
		# raise NotImplementedError
		binary_img = self.segment_image(img)
		boxes = []

		# Post-processing to remove noise
		kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
		kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
		mask = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel2)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
		out = binary_img * mask

		# fig, ax = plt.subplots((figsize=(8,6)))

		# Using findContours
		contours, hierarchy = cv2.findContours(out,cv2.RETR_TREE,\
											cv2.CHAIN_APPROX_SIMPLE)
		contours = sorted(contours, key = cv2.contourArea, reverse = True)

		for contour in contours:
			if cv2.contourArea(contour) >= 1000 and cv2.contourArea(contour) <= 40000:
				# print(cv2.contourArea(contour))
				x,y,w,h = cv2.boundingRect(contour)
				aspect_ratio = float(h)/w
				# print(aspect_ratio)
				if aspect_ratio >= 1.15:
					# rect = mpatches.Rectangle((x,y), w, h,fill=False,\
											# edgecolor='green',linewidth=3)
					# ax.add_patch(rect)
					boxes.append([x,y,x+w,y+h])

		# Using regionprops
		# labels = label(binary_img)
		# for region in regionprops(labels):
		# 	if region.area >= 1500:
		# 		y1,x1,y2,x2 = region.bbox
		# 		# rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1,fill=False,\
		# 					# edgecolor='green,linewidth=3)
		# 		# ax.add_patch(rect)
		# 		boxes.append([x1, y1, x2, y2])

		if not boxes: # if empty list, return largest bbox of largest contour
			print('empty')
			x,y,w,h = cv2.boundingRect(contours[0])
			boxes.append([x,y,x+w,y+h])


		# plt.imshow(binary_img, cmap='gray')
		# plt.title('Bounding Rect')
		# plt.show()
		return boxes


if __name__ == '__main__':
	folder = "trainset"
	# folder = '/Users/jdeguzman/Desktop/ECE276A_HW1/trainset'
	my_detector = BarrelDetector()
	# images = []
	for filename in os.listdir(folder):
		# read one test image
		img = cv2.imread(os.path.join(folder,filename))
		# if img is not None:
			# images.append(img)
		# mask_img = my_detector.segment_image(img)
		# boxes = my_detector.get_bounding_box(img)

	# tic = time.time()
	# for i in range(36,46):
	# 	mask1 = my_detector.segment_image(images[i])
	# toc = time.time()
	# print('DONE!', toc-tic)

		#Display results:
		#(1) Segmented images
		#	 mask_img = my_detector.segment_image(img)
		#(2) Barrel bounding box
		#    boxes = my_detector.get_bounding_box(img)
		#The autograder checks your answers to the functions segment_image() and get_bounding_box()
		#Make sure your code runs as expected on the testset before submitting to Gradescope
