from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# from scipy import ndimage
# import mahotas as mh

import os, sys
input = os.path.join(os.path.dirname(__file__), '..', 'input/zip_codes/')

# Import the cv2 library
from skimage import data
from skimage import filters, transform
from skimage.measure import label, regionprops
from skimage.color import label2rgb

def extract(filename):
	fname= input + filename
	blur_radius = 1.0

	img = Image.open(fname).convert('L')
	img = np.asarray(img)
	img = filters.gaussian(img, sigma=9)
	# print(img)
	val = 105/255
	img = img < val
	# blobs = img < 0.77 * img.mean()

	# all_labels = measure.label(blobs)
	labels = label(img, neighbors=8,  background=0)
	digit_lefts = []
	digits = []
	for region in regionprops(labels):
		# take regions with large enough areas
		if region.area >= 20000:
			# binary image 0, 1
			minr, minc, maxr, maxc = region.bbox
			digit = region.image.astype(float)
			# change background to white, 255
			digit[digit == 0] = 255
			# change digit to black, 0
			digit[digit != 255] = 0

			# pad to square image
			num_rows, num_cols = digit.shape
			pad_width = max(num_rows, num_cols)
			col_pad = int((pad_width - num_cols)/2)+100
			row_pad = int((pad_width - num_rows)/2)+100
			digit = np.pad(digit, ([row_pad,row_pad], [col_pad, col_pad]), 'maximum')
			digit = Image.fromarray(digit)
			# digit = transform.resize(digit, (28, 28))
			# # print(digit)
			# # resize to match mnist
			# threshold = 200
			# digit[digit < threshold] = 0
			# digit[digit >= threshold] = 255

			digit_lefts.append(minc)
			digits.append(digit)

	return digits, digit_lefts






