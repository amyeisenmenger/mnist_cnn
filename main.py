
import torch
import torchvision
# from torchvision import datasets, transforms


# import local files
from functions.classify_digits import *
from functions.classify_even_odd import *
from functions.predict_zipcode import *

def display_sample(images):
	figure = plt.figure()
	num_of_images = 60
	for index in range(1, num_of_images + 1):
		plt.subplot(6, 10, index)
		plt.axis('off')
		plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
	figure




# model = classify_digits()
# classify_even_odd(model)
# classify_even_odd()

predict_zipcode('zip1.JPG')


