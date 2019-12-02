
from .extract_connected_components import extract
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
# from torch import nn, optim
import matplotlib.pyplot as plt
from .net import Net
import os, sys
output = os.path.join(os.path.dirname(__file__), '..', 'output')

def predict_zipcode(filename):
	digits, digit_lefts = extract(filename)
	model = Net()
	model.load_state_dict(torch.load(output + '/model.pth'))
	print(model)
	# model.eval()
	zipcode = []
	for digit in digits:
		test_transforms = transforms.Compose([transforms.Resize(28),
											transforms.ToTensor(),
										])
		# print(digit)

		digit_tensor = test_transforms(digit).float()
		# plt.imshow(digit_tensor)
		# plt.show()
		digit_tensor.unsqueeze_(0)
		im_in = Variable(digit_tensor)
		# im_in = squeeze(im_in)
		# print(digit_tensor)
		result = model(im_in)
		_, predicted = torch.max(result, 1)
		zipcode.append(predicted.item())

	final = ''.join([str(x) for _,x in sorted(zip(digit_lefts,zipcode))])
	return final
	