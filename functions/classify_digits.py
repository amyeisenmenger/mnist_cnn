import torch
from torch import nn, optim
from torch.optim import lr_scheduler


from .net import Net
from .plot_epoch_accuracy import *
from .data import get_datasets
from .visualize_weights import *

import os, sys
output = os.path.join(os.path.dirname(__file__), '..', 'output')
def classify_digits():
	n_epochs = 19

	learning_rate = 0.001
	momentum = 0.4

	log_interval = 10
	random_seed = 1
	# torch.backends.cudnn.enabled = False
	torch.manual_seed(random_seed)
	# CREATE MODEL
	model = Net(dropout=False)
	# model.load_state_dict(torch.load(output + '/model.pth'))
	# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
	optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	criterion = nn.CrossEntropyLoss()
	# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.01)

	model = model.to(model.device)

	# LOAD DATASETS
	train_set, test_set = get_datasets()

	# TRACK TRAINING AND TEST ACCURACY BY EPOCH
	train_epoch_accuracy = []
	test_epoch_accuracy = []

	# TRAIN AND EVALUATE
	train_epoch_accuracy.append(model.test_epoch(train_set, criterion, optimizer))
	test_epoch_accuracy.append(model.test_epoch(test_set, criterion, optimizer))
	for epoch in range(1, n_epochs + 1):
		print("Epoch {}".format(epoch))	
		train_epoch_accuracy.append(model.train_epoch(train_set, criterion, optimizer))
		print("Test Accuracy")
		test_epoch_accuracy.append(model.test_epoch(test_set, criterion, optimizer))
		print("")
	# SAVE MODEL TO USE LATER
	torch.save(model.state_dict(), output + '/model.pth')
	torch.save(optimizer.state_dict(), output + '/optimizer.pth')

	# PLOT ACCURACIES BY EPOCH
	plot_epoch_accuracy(n_epochs, train_epoch_accuracy, test_epoch_accuracy, "digits")

	# VISIUALIZE WEIGHTS OF FIRST LAYER
	layer = model.conv1.weight.data #.numpy()
	visualize_weights(layer)
	return model


