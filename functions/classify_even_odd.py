import torch
from torch import nn, optim

from .net import Net
from .plot_epoch_accuracy import *
from .data import get_datasets_evenodd
# from .visualize_weights import *

import os, sys
output = os.path.join(os.path.dirname(__file__), '..', 'output')

def classify_even_odd(model = None):
	n_epochs = 20
	
	train_set, test_set = get_datasets_evenodd()
	random_seed = 1
	torch.manual_seed(random_seed)

	if model is None:
		# LOAD MODEL FROM FILE
		model = Net(dropout=False)
		model.load_state_dict(torch.load(output + '/model.pth'))

	# FREEZE MODEL WEIGHTS
	for param in model.parameters():
		param.requires_grad = False

	# RESET LAST FULLY CONNECTED LAYER
	num_ftrs = model.fc2.in_features
	model.fc2 = nn.Linear(num_ftrs,2)
	for param in model.fc2.parameters():
	    param.requires_grad = True


	# learning_rate = 0.001
	# momentum = 0.5
	# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
	optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	criterion = nn.CrossEntropyLoss()
	# print(model)

	# RETRAIN
	# model = model.to(model.device)

	# TRACK TRAINING AND TEST ACCURACY BY EPOCH
	train_epoch_accuracy = []
	test_epoch_accuracy = []

	# TRAIN AND EVALUATE
	train_epoch_accuracy.append(model.test_epoch(train_set, criterion, optimizer))
	test_epoch_accuracy.append(model.test_epoch(test_set, criterion, optimizer))
	for epoch in range(1, n_epochs + 1):
		print("Epoch {}".format(epoch))	
		print("Training Accuracy")
		train_acc = model.train_epoch(train_set, criterion, optimizer)
		train_epoch_accuracy.append(train_acc)

		print("Test Accuracy")
		test_acc = model.test_epoch(test_set, criterion, optimizer)
		test_epoch_accuracy.append(test_acc)
		print("")
	# SAVE MODEL TO USE LATER
	torch.save(model.state_dict(), output + '/even_odd_model.pth')
	torch.save(optimizer.state_dict(), output + '/even_odd_optimizer.pth')

	# PLOT ACCURACIES BY EPOCH
	plot_epoch_accuracy(n_epochs, train_epoch_accuracy, test_epoch_accuracy, "even_odd")


