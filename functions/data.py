import torch
from torchvision import datasets, transforms
import os, sys
input = os.path.join(os.path.dirname(__file__), '..', 'input/')

batch_size_train = 64 #can increase for mroe 
batch_size_test = 1000

def get_datasets():
	trns = transforms.Compose([
								transforms.ToTensor(),
								transforms.Normalize((0.1307,), (0.3081,))
							])
	train_set = torch.utils.data.DataLoader(
		datasets.MNIST(input, train=True, download=True, transform=trns),
			batch_size=batch_size_train, shuffle=True, num_workers = 8)

	test_set = torch.utils.data.DataLoader(
		datasets.MNIST(input, train=False, download=True, transform=trns), 
			batch_size=batch_size_test, shuffle=True, num_workers = 8)
	return train_set, test_set

	

def get_datasets_evenodd():
	trns = transforms.Compose([
								transforms.ToTensor(),
								transforms.Normalize((0.1307,), (0.3081,))
							])
	train_data = datasets.MNIST(input, train=True, download=True, transform=trns)
	train_data.targets = train_data.targets % 2 #odd now labelled 1, even now 0 

	train_set = torch.utils.data.DataLoader(
		train_data, batch_size=batch_size_train, shuffle=True, num_workers = 8)

	test_data = datasets.MNIST(input, train=False, download=True, transform=trns)
	test_data.targets = test_data.targets % 2

	test_set = torch.utils.data.DataLoader(
		test_data, batch_size=batch_size_test, shuffle=True, num_workers = 8)

	return train_set, test_set