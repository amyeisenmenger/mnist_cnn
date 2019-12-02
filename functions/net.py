import torch
from torch import nn, optim
import torch.nn.functional as F

# nn.Sequenti
class Net(nn.Module):
	def __init__(self, dropout=False):
		super(Net, self).__init__()
		# DEFINE LAYERS
		self.dropout = dropout
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.pool = nn.MaxPool2d(2, 2)
		if dropout:
			self.conv2_drop = nn.Dropout2d()
		self.fc1 = nn.Linear(20 * 4 * 4, 50)
		self.fc2 = nn.Linear(50, 10)

		# DEFINE CUSTOM PARAMS
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	def forward(self, x):
		# conv1
		# relu
		# pool
		# x = self.conv1(x)
		# x = F.relu(x)
		# x = self.pool(x)
		# x = self.conv2(x)
		# x = F.relu(x)
		# x = self.conv2_drop(x)
		# x = self.pool(x)

		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		# x = x.view(-1, 16 * 4 * 4)
		x = x.view(x.size(0), -1) #same
		# x = x.view(-1, 320) #same
		x = self.fc1(x)
		x = F.relu(x)
		if self.dropout:
			x = F.dropout(x, training=self.training)
		x = self.fc2(x) #same
		# return F.log_softmax(x)
		return x

	def train_epoch(self, train_set, criterion, optimizer):
		self.train()
		correct = 0
		train_loss = 0
		# train by batch
		for data, target in train_set:
			data = data.to(self.device)
			target = target.to(self.device)

			optimizer.zero_grad()
			output = self(data)
			_, preds = torch.max(output, 1)
			loss = criterion(output, target)
			train_loss += loss.item()

			# backward pass and optimzer step
			loss.backward()
			optimizer.step()

			correct += torch.sum(preds == target.data)


		train_loss /= len(train_set.dataset)
		epoch_acc = 100. * correct / len(train_set.dataset)		
		print('Avg. Train  loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
			train_loss, correct, len(train_set.dataset),
			epoch_acc))
		return epoch_acc.item()

	def test_epoch(self, dataset, criterion, optimizer):
		self.eval()
		test_loss = 0
		correct = 0
		with torch.no_grad():
			for data, target in dataset:
				data = data.to(self.device)
				target = target.to(self.device)
				output = self(data)
				test_loss += criterion(output, target).item()
				_, preds = torch.max(output, 1)
				correct += torch.sum(preds == target.data)
		test_loss /= len(dataset.dataset)
		test_acc = 100. * correct / len(dataset.dataset)
		print('Avg. Test loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
			test_loss, correct, len(dataset.dataset),
			test_acc))
		return test_acc.item()