import matplotlib.pyplot as plt
import os, sys
output = os.path.join(os.path.dirname(__file__), '..', 'output')

def plot_epoch_accuracy(n_epochs, trains, tests, file):
	fig = plt.figure()
	plt.plot(list(range(0,n_epochs+1)), trains, color='blue')
	plt.plot(list(range(0,n_epochs+1)), tests, color='red')
	plt.legend(['Train', 'Test'], loc='upper right')
	plt.xlabel('Number of Epochs')
	plt.ylabel('Accuracy')
	fig.savefig(output + '/cnn_acc_'+file+'.png')