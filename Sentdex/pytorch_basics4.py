# pytorch_basics4.py
# Intro to training a neural network in pytorch.
# Source: https://pythonprogramming.net/training-deep-learning-neural-
# network-pytorch/
# Source: https://www.youtube.com/watch?v=9j-_dOze4IM
# Pytorch 1.8
# Windows/MacOS/Linux
# Python 3.7


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


def main():
	# Define and load the dataset.
	train = datasets.MNIST(
		"", train=True, download=True, 
		transform=transforms.Compose([transforms.ToTensor()])
	)
	test = datasets.MNIST(
		"", train=False, download=True, 
		transform=transforms.Compose([transforms.ToTensor()])
	)

	# Load the training and testing dataset.
	trainset = torch.utils.data.DataLoader(
		train, batch_size=10, shuffle=True,
	)
	testset = torch.utils.data.DataLoader(
		test, batch_size=10, shuffle=False,
	)

	# Build a neural net inheritting from nn.Module.
	class Net(nn.Module):
		def __init__(self):
			super(Net, self).__init__()
			
			# Define some (fully connected) layers.
			self.fc1 = nn.Linear(28 * 28, 64)
			self.fc2 = nn.Linear(64, 64)
			self.fc3 = nn.Linear(64, 64)
			self.fc4 = nn.Linear(64, 10)


		def forward(self, x):
			# Defines the path of the data through the neural network.
			x = F.relu(self.fc1(x))
			x = F.relu(self.fc2(x))
			x = F.relu(self.fc3(x))
			x = self.fc4(x)
			return F.log_softmax(x, dim=1),
			

	# Initialize the neural net and print it out.
	net = Net()
	print(net)

	# Initialize the optimizer. Specify which parameters in the neural
	# net to optimize. In this example, we want to optimize all of
	# them. Also specify the learning rate.
	optimizer = optim.Adam(
		net.parameters(), lr=0.001
	)

	# Training loop.
	EPOCHS = 3
	for epoch in range(EPOCHS):
		for data in trainset:
			# Data is a batch of featuresets and labels.
			X, y = data

			# Zero gradient for the batch.
			net.zero_grad()

			# Pass data through the network.
			output = net(X.view(-1, 28 * 28))[0]

			# Calculate the loss.
			loss = F.nll_loss(output, y)

			# Backpropagate the loss.
			loss.backward()

			# Adjust the weights.
			optimizer.step()
		# Print the loss for the epoch.
		print(loss)

	# Get accuracy.
	correct = 0
	total = 0
	with torch.no_grad(): # Don't want gradients to be calculated.
		for data in trainset:
			X, y = data
			output = net(X.view(-1, 28 * 28))[0]

			# Compare the argmax.
			for idx, i in enumerate(output):
				if torch.argmax(i) == y[idx]:
					correct += 1
				total +=1
	print(f"Accuracy: {round((correct / total) * 100, 3)}%")
	
	# Chart an actual result. Show the input image and print out the
	# output from the neural network.
	plt.imshow(X[0].view(28, 28))
	plt.show()
	print(torch.argmax(net(X[0].view(-1, 28 * 28))[0]))

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()