# pytorch_basics3.py
# Intro to building a neural network in pytorch.
# Source: https://pythonprogramming.net/building-deep-learning-neural-
# network-pytorch/
# Source: https://www.youtube.com/watch?v=ixathu7U-LQ
# Pytorch 1.8
# Windows/MacOS/Linux
# Python 3.7


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets


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

	# Note: torch.nn is more of the object oriented operations whereas
	# torch.nn.function is more functional operations. There is already
	# quite a bit of overlap between the two but that is not always the
	# case.

	# Build a neural net inheritting from nn.Module.
	class Net(nn.Module):
		def __init__(self):
			super(Net, self).__init__()
			
			# Define some (fully connected) layers.  Here, the layer
			# being used takes input and output argument shapes. For
			# the first fully connected layer, it takes the flattened
			# 28 x 28 images as input. The output for this layer is 64
			# for the number of neurons in this layer.
			self.fc1 = nn.Linear(28 * 28, 64)

			# Between the first layer, this next one has to take 64 as
			# input because that is what was output by the first layer.
			# The output can still be whatever you want. This rule
			# applies to all hidden (in between) layers.
			self.fc2 = nn.Linear(64, 64)
			self.fc3 = nn.Linear(64, 64)

			# For the output (last) layer, continue to have the input
			# match the last output of the previous layer. The output
			# to the layer is 10 for the number of classes the neural
			# network is expected to output.
			self.fc4 = nn.Linear(64, 10)


		def forward(self, x):
			# Defines the path of the data through the neural network.
			# Here, in a fully connected neural network, the data
			# passes through all the layers. After passing data through
			# each layer, the data also passes through the activation
			# function (relu in this case). On the last (output) layer,
			# use the log softmax activation function to get the
			# probabilities of each class in the output. For the
			# softmax function, be sure to specifiy the dimension to
			# apply the activation on. Output layer will be of shape
			# (batch_size, output_shape), so you want to apply the
			# softmax on dim 1 (the output_shape tensors).
			x = F.relu(self.fc1(x))
			x = F.relu(self.fc2(x))
			x = F.relu(self.fc3(x))
			x = self.fc4(x)
			return F.log_softmax(x, dim=1),
			

	# Initialize the neural net and print it out.
	net = Net()
	print(net)

	# Simulate an input to the neural network.
	x = torch.rand((28, 28))
	print(x)

	# Pass simulated input through the neural network. Be sure to make
	# sure the input shape is correct (batch_shape, input_shape). Here,
	# the -1 refers to an unknown (batch) size dim.
	x = x.view(-1, 28 * 28)
	output = net(x)
	print(output)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()