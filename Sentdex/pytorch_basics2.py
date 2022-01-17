# pytorch_basics2.py
# Intro to data operations in pytorch.
# Source: https://pythonprogramming.net/data-deep-learning-neural-
# network-pytorch/
# Source: https://www.youtube.com/watch?v=i2yPxY2rOzs
# Pytorch 1.8
# Windows/MacOS/Linux
# Python 3.7


import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


def main():
	# Define training and testing datasets. Here, we're using the MNIST
	# dataset. First argument is where the data is going ("" means it
	# will be local). Transform applies the transformations (functions
	# you want to apply) to the data. transforms.ToTensor() will
	# convert the data to tensors.
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

	# Iterate over the data (print a batch).
	for data in trainset:
		print(data)
		break

	# Grab the first element from the batch. Extract the input (x) and
	# output (y) values from that element.
	x, y = data[0][0], data[1][0]
	print(x)
	print(y)

	# Visualze that first element input (x). Note that the input image
	# is a (1, 28, 28) tensor. A typical 28 x 28 image converted to
	# grayscale and converted into a tensor would have a shape of
	# (28 x 28). Pytorch puts a 1 there in front of the image.
	print(x.shape)
	plt.imshow(x.view(28, 28)) # Reshape to allow it to be shown.
	plt.show()

	# Balance the dataset. Note that the MNIST dataset is already quite
	# well balanced but this is to demonstrate how to walk through a
	# dataset to see whether it is balanced or not.

	# Confirm if the dataset is balanced by counting how the number of
	# outputs for each output class.
	total = 0
	counter_dict = {
		1: 0, 2: 0, 3: 0, 
		4: 0, 5: 0, 6: 0,
		7: 0, 8: 0, 9: 0,
		0: 0,
	}
	for data in trainset:
		Xs, ys = data
		for y in ys:
			counter_dict[int(y)] += 1
			total += 1
	print(counter_dict)

	# Prints the percentage distribution. Here we can see that it's
	# already pretty well balanced.
	for i in counter_dict:
		print(f"{i}: {(counter_dict[i] / total) * 100}")

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()