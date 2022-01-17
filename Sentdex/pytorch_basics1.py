# pytorch_basics1.py
# Intro to some very basic functions of pytorch.
# Source: https://pythonprogramming.net/introduction-deep-learning-
# neural-network-pytorch/
# Source: https://www.youtube.com/watch?v=BzcBsTou0C0
# Pytorch 1.8
# Windows/MacOS/Linux
# Python 3.7


import torch


def main():
	# Tensor (array) multiplication.
	x = torch.Tensor([5, 3])
	y = torch.Tensor([2, 1])
	print(x * y)

	# Initialize a tensor of shape (2, 5) of 0's.
	x = torch.zeros([2, 5])
	print(x)
	print(x.shape)

	# Randomly initialize a tensor of shape (2, 5).
	y = torch.rand([2, 5])
	print(y)

	# View (reshape) a tensor from (2, 5) into a (1, 10).
	y = y.view([1, 10])

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()