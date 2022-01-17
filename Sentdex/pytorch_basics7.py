# pytorch_basics7.py
# Intro to gpus in pytorch.
# Source: https://pythonprogramming.net/gpu-deep-learning-neural-
# network-pytorch/
# Source: https://www.youtube.com/watch?v=6gk7giKER6s
# Data Source: https://www.microsoft.com/en-us/download/confirmation.
# aspx?id=54765
# Pytorch 1.8
# Windows/MacOS/Linux
# Python 3.7


import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def main():
	# Check for access to cuda device.
	gpu_available = torch.cuda.is_available()

	# Set GPU device (if available). Most usually use "cuda:0" for the
	# first GPU. Default device is "cpu". To use the set device, add
	# .to_device(device) to the end of any tensor.
	if gpu_available:
		device = torch.device("cuda:0")
		print("Running on the GPU")
	else:
		device = torch.device("cpu")
		print("Running on the CPU")

	# Get the number of GPU devices. It is very easy to send data or
	# train a model on multiple GPU devices (ie encoder-decoder model
	# trains encoder on GPU1 and decoder on GPU2).
	gpu_count = torch.cuda.device_count()
	print(f"Number of GPU devices available: {gpu_count}")

	# Flag to indicate whether to build/rebuild data.
	REBUILD_DATA = False#True


	# Class to preprocess the dataset.
	class DogsVSCats():
		IMG_SIZE = 50 # Image size will be 50 x 50.
		CATS = "PetImages/Cat"
		DOGS = "PetImages/Dog"
		TESTING = "PetImages/Testing"
		LABELS = {CATS: 0, DOGS: 1}
		training_data = []
		catcount = 0
		dogcount = 0


		def make_training_data(self):
			for label in self.LABELS:
				print(label)
				for f in tqdm(os.listdir(label)):
					# Perform actions in try/except block to handle any
					# errors.
					try:
						# Load image in grayscale, resize, and append to
						# training data list along with class label
						# (which is a one-hot encoding).
						path = os.path.join(label, f)
						img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
						img = cv2.resize(
							img, (self.IMG_SIZE, self.IMG_SIZE)
						)
						self.training_data.append(
							[np.array(img), np.eye(2)[self.LABELS[label]]]
						)

						# Increment the respective counters.
						if label == self.CATS:
							self.catcount += 1
						elif label == self.DOGS:
							self.dogcount += 1
					except Exception as e:
						pass

			# Shuffle the data.
			np.random.shuffle(self.training_data)

			# Save the data.
			np.save("training_data.npy", self.training_data)
			print(f"Cats: {self.catcount}")
			print(f"Dogs: {self.dogcount}")


	# Depending on flag, either rebuild the data.
	if REBUILD_DATA:
		dogvcats = DogsVSCats()
		dogvcats.make_training_data()

	training_data = np.load("training_data.npy", allow_pickle=True)
	print(len(training_data))

	# Convolutional neural network.
	class Net(nn.Module):
		def __init__(self):
			super(Net, self).__init__()
			# 2D convolutional layers with a 5x5 kernel.
			self.conv1 = nn.Conv2d(1, 32, 5)
			self.conv2 = nn.Conv2d(32, 64, 5)
			self.conv3 = nn.Conv2d(64, 128, 5)

			# Helps determine input shape from last conv layer to
			# first linear layer (after flattening).
			x = torch.randn(50, 50).view(-1, 1, 50, 50)
			self.to_linear = None
			self.convs(x)

			# Linear layers
			self.fc1 = nn.Linear(self.to_linear, 512)
			self.fc2 = nn.Linear(512, 2)


		def convs(self, x):
			# Pass input through the convolutional layers.
			x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
			x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
			x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

			# Set input shape for first linear layer (works to flatten
			# output from convolutional layers).
			if self.to_linear is None:
				self.to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
			return x


		def forward(self, x):
			x = self.convs(x)
			x = x.view(-1, self.to_linear)
			x = F.relu(self.fc1(x))
			x = self.fc2(x)
			return F.softmax(x, dim=1)


	# Initialize the convolutional neural net. This will take the
	# neural network and send it to be initialized the specified
	# device from above.
	net = Net().to(device)

	# Isolate and normalize input and output data.
	x = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
	x = x / 255.0
	y = torch.Tensor([i[1] for i in training_data])

	# Set up validation split.
	VAL_PCT = 0.1
	val_size = int(len(x) * VAL_PCT)
	print(val_size)

	# Split the data into test and training sets.
	train_x = x[:-val_size]
	train_y = y[:-val_size]
	test_x = x[-val_size:]
	test_y = y[-val_size:]
	print(len(train_x))
	print(len(test_x))

	BATCH_SIZE = 100
	EPOCHS = 3


	def train(net):
		# Initialize optimizer and loss function.
		optimizer = optim.Adam(net.parameters(), lr=0.001)
		loss_function = nn.MSELoss()
		
		# Training loop.
		for epoch in range(EPOCHS):
			for i in tqdm(range(0, len(train_x), BATCH_SIZE)):
				# Send batch data to the GPU so it can be passed
				# through the model.
				batch_x = train_x[i:i + BATCH_SIZE].view(-1, 1, 50, 50)
				batch_x = batch_x.to(device)
				batch_y = train_y[i:i + BATCH_SIZE].to(device)
				batch_y = batch_y.to(device)

				# Zero gradients. 
				net.zero_grad()

				# Pass batch through neural network.
				outputs = net(batch_x)

				# Calculate loss and back propagate.
				loss = loss_function(outputs, batch_y)
				loss.backward()
				optimizer.step()
			print(f"Epoch: {epoch}, Loss: {loss}")

	
	def test(net):
		# Check accuracy with validation/test set.
		correct = 0
		total = 0
		with torch.no_grad():
			for i in tqdm(range(len(test_x))):
				real_class = torch.argmax(test_y[i]).to(device)
				net_out = net(test_x[i].view(-1, 1, 50, 50).to(device))[0]
				predicted_class = torch.argmax(net_out)
				if predicted_class == real_class:
					correct += 1
				total += 1
		print(f"Accuracy: {round((correct / total) * 100, 3)}")


	# Train and evalulate the neural network.
	train(net)
	test(net)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()