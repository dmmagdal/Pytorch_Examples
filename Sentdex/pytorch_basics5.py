# pytorch_basics5.py
# Intro to convolutional neural networks in pytorch.
# Source: https://pythonprogramming.net/convolutional-neural-networks-
# deep-learning-neural-network-pytorch/
# Source: https://www.youtube.com/watch?v=9aYuQmMJvjA
# Data Source: https://www.microsoft.com/en-us/download/confirmation.
# aspx?id=54765
# Pytorch 1.8
# Windows/MacOS/Linux
# Python 3.7


import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def main():
	# Flag to indicate whether to build/rebuild data. This is just to
	# tell the program whether or not to preprocess data before feeding
	# it to the neural network. Some programs process the data and save
	# it so as not to need to reprocess the data every time the program
	# is run.
	REBUILD_DATA = True


	# Class to preprocess the dataset.
	class DogsVSCats():
		IMG_SIZE = 50 # Image size will be 50 x 50.
		CATS = "PetImages/Cat"
		DOGS = "PetImages/Dog"
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
	print(training_data[0])

	# Display an image.
	plt.imshow(training_data[1][0], cmap="gray")
	plt.show()

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()