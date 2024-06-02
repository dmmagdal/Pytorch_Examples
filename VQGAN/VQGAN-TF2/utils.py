# utils.py
# Windows/MacOS/Linux
# Python 3.7


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt


def load_data(args):
	pass


def plot_images(images):
	x = images["input"]
	reconstruction = images["rec"]
	half_sample = images["half_sample"]
	full_sample = images["full_sample"]

	fig, axarr = plt.subplots(1, 4)
	axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
	axarr[1].imshow(
		reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0)
	)
	axarr[2].imshow(
		half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0)
	)
	axarr[3].imshow(
		full_sample.cpu().detach().numpy()[0].transpose(1, 2, 0)
	)
	plt.show()