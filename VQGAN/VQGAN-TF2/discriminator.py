# discriminator.py
# Windows/MacOS/Linux
# Python 3.7


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# class Discriminator(layers.Layer):
class Discriminator(keras.Model):
	def __init__(self, args, num_filers_last=64, n_layers=3):
		super(Discriminator, self).__init__()

		layers_list = [
			layers.Conv2D(
				num_filers_last, kernel_size=4, strides=2, 
				padding="same",
			),
			layers.LeakyReLU(0.2),
		]
		num_filers_mult = 1

		for i in range(1, n_layers + 1):
			num_filers_mult_last = num_filers_mult
			num_filers_mult = min(2 ** i, 8)
			layers_list += [
				layers.Conv2D(
					num_filers_last * num_filers_mult, kernel_size=4,
					strides=2 if i < n_layers else 1, padding="same",
					use_bias=False
				),
				layers.BatchNormalization(),
				layers.LeakyReLU(0.2),
			]

		layers_list.append(
			layers.Conv2D(1, kernel_size=4, strides=1, padding="same")
		)
		self.model = keras.Sequential(layers_list)


	def call(self, x):
		return self.model(x)