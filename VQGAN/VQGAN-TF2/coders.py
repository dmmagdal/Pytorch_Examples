# coders.py
# Encoder/Decoder for VQ-GAN.
# Windows/MacOS/Linux
# Python 3.7


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from helper_layers import *


class Encoder(layers.Layer):
	def __init__(self, args):
		super(Encoder, self).__init__()

		channels = [128, 128, 128, 256, 256, 512]
		attn_resolutions = [16]
		num_res_blocks = 2
		resolution = 256
		layers_list = [
			layers.Conv2D(
				channels[0], kernel_size=3, strides=1, padding="same"
			)
		]
		for i in range(len(channels) - 1):
			in_channels = channels[i]
			out_channels = channels[i + 1]
			for j in range(num_res_blocks):
				layers_list.append(
					ResidualBlock(in_channels, out_channels)
				)
				in_channels = out_channels
				if resolution in attn_resolutions:
					layers_list.append(NonLocalBlock(in_channels))
					# layers_list.append(layers.Attention())
			if i != len(channels) - 2:
				layers_list.append(DownSampleBlock(channels[i + 1]))
				resolution //= 2
		layers_list.append(ResidualBlock(channels[-1], channels[-1]))
		layers_list.append(NonLocalBlock(channels[-1]))
		# layers_list.append(layers.Attention())
		layers_list.append(ResidualBlock(channels[-1], channels[-1]))
		layers_list.append(GroupNorm())
		layers_list.append(Swish())
		layers_list.append(
			layers.Conv2D(
				args.latent_dim, kernel_size=3, strides=1, 
				padding="same"
			)
		)
		self.model = keras.Sequential(layers_list)


	def call(self, x):
		return self.model(x)


class Decoder(layers.Layer):
	def __init__(self, args):
		super(Decoder, self).__init__()

		channels = [512, 256, 256, 128, 128]
		attn_resolutions = [16]
		num_res_blocks = 3
		resolution = 16

		in_channels = channels[0]
		layers_list = [
			layers.Conv2D(
				in_channels, kernel_size=3, strides=1, padding="same"
			),
			ResidualBlock(in_channels, in_channels),
			NonLocalBlock(in_channels),
			# layers.Attention(),
			ResidualBlock(in_channels, in_channels)
		]

		for i in range(len(channels)):
			out_channels = channels[i]
			for j in range(num_res_blocks):
				layers_list.append(
					ResidualBlock(in_channels, out_channels)
				)
				in_channels = out_channels
				if resolution in attn_resolutions:
					layers_list.append(NonLocalBlock(in_channels))
					# layers_list.append(layers.Attention())
			if i != 0:
				layers_list.append(UpSampleBlock(in_channels))
				resolution *= 2

		layers_list.append(GroupNorm())
		layers_list.append(Swish())
		layers_list.append(
			layers.Conv2D(
				args.image_channels, kernel_size=3, strides=1, 
				padding="same"
			)
		)
		self.model = keras.Sequential(layers_list)


	def call(self, x):
		return self.model(x)