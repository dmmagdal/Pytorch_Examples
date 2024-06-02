# vqgan.py
# Windows/MacOS/Linux
# Python 3.7


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from coders import Encoder, Decoder
from codebook import CodeBook


# class VQGAN(layers.Layer):
class VQGAN(keras.Model):
	def __init__(self, args):
		super(VQGAN, self).__init__()
		self.encoder = Encoder(args)
		self.decoder = Decoder(args)
		self.codebook = CodeBook(args)
		self.quant_conv = layers.Conv2D(
			args.latent_dim, kernel_size=1
		)
		self.post_quant_conv = layers.Conv2D(
			args.latent_dim, kernel_size=1
		)


	def call(self, imgs):
		encoded_images = self.encoder(imgs)
		quant_conv_encoded_images = self.quant_conv(encoded_images)
		codebook_mapping, codebook_indices, q_loss = self.codebook(
			quant_conv_encoded_images
		)
		post_quant_conv_mapping = self.post_quant_conv(
			codebook_mapping
		)
		decoded_images = self.decoder(post_quant_conv_mapping)

		return decoded_images, codebook_indices, q_loss


	def encode(self, imgs):
		encoded_images = self.encoder(imgs)
		quant_conv_encoded_images = self.quant_conv(encoded_images)
		codebook_mapping, codebook_indices, q_loss = self.codebook(
			quant_conv_encoded_images
		)
		return codebook_mapping, codebook_indices, q_loss


	def decode(self, z):
		post_quant_conv_mapping = self.post_quant_conv(z)
		decoded_images = self.decoder(post_quant_conv_mapping)
		return decoded_images


	def calculate_lambda(self, perceptual_loss, gan_loss):
		# Adaptive weight loss lambda. See page 4 of taming
		# transformers paper (https://arxiv.org/pdf/2012.09841.pdf).
		# A reddit post points out that they have experienced the value
		# to be very small and removing it should not affect the
		# model's performance (https://www.reddit.com/r/
		# MachineLearning/comments/q6ilpd/d_adaptive_loss_weight_in_
		# vqgan_paper/).
		# last_layer = self.decoder.layers[-1]
		# last_layer_weight = last_layer.weights
		pass


	@staticmethod
	def adopt_weight(disc_factor, i, threshold, value=0.0):
		if i < threshold:
			disc_factor = value
		return disc_factor