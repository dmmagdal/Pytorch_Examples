# layers.py
# TF 2.0 implementation of the helper.py module.
# Windows/MacOS/Linux
# Python 3.7


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


class GroupNorm(layers.Layer):
	# def __init__(self, channels):
	def __init__(self):
		super(GroupNorm, self).__init__()
		self.group_norm = tfa.layers.GroupNormalization(
			# groups=32, 
			epsilon=1e-6,
		)


	def call(self, x):
		return self.group_norm(x)


class Swish(layers.Layer):
	def __init__(self):
		super(Swish, self).__init__()
		self.act = layers.Activation("swish")


	def call(self, x):
		return self.act(x)


class ResidualBlock(layers.Layer):
	def __init__(self, in_channels, out_channels):
		super(ResidualBlock, self).__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels

		self.block = keras.Sequential([
			GroupNorm(),
			Swish(),
			layers.Conv2D(
				out_channels, kernel_size=3, strides=1, padding="same"
			),
			GroupNorm(),
			Swish(),
			layers.Conv2D(
				out_channels, kernel_size=3, strides=1, padding="same"
			),
		])

		if in_channels != out_channels:
			self.channel_up = layers.Conv2D(
				out_channels, kernel_size=1, strides=1, padding="valid"
			)


	def call(self, x):
		if self.in_channels != self.out_channels:
			return self.channel_up(x) + self.block(x)
		else:
			return x + self.block(x)


class UpSampleBlock(layers.Layer):
	def __init__(self, channels):
		super(UpSampleBlock, self).__init__()
		self.conv = layers.Conv2D(
			channels, kernel_size=3, strides=1, padding="same",
		)
		self.upsample = layers.UpSampling2D(size=2)


	def call(self, x):
		x = self.upsample(x)
		return self.conv(x)


class DownSampleBlock(layers.Layer):
	def __init__(self, channels):
		super(DownSampleBlock, self).__init__()
		self.conv = layers.Conv2D(
			channels, kernel_size=3, strides=2, padding="valid",
		)
		# self.conv2 = layers.Conv2D(
		# 	channels, kernel_size=3, strides=2, padding="same",
		# )
		self.conv2 = layers.MaxPooling2D(strides=2, padding="same",)


	def call(self, x):
		# pad = tf.convert_to_tensor([0, 1, 0, 1], dtype=tf.int32)
		pad = tf.convert_to_tensor([[0, 0], [0, 1], [0, 1], [0, 0]], dtype=tf.int32)
		x = tf.pad(x, pad, mode="CONSTANT", constant_values=0)

		# return self.conv2(self.conv(x))
		return self.conv(x)


class NonLocalBlock(layers.Layer):
	def __init__(self, channels):
		super(NonLocalBlock, self).__init__()

		self.attn = layers.Attention()
		self.group_norm = GroupNorm()
		self.q = layers.Conv2D(
			channels, kernel_size=1, strides=1, padding="valid"
		)
		self.k = layers.Conv2D(
			channels, kernel_size=1, strides=1, padding="valid"
		)
		self.v = layers.Conv2D(
			channels, kernel_size=1, strides=1, padding="valid"
		)
		self.proj_out = layers.Conv2D(
			channels, kernel_size=1, strides=1, padding="valid"
		)


	def call(self, x):
		h_ = self.group_norm(x)
		q = self.q(h_)
		k = self.k(h_)
		v = self.v(h_)

		return self.attn([q, k, v])

		'''
		b, h, w, c = tf.shape(q)

		q = tf.reshape(q, [b, h * w, c])
		q = tf.transpose(q, perm=[0, 1, 2])
		k = tf.reshape(k, [b, h * w, c])
		v = tf.reshape(v, [b, h * w, c])

		attn = tf.linalg.matmul(q, k)
		attn = attn * (int(c) ** (-0.5))
		attn = tf.nn.softmax(attn, axis=-1)
		attn = tf.transpose(attn, perm=[0, 1, 2])

		A = tf.linalg.matmul(v, attn)
		A = tf.reshape(A, [b, h, w, c])

		return x + A
		'''