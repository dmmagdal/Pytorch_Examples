# lpips.py
# Windows/MacOS/Linux
# Python 3.7


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import vgg16


class LPIPS(layers.Layer):
	def __init__(self, input_shape):
		super(LPIPS, self).__init__()

		# self.scaling_layer = ScalingLayer()
		self.channels = [64, 128, 256, 512, 512]
		# self.vgg = vgg16.VGG16()
		self.vgg = VGG16(input_shape)
		self.lins = [
			NetLinLayer(self.channels[0]),
			NetLinLayer(self.channels[1]),
			NetLinLayer(self.channels[2]),
			NetLinLayer(self.channels[3]),
			NetLinLayer(self.channels[4]),
		]


	def call(self, real_x, fake_x):
		# Pass inputs to VGG model, first applying a scaling layer
		# (which is a normalization)
		# features_real = self.vgg(self.scaling_layer(real_x))
		# features_fake = self.vgg(self.scaling_layer(fake_x))
		features_real = self.vgg(real_x)
		features_fake = self.vgg(fake_x)	
		diffs = []

		mse = tf.keras.losses.MeanSquaredError()
		return tf.math.reduce_mean(
			[
				mse(
					norm_tensor(features_real[i]), 
					norm_tensor(features_fake[i])
				)
				for i in range(len(self.channels))
			]
		)

		# Take the difference between each of the different feature
		# channels ((x - y)^2 or L2 difference).
		'''
		for i in range(len(self.channels)):
			# diffs[i] = (
			diffs.append((
				norm_tensor(features_real[i]) -\
				norm_tensor(features_fake[i])
			) ** 2)
		
		# Take the average and return the sum.
		print([spatial_average(self.lins[i].model(diffs[i]))
				# tf.math.reduce_mean(diffs[i], axis=-1)
				for i in range(len(self.channels))])
		return sum(
			[
				spatial_average(self.lins[i].model(diffs[i]))
				# tf.math.reduce_mean(diffs[i], axis=-1)
				for i in range(len(self.channels))
			]
		)
		'''


class ScalingLayer(layers.Layer):
	def __init__(self):
		super(ScalingLayer, self).__init__()
		self.shift = tf.reshape(
			tf.convert_to_tensor([-.030, -.088, -.188]),
			[1, 1, 1, 3]
		)
		self.scale = tf.reshape(
			tf.convert_to_tensor([.458, .448, .450]),
			[1, 1, 1, 3]
		)


	def call(self, x):
		return (x - self.shift) / self.scale


class NetLinLayer(layers.Layer):
	def __init__(self, out_channels=1):
		super(NetLinLayer, self).__init__()
		self.model = keras.Sequential([
			layers.Dropout(0.5), # Default rate is 0.5 on pytorch
			layers.Conv2D(
				out_channels, kernel_size=1, strides=1,
				padding="valid", use_bias=False
			),
		])


	def call(self, x):
		return self.model(x)


class VGG16(layers.Layer):
	def __init__(self, input_shape):
		super(VGG16, self).__init__()

		# Load pretrained VGG16 model (trained on ImageNet). 
		self.vgg = vgg16.VGG16(
			weights="imagenet", include_top=False,
			input_shape=input_shape
		)
		self.vgg.trainable = False
		# vgg = keras.Model(
		# 	inputs=vgg.inputs, outputs=vgg.outputs, name="vgg"
		# )
		layer_outputs = [2, 5, 9, 13, 15]
		self.model = keras.Model(
			inputs=self.vgg.inputs, 
			outputs=[self.vgg.layers[i].output for i in layer_outputs]
		)

		# Create sub-models out of the VGG16 model. Each slice ends
		# right before the maxpooling layer (see vgg's model.summary())
		# which means there are 5 slices of the VGG16 model.
		'''
		self.slice1 = keras.Model(
			inputs=vgg.inputs, outputs=vgg.layers[2].output, 
			name="slice1"
		)
		self.slice2 = keras.Model(
			inputs=vgg.inputs, outputs=vgg.layers[5].output, 
			name="slice2"
		)
		self.slice3 = keras.Model(
			inputs=vgg.inputs, outputs=vgg.layers[9].output, 
			name="slice3"
		)
		self.slice4 = keras.Model(
			inputs=vgg.inputs, outputs=vgg.layers[13].output, 
			name="slice4"
		)
		self.slice5 = keras.Model(
			inputs=vgg.inputs, outputs=vgg.layers[15].output, 
			name="slice5"
		)
		'''


	def call(self, x):
		'''
		slice1 = keras.Sequential(self.vgg.layers[:3])(x)
		relu_1 = slice1
		slice2 = keras.Sequential(self.vgg.layers[3:6])(slice1)
		relu_2 = slice2
		slice3 = keras.Sequential(self.vgg.layers[6:10])(slice2)
		relu_3 = slice3
		slice4 = keras.Sequential(self.vgg.layers[10:14])(slice3)
		relu_4 = slice4
		slice5 = keras.Sequential(self.vgg.layers[14:16])(slice4)
		relu_5 = slice5
		return (relu_1, relu_2, relu_3, relu_4, relu_5)
		'''
		return self.model(x)


# Normalize images by their length to make them unit vector
# @param: x, batch of images.
# @return: returns normalized batch of images.
def norm_tensor(x):
	norm_factor = tf.math.sqrt(tf.math.reduce_sum(
		x ** 2, axis=-1, keepdims=True
	))
	return x / (norm_factor + 1e-10)


# Images have batch_size x channels x width x height --> average over
# width and height channel.
# @param: x, batch of images.
# @return: returns averaged images along width and height.
def spatial_average(x):
	# return x.mean([2, 3], keepdim=True)
	return tf.math.reduce_mean(x, axis=[1, 2], keepdims=True)


if __name__ == '__main__':
	# Test input shape
	input_shape = (256, 256, 3)

	# Test input/output
	x = tf.random.normal(input_shape)
	y = tf.random.normal(input_shape)

	# Create new LPIPS instance
	lpips = LPIPS(input_shape)

	# Get LPIPS loss
	loss = lpips(tf.expand_dims(x, 0), tf.expand_dims(y, 0))
	print(loss)