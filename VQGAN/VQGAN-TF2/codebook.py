# codebook.py
# Windows/MacOS/Linux
# Python 3.7


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class CodeBook(layers.Layer):
	# Using vector quantizer from Keras example:
	# https://keras.io/examples/generative/vq_vae/
	def __init__(self, args):
		super(CodeBook, self).__init__()

		self.num_codebook_vectors = args.num_codebook_vectors
		self.latent_dim = args.latent_dim
		self.beta = args.beta

		w_init = tf.random_uniform_initializer(
			-1.0 / self.num_codebook_vectors, 
			1.0 / self.num_codebook_vectors
		)
		self.embedding = tf.Variable(
			initial_value=w_init(
				shape=(self.latent_dim, self.num_codebook_vectors), 
				dtype="float32"
			), trainable=True
		)


	def call(self, z):
		# Calculate the input shape of the inputs and then flatten the
		# inputs keeping "latent_dim" intact.
		z_flattened = tf.reshape(z, [-1, self.latent_dim])

		# L2 normalized distance between inputs and codebook.
		d = tf.math.reduce_sum(z_flattened ** 2, axis=1, keepdims=True) +\
			tf.math.reduce_sum(self.embedding ** 2, axis=0) - 2 *\
			(tf.linalg.matmul(z_flattened, self.embedding))

		# Get encoding indices from distance.
		min_encoding_indices = tf.math.argmin(d, axis=1)
		encodings = tf.one_hot(
			min_encoding_indices, self.num_codebook_vectors
		)
		z_q = tf.linalg.matmul(
			encodings, self.embedding, transpose_b=True
		)
		z_q = tf.reshape(z_q, z.shape)
		# Same as above.
		# Get codebook vectors from embedding matrix. Reshape back to
		# original shape.
		# z_q = tf.reshape(self.embedding(min_encoding_indices), z.shape)

		# Codebook loss (apply stop gradient function). Note: There
		# seems to be some confusion on where to apply multiplying the
		# beta, either at the commitment loss or codebook loss. Keras
		# example peforms this at the commitment loss but the pytorch
		# example applies this to the other loss.
		commitment_loss = self.beta * tf.math.reduce_mean(
			(tf.stop_gradient(z_q) - z) ** 2
		)
		codebook_loss = tf.reduce_mean(
			(z_q - tf.stop_gradient(z)) ** 2
		)
		loss = commitment_loss + codebook_loss
		# Same as above losses.
		# loss = tf.math.reduce_mean((z_q - z) ** 2) +\
		# 	self.beta + tf.math.reduce_mean((z_q - z) ** 2)
		self.add_loss(loss)

		# Straight through estimator.
		z_q = z + tf.stop_gradient(z_q - z)
		# z_q = tf.transpose(z_q, perm=[0, 1, 2, 3])

		# Return quantized latent vectors, indices of the codebook
		# vectors, and loss.
		return z_q, min_encoding_indices, loss