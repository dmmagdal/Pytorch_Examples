# training_vqgan.py
# Windows/MacOS/Linux
# Python 3.7


import os
import argparse
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from discriminator import Discriminator
from lpips import LPIPS
from vqgan import VQGAN
import matplotlib.pyplot as plt


class TrainVQGAN:
	def __init__(self, args):
		# input_shape = (256, 256, 3)
		input_shape = (32, 32, 3)
		self.vqgan = VQGAN(args)
		self.discriminator = Discriminator(args)
		self.perceptual_loss = LPIPS(input_shape)
		self.opt_vq, self.opt_disc = self.configure_optimizers(args)

		self.prepare_training()

		self.train(args)


	def configure_optimizers(self, args):
		lr = args.learning_rate
		opt_vq = keras.optimizers.Adam(
			lr=lr, epsilon=1e-8, beta_1=args.beta1, beta_2=args.beta2
		)
		opt_disc = keras.optimizers.Adam(
			lr=lr, epsilon=1e-8, beta_1=args.beta1, beta_2=args.beta2
		)
		return opt_vq, opt_disc


	@staticmethod
	def prepare_training():
		os.makedirs("results", exist_ok=True)
		os.makedirs("checkpoints", exist_ok=True)


	def train(self, args):
		dataset = tfds.load("cifar10")
		train_dataset = dataset["train"]
		test_dataset = dataset["test"]

		steps_per_epoch = len(train_dataset)
		for epoch in range(args.epochs):
			with tqdm(range(len(train_dataset))) as pbar:
				for i, imgs in zip(pbar, train_dataset):
					imgs = tf.cast(
						tf.expand_dims(imgs["image"], 0), 
						dtype=tf.float32
					)
					with tf.GradientTape() as gan_tape, tf.GradientTape() as disc_tape:
						decoded_images, _, q_loss = self.vqgan(imgs)

						disc_real = self.discriminator(imgs)
						disc_fake = self.discriminator(decoded_images)

						disc_factor = self.vqgan.adopt_weight(
							args.disc_factor, epoch * steps_per_epoch + i,
							threshold=args.disc_start
						)

						perceptual_loss = self.perceptual_loss(
							imgs, decoded_images
						)
						# rec_loss = torch.abs(imgs - decoded_images)
						rec_loss = tf.math.abs(imgs - decoded_images)
						perceptual_rec_loss = args.perceptual_loss_factor *\
							perceptual_loss + args.rec_loss_factor *\
							rec_loss
						# perceptual_rec_loss = perceptual_rec_loss.mean()
						perceptual_rec_loss = tf.math.reduce_mean(
							perceptual_rec_loss
						)
						# g_loss = -torch.mean(disc_fake)
						g_loss = -tf.math.reduce_mean(disc_fake)

						lambDa = self.vqgan.calculate_lambda(
							perceptual_rec_loss, g_loss
						)
						vq_loss = perceptual_rec_loss + q_loss +\
							disc_factor * g_loss
							# disc_factor * lambDa * g_loss

						# d_loss_real = torch.mean(F.relu(1.0 - disc_real))
						# d_loss_fake = torch.mean(F.relu(1.0 + disc_fake))
						d_loss_real = tf.math.reduce_mean(
							tf.nn.relu(1.0 - disc_real)
						)
						d_loss_fake = tf.math.reduce_mean(
							tf.nn.relu(1.0 + disc_fake)
						)
						gan_loss = disc_factor * 0.5 *\
							(d_loss_real + d_loss_fake)

					gan_grads = gan_tape.gradient(
						vq_loss, self.vqgan.trainable_weights
					)
					disc_grads = disc_tape.gradient(
						gan_loss, self.discriminator.trainable_weights
					)

					self.opt_vq.apply_gradients(zip(
						gan_grads, self.vqgan.trainable_weights
					))
					self.opt_disc.apply_gradients(zip(
						disc_grads, 
						self.discriminator.trainable_weights
					))

					if i % 10 == 0:
						real_fake_images = tf.concat(
							(
								imgs / 255,
								decoded_images 
								# tf.math.scalar_mul(
								# 	0.5,
								# 	tf.math.add(
								# 		decoded_images,
								# 		1
								# 	),
								# )
							),
							axis=0
						)
						plt.figure(figsize=(16, 8))
						plt.subplot(231)
						plt.title("Real Image")
						plt.imshow(real_fake_images[0, :, :, :])
						plt.subplot(232)
						plt.title("Fake Image")
						plt.imshow(real_fake_images[1, :, :, :])
						plt.savefig(f"./results/VQGAN_Sample{epoch + 1}_{i}.png")
						plt.close()

					pbar.set_postfix(
						VQ_Loss=np.round(
							vq_loss.numpy().item(), 5
						),
						GAN_Loss=np.round(
							gan_loss.numpy().item(), 3
						),
					)
					pbar.update(0)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="VQGAN")
	parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
	parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
	parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
	parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
	parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
	parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
	parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
	parser.add_argument('--batch-size', type=int, default=6, help='Input batch size for training (default: 6)')
	parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 50)')
	parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate (default: 0.0002)')
	parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param (default: 0.0)')
	parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param (default: 0.999)')
	parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator (default: 0)')
	parser.add_argument('--disc-factor', type=float, default=1., help='')
	parser.add_argument('--rec-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
	parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')

	args = parser.parse_args()
	args.dataset_path = r"C:\Users\dome\datasets\flowers"

	train_vqgan = TrainVQGAN(args)