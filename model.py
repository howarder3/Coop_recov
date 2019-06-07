import tensorflow as tf
import numpy as np
import time
import datetime
import os

from glob import glob
from six.moves import xrange

# --------- self define function ---------
# ops: layers structure
from ops import *

# utils: for loading data, model
from utils import *

class Coop_pix2pix(object):
	def __init__(self, sess, 
				epoch = 1000, 
				batch_size = 1,
				picture_amount = 99999,
				image_size = 256, output_size = 256,
				input_pic_dim = 3, output_pic_dim = 3,	
				langevin_revision_steps = 1, 
				langevin_step_size = 0.002,
				descriptor_learning_rate = 0.01,
				generator_learning_rate = 0.0001,
				encoder_learning_rate = 0.0001,
				dataset_name='edges2handbags', dataset_dir ='./test_datasets', 
				output_dir='./output_dir', checkpoint_dir='./checkpoint_dir', log_dir='./log_dir'):
		"""
		args:
			sess: tensorflow session
			batch_size: how many pic in one group(batch), iteration(num_batch) = picture_amount/batch_size
			input_pic_dim: input picture dimension : colorful = 3, grayscale = 1
			output_pic_dim: output picture dimension : colorful = 3, grayscale = 1 
			langevin_revision_steps = langevin revision steps
			descriptor_learning_rate = descriptor learning rate
			generator_learning_rate = generator learning rate

		"""


		self.sess = sess
		self.epoch = epoch		
		self.batch_size = batch_size
		self.picture_amount = picture_amount
		self.image_size = image_size
		self.output_size = output_size
		self.input_pic_dim = input_pic_dim
		self.output_pic_dim = output_pic_dim

		# descriptor langevin steps
		self.langevin_revision_steps = langevin_revision_steps
		self.langevin_step_size = langevin_step_size

		# learning rate
		self.descriptor_learning_rate = descriptor_learning_rate 
		self.generator_learning_rate  = generator_learning_rate 
		self.encoder_learning_rate  = encoder_learning_rate 
		# print(1e-5) # 0.00001
		

		self.dataset_dir = dataset_dir
		self.dataset_name = dataset_name

		self.output_dir = output_dir
		self.checkpoint_dir = checkpoint_dir
		self.log_dir = log_dir
		self.epoch_startpoint = 0

		self.sigma1 = 0.016
		self.sigma2 = 0.3
		self.beta1 = 0.5

		self.input_latent = tf.placeholder(tf.float32,
				[self.batch_size, 1, 1, 100],
				name='input_latent')
		self.input_picture = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='input_picture')

		# self.input_generated_B = tf.placeholder(tf.float32,
		# 		[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
		# 		name='input_generated_B')
		self.input_real_data_B = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='input_real_data_B')
		self.input_real_data_A = tf.placeholder(tf.float32,
				[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
				name='input_real_data_A')
		# self.input_recovered_A = tf.placeholder(tf.float32,
		# 		[self.batch_size, self.image_size, self.image_size, self.input_pic_dim],
		# 		name='input_recovered_A')

	def build_model(self):

		# generator 
		self.latent = self.encoder(self.input_picture, reuse = False)

		self.color_pic = self.colorful_decoder(self.input_latent, reuse = False)
		self.sketch_pic = self.sketch_decoder(self.input_latent, reuse = False)

		t_vars = tf.trainable_variables()
		self.encoder_vars = [var for var in t_vars if var.name.startswith('encoder')]
		self.color_decode_vars = [var for var in t_vars if var.name.startswith('color_decode')]
		self.sketch_decode_vars = [var for var in t_vars if var.name.startswith('sketch_decode')]

		# # descriptor variables
		# print("\n------  self.des_vars  ------\n")
		# for var in self.des_vars:
		# 	print(var)


		# encode variables
		print("\n------  self.encoder_vars  ------\n")
		for var in self.encoder_vars:
			print(var)
		print("")

		# color_decode variables
		print("\n------  self.color_decode_vars  ------\n")
		for var in self.color_decode_vars:
			print(var)
		print("")

		# sketch_decode variables
		print("\n------  self.sketch_decode_vars  ------\n")
		for var in self.sketch_decode_vars:
			print(var)
		print("")
		

		# # descriptor loss functions
		# self.des_loss = tf.reduce_sum(tf.subtract(tf.reduce_mean(described_revised_B, axis=0), tf.reduce_mean(described_real_data_B, axis=0)))

		# self.des_optim = tf.train.AdamOptimizer(self.descriptor_learning_rate, beta1=self.beta1).minimize(self.des_loss, var_list=self.des_vars)


		# # Compute Mean square error(MSE) for generated data and real data
		# self.mse_loss = tf.reduce_mean(
  #           tf.pow(tf.subtract(tf.reduce_mean(self.input_generated_B, axis=0), tf.reduce_mean(self.input_revised_B, axis=0)), 2))


		# # generator loss functions
		# self.gen_loss = tf.reduce_sum(tf.reduce_mean(1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(self.input_revised_B - self.generated_B), axis=0))
		
		# self.gen_optim = tf.train.AdamOptimizer(self.generator_learning_rate, beta1=self.beta1).minimize(self.gen_loss, var_list=self.gen_vars)


		# recover loss functions
		self.encoder_loss = L1_distance(self.latent, self.input_latent) # tf.reduce_mean((self.recovered_A - self.input_real_data_A)**2)

		# tf.reduce_sum(tf.reduce_mean(1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(self.recovered_A - self.input_real_data_A), axis=0))

		self.encoder_optim = tf.train.AdamOptimizer(self.encoder_learning_rate, beta1=self.beta1).minimize(self.encoder_loss, var_list=self.encoder_vars)

		self.saver = tf.train.Saver(max_to_keep=10)

	def train(self,sess):

		# build model
		self.build_model()

		# prepare training data
		training_data = glob('{}/{}/train/*.jpg'.format(self.dataset_dir, self.dataset_name))

		# iteration(num_batch) = picture_amount/batch_size
		self.num_batch = min(len(training_data), self.picture_amount) // self.batch_size

		# initialize training
		sess.run(tf.global_variables_initializer())

		# sample picture initialize
		# sample_results = np.random.randn(num_batch, self.image_size, self.image_size, 3)

		# counter initialize
		counter = 1
		counter_end = self.epoch * self.num_batch  # 200 * num_batch 

		# load checkpoint
		if self.load(self.checkpoint_dir):
			print(" [v] Loading checkpoint success!!!")
		else:
			print(" [!] Loading checkpoint failed...")

		# start training	
		start_time = time.time()
		print("time: {} , Start training model......".format(str(datetime.timedelta(seconds=int(time.time()-start_time)))))
		

		# print("self.counter = ",self.counter)

		for epoch in xrange(self.epoch_startpoint, self.epoch): # how many epochs to train

			for index in xrange(self.num_batch): # num_batch
				# find picture list index*self.batch_size to (index+1)*self.batch_size (one batch)
				# if batch_size = 2, get one batch = batch[0], batch[1]
				batch_files = training_data[index*self.batch_size:(index+1)*self.batch_size] 

				# load data : list format, amount = one batch
				batch = [load_data(batch_file) for batch_file in batch_files]
				batch_images = np.array(batch).astype(np.float32)

				# data domain A and data domain B
				data_A = batch_images[:, :, :, : self.input_pic_dim] 
				data_B = batch_images[:, :, :, self.input_pic_dim:self.input_pic_dim+self.output_pic_dim] 

				# step 1: sketch -> latent
				latent_var = sess.run(self.latent, feed_dict={self.input_picture: data_A})
				print(" ------ step 1 finish ------ ")

				# step 2: latent -> colorful picture
				color_pic = sess.run(self.color_pic, feed_dict={self.input_latent: latent_var})
				print(" ------ step 2 finish ------ ")

				# step 3: colorful picture -> latent
				recovered_latent_var = sess.run(self.latent, feed_dict={self.input_picture: color_pic})
				print(" ------ step 3 finish ------ ")

				# step 4: latent -> sketch
				sketch_pic = sess.run(self.sketch_pic, feed_dict={self.input_latent: recovered_latent_var})
				print(" ------ step 4 finish ------ ")




				# step D2: update descriptor net
				# descriptor_loss , _ = sess.run([self.des_loss, self.des_optim],
    #                               		feed_dict={self.input_real_data_B: data_B, self.input_revised_B: revised_B})

				# # step G2: update generator net
				# generator_loss , _ = sess.run([self.gen_loss, self.gen_optim],
    #                               		feed_dict={self.input_real_data_A: data_A, self.input_revised_B: revised_B}) # self.input_revised_B: revised_B,

				# self.input_generated_B: generated_B,

				# print(descriptor_loss)

				# step R2: update recover net
				recover_loss , _ = sess.run([self.rec_loss, self.rec_optim],
                                  		feed_dict={self.input_latent: latent_var, self.input_picture: color_pic})


				# Compute Mean square error(MSE) for generated data and revised data
				# mse_loss = sess.run(self.mse_loss, feed_dict={self.input_revised_B: revised_B, self.input_generated_B: generated_B})


				# put picture in sample picture
				# sample_results[index : (index + 1)] = revised_B

				print("Epoch: [{:4d}] [{:4d}/{:4d}] time: {}, eta: {}, d_loss: {:.4f}, g_loss: {:.4f}, rec_loss: {:.4f}"
					.format(epoch, index, self.num_batch, 
						str(datetime.timedelta(seconds=int(time.time()-start_time))),
							str(datetime.timedelta(seconds=int((time.time()-start_time)*(counter_end-(self.epoch_startpoint*self.num_batch)-counter)/counter))),
								 descriptor_loss, generator_loss, recover_loss))

				if np.mod(counter, 10) == 1:
					save_images(data_A, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_01_input_data_A.png'.format(self.output_dir, epoch, index))
					save_images(data_B, [self.batch_size, 1],
						'./{}/ep{:02d}_{:04d}_02_input_data_B.png'.format(self.output_dir, epoch, index))
					

				counter += 1

			self.save(self.checkpoint_dir, epoch)

			
			# # print("time: {:.4f} , Epoch: {} ".format(time.time() - start_time, epoch))
			# print('Epoch #{:d}, avg.descriptor loss: {:.4f}, avg.generator loss: {:.4f}, avg.L2 distance: {:4.4f}, '
			# 	'time: {:.2f}s'.format(epoch, np.mean(des_loss_avg), np.mean(gen_loss_avg), np.mean(mse_avg), time.time() - start_time))

	def encoder(self, input_image, reuse=False):
		with tf.variable_scope("encoder", reuse=reuse):

			conv1 = conv2d(input_image, 64, kernal=(5, 5), strides=(2, 2), padding="SAME", activate_func="leaky_relu", name="conv1")

			conv2 = conv2d(conv1, 128, kernal=(3, 3), strides=(2, 2), padding="SAME", activate_func="leaky_relu", name="conv2")

			conv3 = conv2d(conv2, 256, kernal=(3, 3), strides=(1, 1), padding="SAME", activate_func="leaky_relu", name="conv3")

			fc = fully_connected(conv3, 100, name="fc")

			print("latent shape:",fc.shape)

			return fc

	def colorful_decoder(self, input_data, reuse=False):
		with tf.variable_scope("color_decode", reuse=reuse):
			print(input_data.shape)
			input_data = tf.reshape(input_data, [-1, 1, 1, 100])
			print(input_data.shape)
			trans_conv1 = transpose_conv2d(input_data, (None, self.image_size // 16, self.image_size // 16, 512), kernal=(4, 4),
								strides=(1, 1), padding="VALID", activate_func="leaky_relu", name="trans_conv1")
			trans_conv2 = transpose_conv2d(trans_conv1, (None, self.image_size // 8, self.image_size // 8, 256), kernal=(5, 5),
								strides=(2, 2), padding="SAME", activate_func="leaky_relu", name="trans_conv2")
			trans_conv3 = transpose_conv2d(trans_conv2, (None, self.image_size // 4, self.image_size // 4, 128), kernal=(5, 5),
								strides=(2, 2), padding="SAME", activate_func="leaky_relu", name="trans_conv3")
			trans_conv4 = transpose_conv2d(trans_conv3, (None, self.image_size // 2, self.image_size // 2, 64), kernal=(5, 5),
								strides=(2, 2), padding="SAME", activate_func="leaky_relu", name="trans_conv4")
			trans_conv5 = transpose_conv2d(trans_conv4, (None, self.image_size, self.image_size, 3), kernal=(5, 5),
								strides=(2, 2), padding="SAME", activate_func=None, name="trans_conv5")
			result = tf.nn.tanh(trans_conv5)

			print("color shape:",result.shape)

			return result


	def sketch_decoder(self, input_data, reuse=False):
		with tf.variable_scope("sketch_decode", reuse=reuse):

			print(input_data.shape)
			input_data = tf.reshape(input_data, [-1, 1, 1, 100])
			print(input_data.shape)
			trans_conv1 = transpose_conv2d(input_data, (None, self.image_size // 16, self.image_size // 16, 512), kernal=(4, 4),
								strides=(1, 1), padding="VALID", activate_func="leaky_relu", name="trans_conv1")
			trans_conv2 = transpose_conv2d(trans_conv1, (None, self.image_size // 8, self.image_size // 8, 256), kernal=(5, 5),
								strides=(2, 2), padding="SAME", activate_func="leaky_relu", name="trans_conv2")
			trans_conv3 = transpose_conv2d(trans_conv2, (None, self.image_size // 4, self.image_size // 4, 128), kernal=(5, 5),
								strides=(2, 2), padding="SAME", activate_func="leaky_relu", name="trans_conv3")
			trans_conv4 = transpose_conv2d(trans_conv3, (None, self.image_size // 2, self.image_size // 2, 64), kernal=(5, 5),
								strides=(2, 2), padding="SAME", activate_func="leaky_relu", name="trans_conv4")
			trans_conv5 = transpose_conv2d(trans_conv4, (None, self.image_size, self.image_size, 3), kernal=(5, 5),
								strides=(2, 2), padding="SAME", activate_func=None, name="trans_conv5")
			result = tf.nn.tanh(trans_conv5)

			print("sketch shape:",result.shape)

			return result


	def save(self, checkpoint_dir, step):
		saver_name = "epoch"

		model_dir = "{}".format(self.dataset_name)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
					os.path.join(checkpoint_dir, saver_name),
					global_step=step)


	def load(self, checkpoint_dir):
		print(" [*] Loading checkpoint...")
		model_dir = "{}".format(self.dataset_name)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
		if checkpoint and checkpoint.model_checkpoint_path:
			checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
			print(" [v] Found checkpoint name: {}".format(checkpoint_name))
			self.epoch_startpoint = int(checkpoint_name.split("epoch-", 1)[1])+1
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, checkpoint_name))
			return True
		else:
			return False
