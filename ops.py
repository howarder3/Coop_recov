import tensorflow as tf

def conv2d(input_image, output_dim, kernal=(5, 5), strides=(2, 2), padding='SAME', activate_func=None, name="conv2d"):
	with tf.variable_scope(name):  

		weight = tf.get_variable('weight',[kernal[0], kernal[1], input_image.get_shape()[-1],output_dim], 	
						initializer=tf.random_normal_initializer(stddev=0.01))

		conv_result = tf.nn.conv2d(input_image, weight, strides=[1, strides[0], strides[1], 1], padding=padding)

		bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))

		result = tf.nn.bias_add(conv_result, bias)

		if activate_func == "leaky_relu":
			result = leaky_relu(result)
		elif activate_func == "relu":
			result = relu(result)

		return result

def transpose_conv2d(input_image, output_shape, kernal=(5, 5), strides=(2, 2), padding='SAME', activate_func=None, name="transpose_conv2d"):
	with tf.variable_scope(name):

		# output_shape = list(output_shape)
		# output_shape[0] = tf.shape(input_image)[0]
		# output_shape[0] = input_image.get_shape()[0]
		# output_shape[0] = 1

		# output_shape = [int(input_image.get_shape()[0]), int(output_shape[1]), int(output_shape[2]), int(output_shape[-1])]

		# strides = [1, strides[0], strides[1], 1]
		strides = [1, 2, 2, 1]


		previous_layer_shape = input_image.get_shape()

		# output shape = [batch_size, input_shape*2, input_shape*2, output_dim]
		output_shape = [int(previous_layer_shape[0]), int(previous_layer_shape[1])*2, int(previous_layer_shape[2])*2, int(output_shape[-1])]

		weight = tf.get_variable('weight',[5, 5, int(output_shape[-1]), previous_layer_shape[-1]],
								initializer=tf.random_normal_initializer(stddev=0.02))


		# weight = tf.get_variable('weight', [kernal[0], kernal[1], output_shape[-1], input_image.get_shape()[-1]],
		#                     initializer=tf.random_normal_initializer(stddev=0.005))

		print(output_shape)
		# print(strides)

		# x = (tf.stack(output_shape, axis=0))
		# print(str(x))

		# trans_conv_result = tf.nn.conv2d_transpose(input_image, weight, output_shape=output_shape, strides=strides, padding=padding)
		trans_conv_result = tf.nn.conv2d_transpose(input_image, weight, output_shape=output_shape, strides=strides)

		bias = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(0.0))

		result = tf.nn.bias_add(trans_conv_result, bias)

		if activate_func == "leaky_relu":
			result = leaky_relu(result)
		elif activate_func == "relu":
			result = relu(result)

		return result


def fully_connected(input_image, output_dim, name="fc"):
	with tf.variable_scope(name):

		weight = tf.get_variable('fc_weight', [input_image.get_shape()[1], input_image.get_shape()[2], input_image.get_shape()[-1], output_dim],
		   				 initializer=tf.random_normal_initializer(stddev=0.01))

		conv_result = tf.nn.conv2d(input_image, weight, strides=[1,1,1,1], padding='VALID')

		bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))

		result = tf.nn.bias_add(conv_result, bias)

		return result


# leaky_relu
def leaky_relu(x, leak=0.2, name="leaky_relu"):
	return tf.maximum(x, leak*x)		


# relu
def relu(x, name="relu"):
	return tf.nn.relu(x)


def L1_distance(input_data, target):
    return tf.reduce_mean(tf.abs(input_data - target))


def L2_distance(input_data, target):
    return tf.reduce_mean((input_data-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))








