# This file is part of the hessianlearn package
#
# hessianlearn is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
#
# hessianlearn is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# If not, see <http://www.gnu.org/licenses/>.
#
# Author: Tom O'Leary-Roseberry
# Contact: tom.olearyroseberry@utexas.edu

from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
if int(tf.__version__[0]) > 1:
	import tensorflow.compat.v1 as tf
	tf.disable_v2_behavior()
from abc import ABC, abstractmethod

def my_flatten(tensor_list):
	"""
	This function flattens a list of tensors into a single numpy vector
		-tensor_list: list of tensors stored as numpy arrays
	"""
	flattened_list = []

	for tensor in tensor_list:
		flattened_list.append(tf.reshape(tensor,[np.prod(tensor.shape)]))
	return tf.concat(flattened_list,axis=0)

def initialize_indices(shapes):
	"""
	This function takes a list of shapes and creates an indexing scheme
	used for mapping lists of tensors to contiguous chunks of a vector
		-shapes: list of tensor shapes
	returns:
		-indices: list of indices
	"""
	indices = []
	index = 0
	for shape in shapes:
		index += int(np.prod(shape))
		indices.append(index)
	return indices



class Problem(ABC):
	"""
	This class implements the description of the neural network training problem.

	It takes a neural network model and defines loss function and derivatives
	Also defines update operations.
	"""
	def __init__(self,NeuralNetwork,dtype = tf.float32):
		"""
		The Problem parent class constructor takes a neural network model (typically from tf.keras.Model)
		Children class implement different loss functions which are implemented by the method _initialize_loss

			-NeuralNetwork: keras model for neural network
		"""
		# Boolean to indicate if only input data should be passed into loss function
		self._is_autoencoder = False
		self._is_gan = False
		# Data type
		self._dtype = dtype

		# Initialize the neural network(s)
		self._initialize_network(NeuralNetwork)

		# Define loss function and accuracy in initialize_loss
		self._initialize_loss()

		# Define derivative quantities for output
		self._initialize_derivatives()

		# Define assignment operations
		self._initialize_assignment_ops()

		
		
		


	@property
	def NN(self):
		return self._NN

	@property
	def dtype(self):
		return self._dtype

	@property
	def shapes(self):
		return self._shapes

	@property
	def indices(self):
		return self._indices

	@property
	def w(self):
		return self._flat_w

	@property
	def gradient(self):
		return self._gradient

	@property
	def norm_g(self):
		return self._norm_g

	@property
	def dimension(self):
		return self._dimension

	@property
	def w_hat(self):
		return self._w_hat

	@property
	def H_action(self):
		return self._H_action

	@property
	def H_quadratic(self):
		return self._H_quadratic

	@property
	def loss(self):
		return self._loss

	@property
	def accuracy(self):
		return self._accuracy

	@property
	def is_autoencoder(self):
		return self._is_autoencoder		

	@property
	def is_gan(self):
		return self._is_gan
	

	@property
	def output_dimension(self):
		return self._output_dimension
	

	def _initialize_network(self,NeuralNetwork):
		"""
		This method defines the neural network model
			-NeuralNetwork: the neural network as a tf.keras.model.Model

		Must set member variable self._output_shape
		"""
		self._NN = NeuralNetwork
		self.x = self.NN.inputs[0]
		
		self.y_prediction = self.NN(self.x)

		# Why does the following not work:
		# self.y_true = self.NN.outputs[0]
		# Instead I have to use a placeholder for true output

		output_shape = self.NN.output_shape
		self.y_true = tf.placeholder(self.dtype, output_shape,name='output_placeholder')


		if len(self.y_prediction.shape) > 2:
			self._output_dimension =  1.
			for shape in self.y_prediction.shape[1:]:
				self._output_dimension *= shape.value
		else:
			self._output_dimension = self.y_prediction.shape[-1].value

	def _initialize_loss(self):
		"""
		This method defines the loss as a function of the neural network and 
		placeholder variable for the true data.
		Child class of Problem must implement this method

		This method implements self._loss and self._accuracy
		"""
		raise NotImplementedError("Child class must implement method initialize_loss") 

	def _initialize_derivatives(self):
		"""
		This method defines derivative quantities for the loss function
		"""
		# Assign trainable variables to member variable w "weights"
		self._w = self._NN.trainable_weights
		self._flat_w = my_flatten(self._w)

		self._dimension = self._flat_w.shape[0].value
		# Once loss is defined gradients can be instantiated
		self._gradient = my_flatten(tf.gradients(self.loss,self._w, name = 'gradient'))

		self._norm_g = tf.sqrt(tf.reduce_sum(self.gradient*self.gradient))
		# Initialize vector for Hessian mat-vecs
		self._w_hat = tf.placeholder(self.dtype,self.dimension )
		# Define (g,dw) inner product
		self._g_inner_w_hat = tf.tensordot(self._w_hat,self._gradient,axes = [[0],[0]])
		# Define Hessian action Hdw
		self._H_action = my_flatten(tf.gradients(self._g_inner_w_hat,self._w,stop_gradients = self._w_hat,name = 'hessian_action'))
		# Define Hessian quadratic forms
		self._H_quadratic = tf.tensordot(self._w_hat,self._H_action,axes = [[0],[0]])

	def _initialize_assignment_ops(self):
		"""
		This method defines operations for updating and assigment used during training
		"""
		self._shapes = [tuple(int(wii) for wii in wi.shape) for wi in self._w]
		self._indices = initialize_indices(self.shapes)

		layer_descriptors = {}
		first_indices = [0]+self.indices[:-1]
		for wi,shape,first_index,second_index in zip(self._w,self._shapes,first_indices,self._indices):
			layer_dict = {}
			layer_dict['shape'] = shape
			layer_dict['indices'] = (first_index,second_index)
			layer_descriptors[wi.name] = layer_dict

		self.layer_descriptors = layer_descriptors

		self._update_placeholder = tf.placeholder(self.dtype,[self._dimension],name = 'update_placeholder')
		self._assignment_placeholder = tf.placeholder(self.dtype,[self._dimension],name = 'assignment_placeholder')
		split_indices = []
		index0 = 0
		for index in self._indices:
			split_indices.append(index - index0)
			index0 = index

		unpacked_update = tf.split(self._update_placeholder,split_indices,axis = 0)
		unpacked_assignment = tf.split(self._assignment_placeholder,split_indices,axis = 0)
		update = [tf.reshape(update_,shape) for update_,shape in zip(unpacked_update,self.shapes)]
		assignment = [tf.reshape(assignment_,shape) for assignment_,shape in zip(unpacked_assignment,self.shapes)]

		update_ops = []
		update_and_w = list(zip(update,self._w))
		for v, w in reversed(update_and_w):
			with tf.control_dependencies(update_ops):
				update_ops.append(tf.assign_add(w,v))
		self._update_ops = tf.group(*update_ops)

		assignment_ops = []
		assignment_and_w = list(zip(assignment,self._w))
		for v, w in reversed(assignment_and_w):
			with tf.control_dependencies(assignment_ops):
				assignment_ops.append(tf.assign(w,v))
		self._assignment_ops = tf.group(*assignment_ops)


	def _zero_layers(self,array_like_w,list_of_layer_names):
		"""
		This method takes an array like w, and names of layers 
		which are to be zeroed. This is useful for example in zeroing
		biases when one wants to use a simple Gaussian for initializing weights
			-array_like_w: the array which is to be modified
			-list_of_layer_names: list of layer names to be zeroed.
		"""
		assert array_like_w.shape == self._flat_w.shape
		# Make sure that each layer name is in the layer_descriptors dictionary
		for layer_name in list_of_layer_names:
			assert layer_name in self.layer_descriptors.keys()

		for layer_name in list_of_layer_names:
			indices = self.layer_descriptors[layer_name]['indices']
			array_like_w[indices[0]:indices[1]] = np.zeros(indices[1] - indices[0])

		return array_like_w

	def _set_layer(self,array_like_w,array_like_layer,layer_name):
		"""
		This method takes an array like w, and name of a layers to be set,
		and the data which it is to be set to. Useful when one wants to set 
		specific layers to good initial guesses
			-array_like_w: the array which is to be modified
			-list_of_layer_names: list of layer names to be zeroed.
		"""
		assert array_like_w.shape == self._flat_w.shape
		assert layer_name in self.layer_descriptors.keys()
		indices = self.layer_descriptors[layer_name]['indices']
		assert len(array_like_layer) == indices[1] - indices[0]
		array_like_w[indices[0]:indices[1]] = array_like_layer
		return array_like_w

	def _partition_dictionaries(self,data_dictionary,n_partitions):
		"""
		This method partitions one data dictionary into n_partitions.
		The data that are used in the child class are problem specific 
		and for this reason the child class implements this method.
		"""
		raise NotImplementedError("Child class should implement method _partition_dictionaries") 



class ClassificationProblem(Problem):
	"""
	This class implements the description of basic classification problems. 

	"""
	def __init__(self,NeuralNetwork,loss_type = 'cross_entropy',dtype = tf.float32):
		"""
		The class constructor is like that of the parent except it takes an additional flag 
		for the type of the loss function to be employed.
			-NeuralNetwork: the neural network represented as a keras Model
			-loss_type: a string for the loss type used in classification
			-dtype: the data type used.

		"""
		assert loss_type in ['cross_entropy','least_squares','mixed','squared_hinge']
		self._loss_type = loss_type
		super(ClassificationProblem,self).__init__(NeuralNetwork,dtype = dtype)

	@property
	def loss_type(self):
		return self._loss_type
	
	@property
	def accuracy(self):
		return self._accuracy

	@property
	def rel_error(self):
		return self._rel_error
	

	def _initialize_loss(self):
		"""
		This method is called during the construction of the problem class.
		The member variable self.loss_type decides what type of loss function
		is instantiated.
		Regardless of the loss type this function also defines classification
		accuracy to be used to monitor training.
		"""
		if self.loss_type == 'cross_entropy':
			# print('self.y_true.shape = ',self.y_true.shape)
			with tf.name_scope('loss'):
				# self._loss = tf.reduce_mean(-tf.reduce_sum(self.y_true*tf.nn.log_softmax(self.y_prediction), [1]))
				# self._loss = tf.reduce_mean(-tf.reduce_sum(self.y_true*tf.nn.log_softmax(self.y_prediction)+
															# (tf.ones_like(self.y_true)-self.y_true)*tf.nn.log_softmax(tf.ones_like(self.y_prediction) - self.y_prediction), [1]))
				self._loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(self.y_true, self.y_prediction,from_logits=True))											
		elif self.loss_type == 'least_squares':
			with tf.name_scope('loss'):
				self._loss = tf.losses.mean_squared_error(labels=self.y_true, predictions=self.y_prediction)
		elif self.loss_type == 'mixed':
			mse = tf.losses.mean_squared_error(labels=self.y_true, predictions=self.y_prediction)
			xe = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(self.y_true, self.y_prediction,from_logits=True))	
			self._loss = mse + xe
		elif self.loss_type == 'squared_hinge':
			with tf.name_scope('loss'):
				self._loss = tf.reduce_mean(tf.keras.losses.SquaredHinge()(self.y_true,self.y_prediction))
		else:
			raise
		with tf.name_scope('rel_error'):
			self._rel_error = tf.sqrt(tf.reduce_mean(tf.pow(self.y_true-self.y_prediction,2))\
							/tf.reduce_mean(tf.pow(self.y_true,2)))

		with tf.name_scope('accuracy'):
			y_prediction_sm = tf.nn.softmax(self.y_prediction)
			correct_prediction = tf.equal(tf.argmax(self.y_prediction, 1), tf.argmax(self.y_true,1))
			correct_prediction = tf.cast(correct_prediction, self.dtype)
			self._accuracy = tf.reduce_mean(correct_prediction)


	def _partition_dictionaries(self,data_dictionary,n_partitions):
		"""
		This method partitions one data dictionary into n_partitions.
		For classification this is just inputs to outputs.
		"""
		assert type(n_partitions) == int
		data_xs = data_dictionary[self.x]
		data_ys = data_dictionary[self.y_true]
		if n_partitions > len(data_xs):
			n_partitions = len(data_xs)
		chunk_size = int(data_xs.shape[0]/n_partitions)
		dictionary_partitions = []
		for chunk_i in range(n_partitions):
			# Array slicing should be a view, not a copy
			# So this should not be a memory issue
			my_chunk_x = data_xs[chunk_i*chunk_size:(chunk_i+1)*chunk_size]
			my_chunk_y = data_ys[chunk_i*chunk_size:(chunk_i+1)*chunk_size]
			dictionary_partitions.append({self.x:my_chunk_x, self.y_true: my_chunk_y})
		return dictionary_partitions



class RegressionProblem(Problem):
	"""
	This class implements the description of basic regression problems. 

	"""
	def __init__(self,NeuralNetwork,y_mean = None,dtype = tf.float32):
		"""
		The constructor for this class takes:
			-NeuralNetwork: the neural network represented as a tf.keras Model

		"""
		if y_mean is not None:
			self.y_mean = tf.constant(y_mean,dtype = dtype)
		else:
			self.y_mean = None
		super(RegressionProblem,self).__init__(NeuralNetwork,dtype = dtype)

	@property
	def variance_reduction(self):
		return self._variance_reduction

	@property
	def rel_error(self):
		return self._rel_error
	
	

	def _initialize_loss(self):
		"""
		This method defines the least squares loss function as well as relative error and accuracy
		"""
		with tf.name_scope('loss'):
			self._loss = tf.losses.mean_squared_error(labels=self.y_true, predictions=self.y_prediction)
		with tf.name_scope('rel_error'):
			self._rel_error = tf.sqrt(tf.reduce_mean(tf.pow(self.y_true-self.y_prediction,2))\
							/tf.reduce_mean(tf.pow(self.y_true,2)))
		self._accuracy = 1. - self._rel_error
		with tf.name_scope('variance_reduction'):
			# For use in constructing a regressor to serve as a control variate.
			# 
			assert self.y_mean is not None
			self._variance_reduction = tf.sqrt(tf.reduce_mean(tf.pow(self.y_true-self.y_prediction,2))\
							/tf.reduce_mean(tf.pow(self.y_true - self.y_mean,2)))
		with tf.name_scope('mad'):
			try:
				import tensorflow_probability as tfp
				absolute_deviation = tf.math.abs(self.y_true - self.y_prediction)
				self.mad = tfp.stats.percentile(absolute_deviation,50.0,interpolation = 'midpoint')
			except:
				self.mad = None

	def _partition_dictionaries(self,data_dictionary,n_partitions):
		"""
		This method partitions one data dictionary into n_partitions.
		For regression this is just inputs to outputs.
		"""
		assert type(n_partitions) == int
		data_xs = data_dictionary[self.x]
		data_ys = data_dictionary[self.y_true]
		if n_partitions > len(data_xs):
			n_partitions = len(data_xs)
		chunk_size = int(data_xs.shape[0]/n_partitions)
		dictionary_partitions = []
		for chunk_i in range(n_partitions):
			# Array slicing should be a view, not a copy
			# So this should not be a memory issue
			my_chunk_x = data_xs[chunk_i*chunk_size:(chunk_i+1)*chunk_size]
			my_chunk_y = data_ys[chunk_i*chunk_size:(chunk_i+1)*chunk_size]
			dictionary_partitions.append({self.x:my_chunk_x, self.y_true: my_chunk_y})
		return dictionary_partitions


class AutoencoderProblem(Problem):
	"""
	This class implements the description of basic autoencoder problems. 

	"""
	def __init__(self,NeuralNetwork,dtype = tf.float32):
		"""
		The constructor for this class takes:
			-NeuralNetwork: the tf.keras Model representation of the neural network
		"""
		super(AutoencoderProblem,self).__init__(NeuralNetwork,dtype = dtype)
		self._is_autoencoder = True

	@property
	def rel_error(self):
		return self._rel_error


	def _initialize_loss(self):
		"""
		This method defines the least squares loss function as well as relative error and accuracy
		"""
		with tf.name_scope('loss'): # 
			self._loss = tf.reduce_mean(tf.pow(self.x-self.y_prediction,2)) 
			self._rel_error = tf.sqrt(tf.reduce_mean(tf.pow(self.x-self.y_prediction,2))\
							/tf.reduce_mean(tf.pow(self.x,2)))
			self._accuracy = 1. - self.rel_error

	def _partition_dictionaries(self,data_dictionary,n_partitions):
		"""
		This method partitions one data dictionary into n_partitions.
		For autoencoders this is just inputs.
		"""
		assert type(n_partitions) == int
		data_xs = data_dictionary[self.x]
		if n_partitions > len(data_xs):
			n_partitions = len(data_xs)
		chunk_size = int(data_xs.shape[0]/n_partitions)
		dictionary_partitions = []
		for chunk_i in range(n_partitions):
			# Array slicing should be a view, not a copy
			# So this should not be a memory issue
			my_chunk_x = data_xs[chunk_i*chunk_size:(chunk_i+1)*chunk_size]
			dictionary_partitions.append({self.x:my_chunk_x})
		return dictionary_partitions
			


class VariationalAutoencoderProblem(Problem):
	"""
	This class implements the description of basic variational autoencoder problems. 

	"""
	def __init__(self,NeuralNetwork,z_mean,z_log_sigma,loss_type = 'least_squares',dtype = tf.float32):
		"""
		The constructor for this class takes:
			-NeuralNetwork: the tf.keras Model representation of the neural network
			-z_mean: The mean for the Gaussian latent variable probability model
						to be learned during VAE training
			-z_log_sigma: The diagonal covariance for the Gaussian latent variable 
							probability distribution model to be learned during VAE training
			-loss_type: the loss type used for the VAE model
		"""
		assert loss_type in ['cross_entropy','least_squares']
		self._loss_type = loss_type
		self.z_mean = z_mean
		self.z_log_sigma = z_log_sigma
		
		super(VariationalAutoencoderProblem,self).__init__(NeuralNetwork,dtype = dtype)
		self._is_autoencoder = True

	@property
	def loss_type(self):
		return self._loss_type

	@property
	def rel_error(self):
		return self._rel_error
	

	def _initialize_loss(self):
		"""
		This method initializes the loss function used for variational autoencoder training

		"""
		with tf.name_scope('loss'): # 
			if self.loss_type == 'cross_entropy':
				pass
			elif self.loss_type == 'least_squares':
				# VAE tutorials rescale the mse by the input dimension
				least_squares_loss = np.prod(self.NN.input_shape[1:])*tf.reduce_mean(tf.pow(self.x-self.y_prediction,2)) 
				kl_loss = tf.reduce_mean(-0.5*tf.reduce_sum(1 + self.z_log_sigma - tf.pow(self.z_mean,2) - tf.exp(self.z_log_sigma),axis = -1))
				# kl_loss = 0.0
				self._loss = least_squares_loss + kl_loss
			else:
				raise

		self._rel_error = tf.sqrt(tf.reduce_mean(tf.pow(self.x-self.y_prediction,2))\
						/tf.reduce_mean(tf.pow(self.x,2)))
		self._accuracy = 1. - self.rel_error

	def _partition_dictionaries(self,data_dictionary,n_partitions):
		"""
		This method partitions one data dictionary into n_partitions.
		For autoencoders this is just inputs.
		"""
		assert type(n_partitions) == int
		data_xs = data_dictionary[self.x]
		if n_partitions > len(data_xs):
			n_partitions = len(data_xs)
		chunk_size = int(data_xs.shape[0]/n_partitions)
		dictionary_partitions = []
		for chunk_i in range(n_partitions):
			# Array slicing should be a view, not a copy
			# So this should not be a memory issue
			my_chunk_x = data_xs[chunk_i*chunk_size:(chunk_i+1)*chunk_size]
			dictionary_partitions.append({self.x:my_chunk_x})
		return dictionary_partitions


class GenerativeAdversarialNetworkProblem(Problem):
	"""
	This class implements the description of basic generative adversarial network problems. 

	"""
	def __init__(self,generator,discriminator,loss_type = 'least_squares',dtype = tf.float32):
		"""
		The constructor for this class takes:
			-generator: the tf.keras.model.Model description of the generator neural network
			-discriminator: the tf.keras.model.Model description of the discriminator neural network
		"""
		assert loss_type in ['cross_entropy','least_squares']
		self._loss_type = loss_type

		super(GenerativeAdversarialNetworkProblem,self).__init__([generator,discriminator],dtype = dtype)

		self._is_gan = True

	@property
	def loss_type(self):
		return self._loss_type

	@property
	def generator(self):
		return self._generator
	
	@property
	def discriminator(self):
		return self._discriminator

	@property
	def noise(self):
		return self._noise

	@property
	def noise_dimension(self):
		return self._noise_dimension
	
	
	@property
	def generated_images(self):
		return self._generated_images
	
	@property
	def input_images(self):
		return self._input_images
	
	@property
	def real_predictions(self):
		return self._real_predictions
	
	@property
	def fake_predictions(self):
		return self._fake_predictions

	@property
	def generator_loss(self):
		return self._generator_loss
	
	@property
	def discriminator_loss(self):
		return self._discriminator_loss
	
	@property
	def generator_w(self):
		return self._generator_w
	
	@property
	def discriminator_w(self):
		return self._discriminator_w
	
	

	def _initialize_network(self, NeuralNetworks):
		"""
		This method defines the neural network model
			-NeuralNetworks: list of [generator,discriminator] as tf.keras.model.Model 

		Must set member variable self._output_shape
		"""
		self._generator, self._discriminator = NeuralNetworks

		self._noise = self.generator.inputs[0]

		self._noise_dimension = self._noise.shape[-1].value

		self._generated_images = self.generator(self._noise)

		self.x = self.discriminator.inputs[0]

		self._real_prediction = self.discriminator(self.x)

		self._fake_prediction = self.discriminator(self._generated_images)

		if len(self._real_prediction.shape) > 2:
			self._output_shape =  1.
			for shape in self._real_prediction.shape[1:]:
				self._output_dimension *= shape.value
		else:
			self._output_dimension = self._real_prediction.shape[-1].value



	def _initialize_loss(self):
		"""
		This method initializes the loss function used for variational autoencoder training

		"""
		with tf.name_scope('loss'): # 
			if self.loss_type == 'cross_entropy':
				self._generator_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(\
							tf.ones_like(self._fake_prediction), self._fake_prediction,from_logits=True))
				self._discriminator_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(\
							tf.ones_like(self._real_prediction), self._real_prediction,from_logits=True))+\
							 tf.reduce_mean(tf.keras.losses.categorical_crossentropy(\
							tf.zeros_like(self._fake_prediction), self._fake_prediction,from_logits=True))
				self._loss = self._generator_loss + self._discriminator_loss

			elif self.loss_type == 'least_squares':
				self._generator_loss = tf.losses.mean_squared_error(labels=tf.ones_like(self._fake_prediction), predictions=self._fake_prediction)
				self._discriminator_loss = tf.losses.mean_squared_error(labels=tf.ones_like(self._real_prediction), predictions=self._real_prediction)+\
											tf.losses.mean_squared_error(labels=tf.zeros_like(self._fake_prediction), predictions=self._fake_prediction)
				self._loss = self._generator_loss + self._discriminator_loss

			# self._loss = self._generator_loss + self._discriminator_loss

	def _initialize_derivatives(self):
		"""
		This method defines derivative quantities for the loss function

		The convention here will be w = [generator_w,discriminator_w]
		"""
		# Assign trainable variables to member variable w "weights"
		self._generator_w = self._generator.trainable_weights

		self._discriminator_w = self._discriminator.trainable_weights
		# w = [generator_w,discriminator_w]
		self._w = self._generator.trainable_weights + self._discriminator.trainable_weights

		self._flat_w = my_flatten(self._w)

		self._dimension = self._flat_w.shape[0].value

		# Gradients are partials wrt each loss and network respectively
		self._generator_gradient = my_flatten(tf.gradients(self._generator_loss,self._generator_w,name = 'generator_gradient'))
		self._generator_dimension = self._generator_gradient.shape[0].value

		self._discriminator_gradient = my_flatten(tf.gradients(self._discriminator_loss,self._discriminator_w,name = 'generator_gradient'))
		self._discriminator_dimension = self._discriminator_gradient.shape[0].value

		# Concatenate partial gradients into one gradient 
		self._gradient = tf.concat([self._generator_gradient,self._discriminator_gradient],axis = 0)

		self._norm_g = tf.sqrt(tf.reduce_sum(self.gradient*self.gradient))

		# Hessian mat-vecs
		self._w_hat = tf.placeholder(self.dtype, self.dimension)
		# Split the placeholder
		self._generator_w_hat,self._discriminator_w_hat = tf.split(self._w_hat,[self._generator_dimension,self._discriminator_dimension])

		# Define generator (g,dw) inner product
		self._generator_g_inner_w_hat = tf.tensordot(self._generator_w_hat,self._generator_gradient,axes = [[0],[0]])
		# Define discriminator (g,dw) inner product
		self._discriminator_g_inner_w_hat = tf.tensordot(self._discriminator_w_hat,self._discriminator_gradient,axes = [[0],[0]])
		# Define Hessian action Hdw
		self._generator_H_action = my_flatten(tf.gradients(self._generator_g_inner_w_hat,self._generator_w,\
										stop_gradients = self._generator_w_hat,name = 'generator_hessian_action'))

		self._discriminator_H_action = my_flatten(tf.gradients(self._discriminator_g_inner_w_hat,self._discriminator_w,\
										stop_gradients = self._discriminator_w_hat,name = 'discriminator_hessian_action'))

		self._H_action = tf.concat([self._generator_H_action,self._discriminator_H_action],axis = 0)

		# Define Hessian quadratic forms
		self._H_quadratic = tf.tensordot(self._w_hat,self._H_action,axes = [[0],[0]])

	def _partition_dictionaries(self,data_dictionary,n_partitions):
		"""
		This method partitions one data dictionary into n_partitions.
		For GANs this is images being partitioned as well as noise
		"""
		assert type(n_partitions) == int
		data_xs = data_dictionary[self.x]
		data_noise = data_dictionary[self.noise]
		if n_partitions > len(data_xs):
			n_partitions = len(data_xs)
		chunk_size = int(data_xs.shape[0]/n_partitions)
		dictionary_partitions = []
		for chunk_i in range(n_partitions):
			# Array slicing should be a view, not a copy
			# So this should not be a memory issue
			my_chunk_x = data_xs[chunk_i*chunk_size:(chunk_i+1)*chunk_size]
			my_chunk_noise = data_noise[chunk_i*chunk_size:(chunk_i+1)*chunk_size]
			dictionary_partitions.append({self.x:my_chunk_x, self.noise: my_chunk_noise})
		return dictionary_partitions



