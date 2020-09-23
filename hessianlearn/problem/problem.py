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

		self._dtype = dtype

		self._NN = NeuralNetwork
		self.x = self.NN.inputs[0]
		
		self.y_prediction = self.NN(self.x)

		# Why does the following not work:
		# self.y_true = self.NN.outputs[0]
		# Instead I have to use a placeholder for true output

		output_shape = self.NN.output_shape
		self.y_true = tf.placeholder(self.dtype, output_shape,name='output_placeholder')

		# Assign trainable variables to member variable w "weights"
		self._w = tf.trainable_variables()

		self._flat_w = my_flatten(self._w)

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

		dims = [np.prod(shape) for shape in self.shapes]
		self._dimension = np.sum(dims)

		# Define loss function and accuracy in initialize_loss
		self._initialize_loss()
		# Once loss is defined gradients can be instantiated

		self._gradient = my_flatten(tf.gradients(self.loss,self._w, name = 'gradient'))
		self._norm_g = tf.sqrt(tf.reduce_sum(self.gradient*self.gradient))
		# Initialize vector for Hessian mat-vecs
		self._w_hat = tf.placeholder(self.dtype,self.dimension )
		# Define (g,dw) inner product
		self._g_inner_w_hat = tf.tensordot(self._w_hat,self._gradient,axes = [[0],[0]])
		# Define Hessian action Hdw
		self._H_action = my_flatten(tf.gradients(self._g_inner_w_hat,self._w,stop_gradients = self._w_hat,name = 'hessian_action'))
		self._H_quadratic = tf.tensordot(self._w_hat,self._H_action,axes = [[0],[0]])
		# Define operations for updating and assigment used during training
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
		# Boolean to indicate if only input data should be passed into loss function
		self._is_autoencoder = False


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
	def is_autoencoder(self):
		return self._is_autoencoder		

	def _initialize_loss(self):
		"""
		This method defines the loss as a function of the neural network and 
		placeholder variable for the true data.
		Child class of Problem must implement this method
		"""
		raise NotImplementedError("Child class must implement method initialize_loss") 

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
		raise NotImplementedError("Child class should implement method _parition_dictionaries") 



class ClassificationProblem(Problem):
	def __init__(self,NeuralNetwork,loss_type = 'cross_entropy',dtype = tf.float32):
		super(ClassificationProblem,self).__init__(NeuralNetwork,dtype = dtype)

	def _initialize_loss(self):
		with tf.name_scope('loss'):
			# scce = tf.keras.losses.SparseCategoricalCrossentropy()
			# self._loss = scce(self.y_true,self.y_prediction)
			self._loss = tf.reduce_mean(-tf.reduce_sum(self.y_true*tf.nn.log_softmax(self.y_prediction), [1]))

		with tf.name_scope('accuracy'):
			y_prediction_sm = tf.nn.softmax(self.y_prediction)
			correct_prediction = tf.equal(tf.argmax(self.y_prediction, 1), tf.argmax(self.y_true,1))
			correct_prediction = tf.cast(correct_prediction, self.dtype)
			self.accuracy = tf.reduce_mean(correct_prediction)

	def _partition_dictionaries(self,data_dictionary,n_partitions):
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




class LeastSquaresClassificationProblem(Problem):
	def __init__(self,NeuralNetwork,dtype = tf.float32):
		super(LeastSquaresClassificationProblem,self).__init__(NeuralNetwork,dtype = dtype)
		

	def _initialize_loss(self):
		with tf.name_scope('loss'):
			self._loss = tf.losses.mean_squared_error(labels=self.y_true, predictions=self.y_prediction)
		with tf.name_scope('rel_error'):
			self.rel_error = tf.sqrt(tf.reduce_mean(tf.pow(self.y_true-self.y_prediction,2))\
							/tf.reduce_mean(tf.pow(self.y_true,2)))
		with tf.name_scope('accuracy'):
			y_prediction_sm = tf.nn.softmax(self.y_prediction)
			correct_prediction = tf.equal(tf.argmax(self.y_prediction, 1), tf.argmax(self.y_true,1))
			correct_prediction = tf.cast(correct_prediction, self.dtype)
			self.accuracy = tf.reduce_mean(correct_prediction)

	def _partition_dictionaries(self,data_dictionary,n_partitions):
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
	def __init__(self,NeuralNetwork,y_mean = None,dtype = tf.float32):
		if y_mean is not None:
			self.y_mean = tf.constant(y_mean,dtype = dtype)
		else:
			self.y_mean = None
		super(RegressionProblem,self).__init__(NeuralNetwork,dtype = dtype)
		

	def _initialize_loss(self):
		with tf.name_scope('loss'):
			self._loss = tf.losses.mean_squared_error(labels=self.y_true, predictions=self.y_prediction)
		with tf.name_scope('rel_error'):
			self.rel_error = tf.sqrt(tf.reduce_mean(tf.pow(self.y_true-self.y_prediction,2))\
							/tf.reduce_mean(tf.pow(self.y_true,2)))
		with tf.name_scope('improvement'):
			assert self.y_mean is not None
			self.improvement = tf.sqrt(tf.reduce_mean(tf.pow(self.y_true-self.y_prediction,2))\
							/tf.reduce_mean(tf.pow(self.y_true - self.y_mean,2)))
		with tf.name_scope('mad'):
			try:
				import tensorflow_probability as tfp
				absolute_deviation = tf.math.abs(self.y_true - self.y_prediction)
				self.mad = tfp.stats.percentile(absolute_deviation,50.0,interpolation = 'midpoint')
			except:
				self.mad = None

	def _partition_dictionaries(self,data_dictionary,n_partitions):
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
	def __init__(self,NeuralNetwork,inputs = None,dtype = tf.float32):
		super(AutoencoderProblem,self).__init__(NeuralNetwork,dtype = dtype)
		self._is_autoencoder = True


	def _initialize_loss(self):
		with tf.name_scope('loss'): # 
			self._loss = tf.reduce_mean(tf.pow(self.x-self.y_prediction,2)) 
			self.rel_error = tf.sqrt(tf.reduce_mean(tf.pow(self.x-self.y_prediction,2))\
							/tf.reduce_mean(tf.pow(self.x,2)))
			self.accuracy = 1. - self.rel_error

	def _partition_dictionaries(self,data_dictionary,n_partitions):
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
	def __init__(self,NeuralNetwork,z_mean,z_log_sigma,inputs,dtype = tf.float32):
		self.z_mean = z_mean
		self.z_log_sigma = z_log_sigma
		super(VariationalAutoencoderProblem,self).__init__(NeuralNetwork,dtype = dtype)
		self._is_autoencoder = True

	def _initialize_loss(self,cross_entropy = False):
		with tf.name_scope('loss'): # 
			if cross_entropy:
				pass
			else:
				# VAE tutorials rescale the mse by the input dimension
				least_squares_loss = np.prod(self.NN.input_shape[1:])*tf.reduce_mean(tf.pow(self.x-self.y_prediction,2)) 
				kl_loss = tf.reduce_mean(-0.5*tf.reduce_sum(1 + self.z_log_sigma - tf.pow(self.z_mean,2) - tf.exp(self.z_log_sigma),axis = -1))
				# kl_loss = 0.0

				self._loss = least_squares_loss + kl_loss

				self.rel_error = tf.sqrt(tf.reduce_mean(tf.pow(self.x-self.y_prediction,2))\
								/tf.reduce_mean(tf.pow(self.x,2)))
				self.accuracy = 1. - self.rel_error

	def _partition_dictionaries(self,data_dictionary,n_partitions):
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




