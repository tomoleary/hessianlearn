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
	flattened_list = []
	for tensor in tensor_list:
		flattened_list.append(tf.reshape(tensor,[np.prod(tensor.shape)]))
	return tf.concat(flattened_list,axis=0)

def initialize_indices(shapes):
	indices = []
	index = 0
	for shape in shapes:
		index += int(np.prod(shape))
		indices.append(index)
	return indices

def tensor_to_vec(x,shapes):
	# Inputs x a tensor
	storage = None
	for x,shape  in zip(x,shapes):

		x_flat = np.reshape(x,int(np.prod(shape)))
		if storage is None:
			storage = x_flat
		else:
			storage = np.concatenate((storage,x_flat))
	return storage

def vec_to_tensor(x,shapes,indices):
	x = np.squeeze(x)
	storage = []
	index_1 = 0
	for index_2,shape in zip(indices,shapes):
		chunk = x[index_1:index_2]
		index_1 = index_2
		reshaped = np.reshape(chunk,shape)
		storage.append(reshaped)
	return storage

def placeholder_like(shapes,name,dtype):
	placeholder = []
	for shape in shapes:
		placeholder.append(tf.placeholder(dtype,shape,name=name))
	return placeholder



class Problem(ABC):
	def __init__(self,NeuralNetwork = None,dtype = tf.float32):
		self._dtype = dtype
		# placeholder for input
		# input_shape = NeuralNetwork.input_shape

		# self.x = tf.placeholder(self.dtype,input_shape,name = 'image_placeholder')
		try:
			self.x = NeuralNetwork.x
		except:
			input_shape = NeuralNetwork.input_shape
			self.x = tf.placeholder(self.dtype,input_shape,name = 'image_placeholder')
		# self.x_shape = NeuralNetwork.input_shape
		# placeholder for true output
		try:
			output_shape = NeuralNetwork.output_shape
			# if output_shape[0] == -1:
			# 	output_shape[0] = None
			self.y_true = tf.placeholder(self.dtype, output_shape,name='label_placeholder')
		except:
			# self.y_true = tf.placeholder(self.dtype, [None,NeuralNetwork.n_outputs],name='label_placeholder')
			pass

		try:
			self.y_prediction = NeuralNetwork.y_prediction
		except:
			self.y_prediction = NeuralNetwork(self.x)
		# self.y_shape = y_prediction.shape
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
		# Initialize indexing for vector representation if needed
		# self._initialize_indices()
		# Initialize vector for Hessian mat-vecs
		self._w_hat = tf.placeholder(self.dtype,self.dimension )
		# w_hat = self.placeholder_like_g('w_hat')
		# self.w_hat = tuple(w_hat)
		# Define (g,dw) inner product
		self._g_inner_w_hat = tf.tensordot(self._w_hat,self._gradient,axes = [[0],[0]])
		# self.g_inner_w_hat = [tf.reduce_sum(g * w_hat) for g, w_hat in zip(self.gradient, self.w_hat)]
		# Define Hessian action Hdw
		self._H_w_hat = my_flatten(tf.gradients(self._g_inner_w_hat,self._w,stop_gradients = self._w_hat,name = 'hessian_action'))
		
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
	def H_w_hat(self):
		return self._H_w_hat

	def _initialize_loss(self):
		raise NotImplementedError("Child class should implement method initialize_loss") 


	def _initialize_indices(self):
		assert self.gradient is not None
		shapes = []
		indices = []
		index = 0
		for g in self.gradient:
			shape = list(g.shape)
			shapes.append(shape)
			# print(type(np.prod(shape)),type(index))
			index += int(np.prod(shape))
			indices.append(index)
		self.w_dim = index
		self.shapes = shapes
		self.indices = indices


	def tensor_to_vec(self,x):
		# Inputs x a tensor
		storage = None
		for x,shape  in zip(x,self.shapes):

			x_flat = np.reshape(x,int(np.prod(shape)))
			if storage is None:
				storage = x_flat
			else:
				storage = np.concatenate((storage,x_flat))
		return storage

	def vec_to_tensor(self,x):
		storage = []
		index_1 = 0
		for index_2,shape in zip(self.indices,self.shapes):
			chunk = x[index_1:index_2]
			index_1 = index_2
			reshaped = np.reshape(chunk,shape)
			storage.append(reshaped)
		return storage

	def zeros_like_g(self):
		zeros = []
		for shape in self.shapes:
			zeros.append(np.zeros(shape))
		return zeros

	def placeholder_like_g(self,name ):

		placeholder = []
		for shape in self.shapes:
			placeholder.append(tf.placeholder(self.dtype,shape,name=name))
		return placeholder

	def _zero_layers(self,array_like_w,list_of_layer_names):
		assert array_like_w.shape == self._flat_w.shape
		# Make sure that each layer name is in the layer_descriptors dictionary
		for layer_name in list_of_layer_names:
			assert layer_name in self.layer_descriptors.keys()

		for layer_name in list_of_layer_names:
			indices = self.layer_descriptors[layer_name]['indices']
			array_like_w[indices[0]:indices[1]] = np.zeros(indices[1] - indices[0])

		return array_like_w

	def _set_layer(self,array_like_w,array_like_layer,layer_name):
		assert array_like_w.shape == self._flat_w.shape
		assert layer_name in self.layer_descriptors.keys()
		indices = self.layer_descriptors[layer_name]['indices']
		assert len(array_like_layer) == indices[1] - indices[0]
		array_like_w[indices[0]:indices[1]] = array_like_layer
		return array_like_w




class ClassificationProblem(Problem):
	def __init__(self,NeuralNetwork,dtype = tf.float32):
		super(ClassificationProblem,self).__init__(NeuralNetwork,dtype)

	def _initialize_loss(self):
		with tf.name_scope('loss'):
			# scce = tf.keras.losses.SparseCategoricalCrossentropy()
			# self.loss = scce(self.y_true,self.y_prediction)
			self.loss = tf.reduce_mean(-tf.reduce_sum(self.y_true*tf.nn.log_softmax(self.y_prediction), [1]))

		with tf.name_scope('accuracy'):
			y_prediction_sm = tf.nn.softmax(self.y_prediction)
			correct_prediction = tf.equal(tf.argmax(self.y_prediction, 1), tf.argmax(self.y_true,1))
			correct_prediction = tf.cast(correct_prediction, self.dtype)
			self.accuracy = tf.reduce_mean(correct_prediction)


class LeastSquaresClassificationProblem(Problem):
	def __init__(self,NeuralNetwork,y_mean = None,dtype = tf.float32):
		super(LeastSquaresClassificationProblem,self).__init__(NeuralNetwork,dtype)
		

	def _initialize_loss(self):
		with tf.name_scope('loss'):
			self.loss = tf.losses.mean_squared_error(labels=self.y_true, predictions=self.y_prediction)
		with tf.name_scope('rel_error'):
			self.rel_error = tf.sqrt(tf.reduce_mean(tf.pow(self.y_true-self.y_prediction,2))\
							/tf.reduce_mean(tf.pow(self.y_true,2)))
		with tf.name_scope('accuracy'):
			y_prediction_sm = tf.nn.softmax(self.y_prediction)
			correct_prediction = tf.equal(tf.argmax(self.y_prediction, 1), tf.argmax(self.y_true,1))
			correct_prediction = tf.cast(correct_prediction, self.dtype)
			self.accuracy = tf.reduce_mean(correct_prediction)


class RegressionProblem(Problem):
	def __init__(self,NeuralNetwork,y_mean = None,dtype = tf.float32):
		if y_mean is not None:
			self.y_mean = tf.constant(y_mean,dtype = dtype)
		else:
			self.y_mean = None
		super(RegressionProblem,self).__init__(NeuralNetwork,dtype)
		

	def _initialize_loss(self):
		with tf.name_scope('loss'):
			self.loss = tf.losses.mean_squared_error(labels=self.y_true, predictions=self.y_prediction)
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


class AutoencoderProblem(Problem):
	def __init__(self,NeuralNetwork,dtype = tf.float32):
		super(AutoencoderProblem,self).__init__(NeuralNetwork,dtype)


	def _initialize_loss(self):
		with tf.name_scope('loss'): # 
			self.loss = tf.reduce_mean(tf.pow(self.x-self.y_prediction,2)) 
			self.rel_error = tf.sqrt(tf.reduce_mean(tf.pow(self.x-self.y_prediction,2))\
							/tf.reduce_mean(tf.pow(self.x,2)))
			self.accuracy = 1. - self.rel_error








