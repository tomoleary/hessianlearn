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
import math
import sys
import tensorflow as tf
if int(tf.__version__[0]) > 1:
	import tensorflow.compat.v1 as tf
	tf.disable_v2_behavior()


class NeuralNetwork(object):
	def __init__(self,architecture = None,seed = 0,dtype = tf.float32):
		self.architecture = architecture
		self.input_shape = architecture['input_shape']
		try:
			self.output_shape = architecture['output_shape']
		except:
			pass
		self.seed = 0
		self.dtype = dtype
		self.x = None
		self.y_prediction = None
		pass

	def __call__(self,x):
		raise NotImplementedError("Child class must implement __call__")


	# @property
	# def input_shape(self):
	# 	return self._input_shape

	# @property
	# def output_shape(self):
	# 	return self._output_shape


class ProjectedDenseEncoderDecoder(NeuralNetwork):
	def __init__(self, architecture, V, U, seed = 0,dtype = tf.float32):
		super(ProjectedDenseEncoderDecoder,self).__init__(architecture,seed,dtype)
		# input output Jacobian is J = USV', V is heuristically input subspace, U output subspace
		# input projector is V

		self._V = tf.cast(V,self.dtype)
		self._U = tf.cast(U,self.dtype)

		# Setup shapes for neural network construction
		self._input_shape = [-1] + self.input_shape[1:]
		self._output_shape = [-1] + self.output_shape[1:]

		assert self._input_shape[-1] == self._V.shape[0], 'Input dimension and input projector do not agree'
		assert self._output_shape[-1] == self._U.shape[0], 'Output dimension and output projector do not agree'

		self.n_inputs = np.prod(self.input_shape[1:])
		self.n_outputs = np.prod(self.output_shape[1:])
	
		first_index = [V.shape[1]]+ list(architecture['layer_dimensions'])
		second_index = list(architecture['layer_dimensions'])+[U.shape[-1]]

		self.shapes = list(zip(first_index,second_index))


		self.reshaped_input = [-1,self.n_inputs]
		self.reshaped_output = [-1,self.n_outputs]

		self.x = tf.placeholder(self.dtype,self.input_shape,name = 'image_placeholder')
		with tf.name_scope('reshape'):
			x = tf.reshape(self.x,self.reshaped_input)


		############################################################################################################

		# Initialize weights and activation functions
		# Variables are instantiated in the order they appear in the NN
		# Check to see if some layers should not be trained
		if 'trainable_bools' in architecture.keys():
			assert len(architecture['trainable_bools']) == 4 + 2*(len(architecture['layer_dimensions'])+1)
			trainable_bools = architecture['trainable_bools']
		else:
			trainable_bools = (4 + 2*(len(architecture['layer_dimensions'])+1))*[True]

		self._V = tf.Variable(self._V, name = 'input_projector',trainable = trainable_bools[0])
		input_bias = tf.Variable(tf.zeros(self._V.shape[-1],dtype = self.dtype),\
									name = 'input_bias',trainable = trainable_bools[1])

		# Inner dense NN
		inner_weights = []
		inner_biases = []

		for k, shape in enumerate(self.shapes):
			init = tf.random_normal(shape,stddev=0.35,seed = self.seed,dtype = self.dtype)
			inner_weights.append(tf.Variable(init,name='inner_weights%d'%k,trainable = trainable_bools[2+2*k]))
			inner_biases.append(tf.Variable(tf.zeros(shape[1],dtype = self.dtype),\
							 name='inner_bias%d'%k,trainable = trainable_bools[3+2*k]))

		self._U = tf.Variable(self._U, name = 'output_projector',trainable = trainable_bools[4+2*k])
		output_bias = tf.Variable(tf.zeros(self._output_shape[-1],dtype = self.dtype),\
									name = 'output_bias',trainable = trainable_bools[5+2*k])

		try:
			activation_functions = architecture['activation_functions']
			assert len(activation_functions) is len(inner_weights)+2
			activation_functions[-1] = tf.identity
		except:
			# Need one activation function for the first projection, one for each 
			# subsequent "inner layer" and then identity for the output
			self.activation_functions = [tf.nn.softmax for w in inner_weights] + [tf.nn.softplus] + [tf.identity]


		# Build the neural network
		h = self.activation_functions[0](tf.tensordot(x,self._V, axes = [[1],[0]] ))
		
		h += input_bias

		for i, (w,b,activation) in enumerate(zip(inner_weights,inner_biases,self.activation_functions[1:-1])):
			hw = tf.tensordot(h,w,axes = [[1],[0]])
			h = activation(hw)+b

		

		h = self.activation_functions[-1](tf.tensordot(self._U,h,axes = [[1],[1]]))
		h = tf.reshape(h,self._output_shape)
		
		h += output_bias
		self.y_prediction = h


class ProjectedLowRankResidualEncoderDecoder(NeuralNetwork):
	def __init__(self, architecture, V, U, seed = 0,dtype = tf.float32):
		super(ProjectedLowRankResidualEncoderDecoder,self).__init__(architecture,seed,dtype)
		# input output Jacobian is J = USV', V is heuristically input subspace, U output subspace
		# input projector is V
		assert 'layer_dimensions' and 'layer_ranks' in architecture.keys()
		self._V = tf.cast(V,self.dtype)
		self._U = tf.cast(U,self.dtype)


		# Setup shapes for neural network construction
		self._input_shape = [-1] + self.input_shape[1:]
		self._output_shape = [-1] + self.output_shape[1:]

		assert self._input_shape[-1] == self._V.shape[0], 'Input dimension and input projector do not agree'
		assert self._output_shape[-1] == self._U.shape[0], 'Output dimension and output projector do not agree'

		self.n_inputs = np.prod(self.input_shape[1:])
		self.n_outputs = np.prod(self.output_shape[1:])
		self.reshaped_input = [-1,self.n_inputs]
		self.reshaped_output = [-1,self.n_outputs]

		assert len(architecture['layer_ranks']) == len(architecture['layer_dimensions']) + 1
		assert V.shape[1] == U.shape[-1]
		if len(architecture['layer_dimensions']) > 0:
			assert architecture['layer_dimensions'][0] == V.shape[1]
			assert (np.array(architecture['layer_dimensions']) == architecture['layer_dimensions'][0]).all()
	
		right_indices = [V.shape[1]]+ list(architecture['layer_dimensions'])
		left_indices = list(architecture['layer_dimensions'])+[U.shape[-1]]

		self.left_shapes = list(zip(left_indices,architecture['layer_ranks']))
		self.right_shapes = list(zip(right_indices,architecture['layer_ranks']))
		
		self.x = tf.placeholder(self.dtype,self.input_shape,name = 'image_placeholder')
		with tf.name_scope('reshape'):
			x = tf.reshape(self.x,self.reshaped_input)
		############################################################################################################

		# Initialize weights and activation functions
		# Variables are instantiated in the order they appear in the NN
		# Check to see if some layers should not be trained
		if 'trainable_bools' in architecture.keys():
			assert len(architecture['trainable_bools']) == 4 + 3*(len(architecture['layer_dimensions'])+1)
			trainable_bools = architecture['trainable_bools']
		else:
			trainable_bools = (4 + 3*(len(architecture['layer_dimensions'])+1))*[True]

		self._V = tf.Variable(self._V, name = 'input_projector',trainable = trainable_bools[0])
		input_bias = tf.Variable(tf.zeros(self._V.shape[-1],dtype = self.dtype),\
									name = 'input_bias',trainable = trainable_bools[1])
		# Inner dense NN
		inner_left_weights = []
		inner_right_weights = []
		inner_biases = []

		for k, (left_shape, right_shape) in enumerate(zip(self.left_shapes,self.right_shapes)):
			left_init = tf.random_normal(left_shape,stddev=0.35,seed = self.seed,dtype = self.dtype)
			inner_left_weights.append(tf.Variable(left_init,name='inner_left_weights%d'%k,trainable = trainable_bools[2+3*k]))
			right_init = tf.random_normal(right_shape,stddev=0.35,seed = self.seed,dtype = self.dtype)
			inner_right_weights.append(tf.Variable(right_init,name='inner_right_weights%d'%k,trainable = trainable_bools[3+3*k]))
			inner_biases.append(tf.Variable(tf.zeros(left_shape[0],dtype = self.dtype),\
							 name='inner_bias%d'%k,trainable = trainable_bools[4+3*k]))

		self._U = tf.Variable(self._U, name = 'output_projector',trainable = trainable_bools[5+3*k])
		output_bias = tf.Variable(tf.zeros(self._output_shape[-1],dtype = self.dtype),\
									name = 'output_bias',trainable = trainable_bools[6+3*k])

		try:
			activation_functions = architecture['activation_functions']
			assert len(activation_functions) is len(inner_weights)+2
			activation_functions[-1] = tf.identity
		except:
			# Need one activation function for the first projection, one for each 
			# subsequent "inner layer" and then identity for the output
			self.activation_functions = [tf.nn.softmax for w in inner_biases]+ [tf.nn.softplus] + [tf.identity]


		# Build the neural network
		h = self.activation_functions[0](tf.tensordot(x,self._V, axes = [[1],[0]] ))
		h += input_bias

		for i, (w_left,w_right,b,activation) in enumerate(zip(inner_left_weights,inner_right_weights,\
															inner_biases,self.activation_functions[1:-1])):
			h_wright = tf.tensordot(h,w_right,axes = [[1],[0]])
			h += b + tf.tensordot(activation(h_wright),w_left,axes = [[1],[1]])
		
		h = self.activation_functions[-1](tf.tensordot(self._U,h,axes = [[1],[1]]))
		h = tf.reshape(h,self._output_shape)
		
		h += output_bias
		self.y_prediction = h
		
class ProjectedResidualEncoderDecoder(NeuralNetwork):
	def __init__(self, architecture, V, U, seed = 0,dtype = tf.float32):
		super(ProjectedResidualEncoderDecoder,self).__init__(architecture,seed,dtype)
		# input output Jacobian is J = USV', V is heuristically input subspace, U output subspace
		# input projector is V

		self._V = tf.cast(V,self.dtype)
		self._U = tf.cast(U,self.dtype)



		# Setup shapes for neural network construction
		self._input_shape = [-1] + self.input_shape[1:]
		self._output_shape = [-1] + self.output_shape[1:]

		assert self._input_shape[-1] == self._V.shape[0], 'Input dimension and input projector do not agree'
		assert self._output_shape[-1] == self._U.shape[0], 'Output dimension and output projector do not agree'

		self.n_inputs = np.prod(self.input_shape[1:])
		self.n_outputs = np.prod(self.output_shape[1:])


		assert V.shape[1] == U.shape[-1]
		if len(architecture['layer_dimensions']) > 0:
			assert (np.array(architecture['layer_dimensions']) == architecture['layer_dimensions'][0]).all()
	
		first_index = [V.shape[1]]+ list(architecture['layer_dimensions'])
		second_index = list(architecture['layer_dimensions'])+[U.shape[-1]]

		self.shapes = list(zip(first_index,second_index))


		self.reshaped_input = [-1,self.n_inputs]
		self.reshaped_output = [-1,self.n_outputs]

		self.x = tf.placeholder(self.dtype,self.input_shape,name = 'image_placeholder')
		with tf.name_scope('reshape'):
			x = tf.reshape(self.x,self.reshaped_input)


		############################################################################################################

		# Initialize weights and activation functions
		# Variables are instantiated in the order they appear in the NN
		# Check to see if some layers should not be trained
		if 'trainable_bools' in architecture.keys():
			assert len(architecture['trainable_bools']) == 6 + 2*len(architecture['layer_dimensions'])
			trainable_bools = architecture['trainable_bools']
		else:
			trainable_bools = (6 + 2*len(architecture['layer_dimensions']))*[True]

		self._V = tf.Variable(self._V, name = 'input_projector',trainable = trainable_bools[0])
		input_bias = tf.Variable(tf.zeros(self._V.shape[-1],dtype = self.dtype),\
									name = 'input_bias',trainable = trainable_bools[1])

		# Inner dense NN
		init_ws = []
		inner_weights = []
		inner_biases = []

		for k, shape in enumerate(self.shapes):
			init = tf.random_normal(shape,stddev=0.35,seed = self.seed,dtype = self.dtype)
			init_ws.append(init)
			inner_weights.append(tf.Variable(init,name='inner_weights%d'%k,trainable = trainable_bools[2+k],dtype = self.dtype))
			inner_biases.append(tf.Variable(tf.zeros(shape[1],dtype = self.dtype),\
							 name='inner_bias%d'%k,trainable = trainable_bools[3+k]))

		try:
			activation_functions = architecture['activation_functions']
			assert len(activation_functions) is len(inner_weights)+2
			activation_functions[-1] = tf.identity
		except:
			# Need one activation function for the first projection, one for each 
			# subsequent "inner layer" and then identity for the output
			self.activation_functions = [tf.nn.softmax for w in inner_weights] + [tf.nn.softplus] + [tf.identity]


		# Build the neural network
		h = self.activation_functions[0](tf.tensordot(x,self._V, axes = [[1],[0]] ))
		
		h += input_bias

		for i, (w,b,activation) in enumerate(zip(inner_weights,inner_biases,self.activation_functions[1:-1])):
			hw = tf.tensordot(h,w,axes = [[1],[0]])
			h += activation(hw)+b

		self._U = tf.Variable(self._U, name = 'output_projector',trainable = trainable_bools[4+k])
		output_bias = tf.Variable(tf.zeros(self._output_shape[-1],dtype = self.dtype),\
									name = 'output_bias',trainable = trainable_bools[5+k])

		h = self.activation_functions[-1](tf.tensordot(self._U,h,axes = [[1],[1]]))
		h = tf.reshape(h,self._output_shape)
		h += output_bias
		self.y_prediction = h







	