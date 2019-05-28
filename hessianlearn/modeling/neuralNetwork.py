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
		pass

	def __call__(self,x):
		raise NotImplementedError("Child class must implement __call__")


	# @property
	# def input_shape(self):
	# 	return self._input_shape

	# @property
	# def output_shape(self):
	# 	return self._output_shape

		

class GenericDNN(NeuralNetwork):
	def __init__(self,architecture = None,seed = 0,dtype = tf.float32):
		super(GenericDNN,self).__init__(architecture,seed,dtype)
		self._input_shape = [-1] + self.input_shape[1:]
		self._output_shape = [-1] + self.output_shape[1:]

		self.n_inputs = np.prod(self.input_shape[1:])

		self.n_outputs = np.prod(self.output_shape[1:])

		first_index = [self.n_inputs]+ list(architecture['layer_dimensions'])
		second_index = list(architecture['layer_dimensions'])+[self.n_outputs]
		self.shapes = list(zip(first_index,second_index))

		self.reshaped_input = [-1,self.n_inputs]
		self.reshaped_output = [-1,self.n_outputs]

	def __call__(self,x):
		with tf.name_scope('reshape'):
			x = tf.reshape(x,self.reshaped_input)
		############################################################################################################
		# Encoding layer
		init_ws = []
		e_ws = []
		e_bs = []
		for k, shape in enumerate(self.shapes):
			print('shape',shape)
			init = tf.random_normal(shape,stddev=0.35,seed = self.seed)
			init_ws.append(init)
			e_ws.append(tf.Variable(init,name='encoder_weights%d'%k))
			e_bs.append(tf.Variable(tf.zeros(shape[1]), name='encoder_bias%d'%k))

			
		ws = e_ws 
		bs = e_bs 

		try:
			activation_functions = architecture['activation_functions']
			assert len(activation_functions) is len(ws)
		except:
			self.activation_functions = [tf.nn.softmax for w in ws[:-1]] + [tf.identity]
			# self.activation_functions = [tf.nn.softmax for w in ws]

		h = x
		for w,b,activation in zip(ws,bs,self.activation_functions):
			h = activation(tf.matmul(h,w)+b)
		h = tf.reshape(h,self._output_shape)
		return h


class GenericCDNN(NeuralNetwork):
	def __init__(self,architecture,seed = 0, dtype = tf.float32):
		super(GenericCDNN,self).__init__(architecture,seed,dtype)
		# self.input_shape = input_shape
		self.n_filters = architecture['n_filters']
		self.filter_sizes = architecture['filter_sizes']

		self.output_shape = architecture['output_shape']

		self.n_outputs = np.prod(self.output_shape[1:])
		self._output_shape = [-1] + self.output_shape[1:]


		self.reshaped_output = [-1,self.n_outputs]


	def __call__(self,x):
		# x = tf.placeholder(
		#     tf.float32, input_shape, name='x')
		x_shape = tf.shape(x)
		batch_size = x_shape[0]
		# if len(x_shape) ==3:
		# 	current_input = tf.reshape(x,x_shape+[1])
		# else:
		current_input = x
		batch_size = tf.shape(x)[0]
		n_layers = len(self.n_filters)
		# %%
		# Build the encoder
		encoder = []
		shapes = []
		for layer_i, n_output in enumerate(self.n_filters[1:]):
			n_input = current_input.get_shape().as_list()[3]
			shapes.append(current_input.get_shape().as_list())
			W = tf.Variable(
				tf.random_uniform([
					self.filter_sizes[layer_i],
					self.filter_sizes[layer_i],
					n_input, n_output],
					-1.0 / math.sqrt(n_input),
					1.0 / math.sqrt(n_input),seed = self.seed))
			b = tf.Variable(tf.zeros([n_output]))
			encoder.append(W)
			output = tf.nn.softmax(
				tf.add(tf.nn.conv2d(
					current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
			current_input = output

		latent_shape = current_input.shape
		# print(latent_shape)
		# print(type(latent_shape))
		latent_dimension = int(np.prod(latent_shape[1:]))
		print(latent_dimension)
		print(type(latent_dimension))
		reshaped_latent = [-1, latent_dimension]
		if self.architecture['layer_dimensions'] ==[]:
			self.DNN_shapes = [(latent_dimension,self.n_outputs)]
		else:
			first_index = [latent_dimension]+ list(self.architecture['layer_dimensions'])
			second_index = list(self.architecture['layer_dimensions'])+[self.n_outputs]
			self.DNN_shapes = list(zip(first_index,second_index))

		with tf.name_scope('reshape_latent'):
			current_input = tf.reshape(current_input,reshaped_latent)
		############################################################################################################
		# Encoding layer
		init_ws = []
		e_ws = []
		e_bs = []
		for k, shape in enumerate(self.DNN_shapes):
			print('shape',shape)
			init = tf.random_normal(shape,stddev=0.35,seed = self.seed)
			init_ws.append(init)
			e_ws.append(tf.Variable(init,name='encoder_weights%d'%k))
			e_bs.append(tf.Variable(tf.zeros(shape[1]), name='encoder_bias%d'%k))

			
		ws = e_ws 
		bs = e_bs 

		try:
			activation_functions = self.architecture['activation_functions']
			assert len(activation_functions) is len(ws)
		except:
			self.activation_functions = [tf.nn.softmax for w in ws[:-1]] + [tf.identity]

		h = current_input
		for w,b,activation in zip(ws,bs,self.activation_functions):
			h = activation(tf.matmul(h,w)+b)
		h = tf.reshape(h,self._output_shape)

		# %%
		# now have the reconstruction through the network
		return h


class GenericCED(NeuralNetwork):
	def __init__(self,architecture,seed = 0, dtype = tf.float32):
		super(GenericCED,self).__init__(architecture,seed,dtype)
		# self.input_shape = input_shape
		self.n_filters = architecture['n_filters']
		self.filter_sizes = architecture['filter_sizes']

	def __call__(self,x):
		# x = tf.placeholder(
		#     tf.float32, input_shape, name='x')
		x_shape = tf.shape(x)
		batch_size = x_shape[0]
		# if len(x_shape) ==3:
		# 	current_input = tf.reshape(x,x_shape+[1])
		# else:
		current_input = x
		batch_size = tf.shape(x)[0]
		n_layers = len(self.n_filters)
		# %%
		# Build the encoder
		encoder = []
		shapes = []
		for layer_i, n_output in enumerate(self.n_filters[1:]):
			n_input = current_input.get_shape().as_list()[3]
			shapes.append(current_input.get_shape().as_list())
			W = tf.Variable(
				tf.random_uniform([
					self.filter_sizes[layer_i],
					self.filter_sizes[layer_i],
					n_input, n_output],
					-1.0 / math.sqrt(n_input),
					1.0 / math.sqrt(n_input),seed = self.seed))
			b = tf.Variable(tf.zeros([n_output]))
			encoder.append(W)
			output = tf.nn.softmax(
				tf.add(tf.nn.conv2d(
					current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
			current_input = output
		# %%
		# store the latent representation
		z = current_input
		encoder.reverse()
		shapes.reverse()
		# %%
		# Build the decoder using the same weights
		for layer_i, shape in enumerate(shapes):
			n_input = current_input.get_shape().as_list()[3]
			W = tf.Variable(
				tf.random_uniform(tf.shape(encoder[layer_i]),
					-1.0 / math.sqrt(n_input),
					1.0 / math.sqrt(n_input),seed = self.seed))
			# W = encoder[layer_i]
			b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
			if layer_i == n_layers -2:
				print(layer_i,'Last layer identity')
				output = tf.identity(tf.add(
					tf.nn.conv2d_transpose(
						current_input, W,
						tf.stack([batch_size, shape[1], shape[2], shape[3]]),
						strides=[1, 2, 2, 1], padding='SAME'), b))

			else:
				print(layer_i,'Inner layer softmax')
				output = tf.nn.softmax(tf.add(
					tf.nn.conv2d_transpose(
						current_input, W,
						tf.stack([batch_size, shape[1], shape[2], shape[3]]),
						strides=[1, 2, 2, 1], padding='SAME'), b))
			current_input = output
		# %%
		# now have the reconstruction through the network
		y = current_input
		return y





class GenericCAE(NeuralNetwork):
	def __init__(self,architecture, seed = 0, dtype = tf.float32):
		super(GenericCAE,self).__init__(architecture,seed,dtype)
		# self.input_shape = input_shape
		self.n_filters = architecture['n_filters']
		self.filter_sizes = architecture['filter_sizes']

	def __call__(self,x ):
		# x = tf.placeholder(
		#     tf.float32, input_shape, name='x')
		current_input = x
		batch_size = tf.shape(x)[0]
		n_layers = len(self.n_filters)
		# %%
		# Build the encoder
		encoder = []
		shapes = []
		for layer_i, n_output in enumerate(self.n_filters[1:]):
			n_input = current_input.get_shape().as_list()[3]
			shapes.append(current_input.get_shape().as_list())
			W = tf.Variable(
				tf.random_uniform([
					self.filter_sizes[layer_i],
					self.filter_sizes[layer_i],
					n_input, n_output],
					-1.0 / math.sqrt(n_input),
					1.0 / math.sqrt(n_input),seed = self.seed))
			b = tf.Variable(tf.zeros([n_output]))
			encoder.append(W)
			output = tf.nn.softmax(
				tf.add(tf.nn.conv2d(
					current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
			current_input = output
		# %%
		# store the latent representation
		z = current_input
		encoder.reverse()
		shapes.reverse()
		# %%
		# Build the decoder using the same weights
		for layer_i, shape in enumerate(shapes):
			n_input = current_input.get_shape().as_list()[3]
			W = tf.Variable(
				tf.random_uniform(tf.shape(encoder[layer_i]),
					-1.0 / math.sqrt(n_input),
					1.0 / math.sqrt(n_input),seed = self.seed))
			# W = encoder[layer_i]
			b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
			if layer_i == n_layers -2:
				# print(layer_i,'Last layer identity')
				output = tf.identity(tf.add(
					tf.nn.conv2d_transpose(
						current_input, W,
						tf.stack([batch_size, shape[1], shape[2], shape[3]]),
						strides=[1, 2, 2, 1], padding='SAME'), b))

			else:
				# print(layer_i,'Inner layer softmax')
				output = tf.nn.softmax(tf.add(
					tf.nn.conv2d_transpose(
						current_input, W,
						tf.stack([batch_size, shape[1], shape[2], shape[3]]),
						strides=[1, 2, 2, 1], padding='SAME'), b))
			current_input = output
		# %%
		# now have the reconstruction through the network
		y = current_input
		return y


class GenericDAE(NeuralNetwork):
	def __init__(self,architecture = None,seed = 0,dtype = tf.float32):
		super(GenericDAE,self).__init__(architecture,seed,dtype)
		self.input_shape = architecture['input_shape']
		self._input_shape = [-1] + self.input_shape[1:]

		self.n_inputs = np.prod(self.input_shape[1:])

		first_index = [self.n_inputs]+ list(architecture['layer_dimensions'])[:-1]
		second_index = list(architecture['layer_dimensions'])
		self.shapes = list(zip(first_index,second_index))

		self.reshaped_input = [-1,self.n_inputs]
		pass

	def __call__(self,x):
		with tf.name_scope('reshape'):
			x = tf.reshape(x,self.reshaped_input)
		############################################################################################################
		# Encoding layer
		init_ws = []
		e_ws = []
		e_bs = []
		for k, shape in enumerate(self.shapes):
			print('shape',shape)
			init = tf.random_normal(shape,stddev=0.35,seed = self.seed)
			init_ws.append(init)
			e_ws.append(tf.Variable(init,name='encoder_weights%d'%k))
			e_bs.append(tf.Variable(tf.zeros(shape[1]), name='encoder_bias%d'%k))
		############################################################################################################
		# Decoding
		d_ws = []
		d_bs = []


		k = len(init_ws)
		for init,shape in reversed(list(zip(init_ws,self.shapes))):
			print(shape)
			d_ws.append(tf.Variable(tf.transpose(init),name='decoder_weights%d'%k))
			d_bs.append(tf.Variable(tf.zeros(shape[0]), name='decoder_bias%d'%k))
			k -= 1



		ws = e_ws + d_ws 
		bs = e_bs + d_bs

		try:
			activation_functions = architecture['activation_functions']
			assert len(activation_functions) is len(ws)
		except:
			self.activation_functions = [tf.nn.softmax for w in ws[:-1]] + [tf.identity]

		h = x
		for w,b,activation in zip(ws,bs,self.activation_functions):
			h = activation(tf.matmul(h,w)+b)
		h = tf.reshape(h,self._input_shape)
		return h


class ProjectedGenericDNN(NeuralNetwork):
    def __init__(self,architecture = None,V = None,C = None,seed = 0,dtype = tf.float32):
        super(ProjectedGenericDNN,self).__init__(architecture,seed,dtype)
        self._V = tf.cast(V,self.dtype)
        if C is not None:
            pass
            self._C = tf.cast(C,self.dtype)
        else:
            self._C = None
        




        self._input_shape = [-1] + self.input_shape[1:]
        self._output_shape = [-1] + self.output_shape[1:]

        self.n_inputs = np.prod(self.input_shape[1:])

        self.n_outputs = np.prod(self.output_shape[1:])
        
       
        
        if V is None:
            print('V is None')
            first_index = [self.n_inputs]+ list(architecture['layer_dimensions'])
        else:
            first_index = [V.shape[0]]+ list(architecture['layer_dimensions'])
        second_index = list(architecture['layer_dimensions'])+[self.n_outputs]
        self.shapes = list(zip(first_index,second_index))

        self.reshaped_input = [-1,self.n_inputs]
        print('reshaped input = ', self.reshaped_input)
        print('V shape = ', self._V.shape)
        self.reshaped_output = [-1,self.n_outputs]

    def __call__(self,x):
        with tf.name_scope('reshape'):
            x = tf.reshape(x,self.reshaped_input)
        ############################################################################################################
        # Encoding layer
        init_ws = []
        e_ws = []
        e_bs = []
        print('x shape before divide = ', x.shape)
        if not (self._C is None):
            x /= self._C
        print('x shape after divide = ', x.shape)
        x = tf.tensordot(self._V,x, axes = [[1],[1]] )
        print('x shape after projection  = ', x.shape)
        
        for k, shape in enumerate(self.shapes):
            print('shape',shape)
            init = tf.random_normal(shape,stddev=0.35,seed = self.seed)
            init_ws.append(init)
            e_ws.append(tf.Variable(init,name='encoder_weights%d'%k))
            e_bs.append(tf.Variable(tf.zeros(shape[1]), name='encoder_bias%d'%k))


        ws = e_ws 
        bs = e_bs 

        try:
            activation_functions = architecture['activation_functions']
            assert len(activation_functions) is len(ws)
        except:
            self.activation_functions = [tf.nn.softmax for w in ws[:-1]] + [tf.identity]
            # self.activation_functions = [tf.nn.softmax for w in ws]

        h = x
        for w,b,activation in zip(ws,bs,self.activation_functions):
            h = activation(tf.matmul(h,w)+b)
            print(type(h))
        h = tf.reshape(h,self._output_shape)
        return h




	
class MNISTDNN(NeuralNetwork):
	def __init__(self,architecture = None,dtype = tf.float32):
		super(MNISTDNN,self).__init__(architecture,dtype)
		self.n_inputs = 28*28
		self.n_hidden1 = 300
		self.n_hidden2 = 100
		self.n_outputs = 10
		self.reshaped_input = [-1,self.n_inputs]
		pass

	def __call__(self,x):
		with tf.name_scope('reshape'):
			x = tf.reshape(x,self.reshaped_input)
		""" Constructing simple neural network """
		with tf.name_scope('dnn'):
			with tf.name_scope('layer_1'):
				init_1 = tf.random_uniform((self.n_inputs, self.n_hidden1), -1.0, 1.0, dtype=self.dtype)
				w_1 = tf.Variable(init_1, name='weights_layer_1', dtype=self.dtype)
				b_1 = tf.Variable(tf.zeros([self.n_hidden1], dtype=self.dtype), name='bias_1', dtype=self.dtype)
				y_1 = tf.nn.softmax(tf.matmul(x, w_1) + b_1)

			with tf.name_scope('layer_2'):
				n_inputs_2 = int(y_1.get_shape()[1])
				init_2 = tf.random_uniform((n_inputs_2, self.n_hidden2), -1.0, 1.0, dtype=self.dtype)
				w_2 = tf.Variable(init_2, name='weights_layer_2', dtype=self.dtype)
				b_2 = tf.Variable(tf.zeros([self.n_hidden2], dtype=self.dtype), name='bias_2', dtype=self.dtype)
				y_2 = tf.nn.softmax(tf.matmul(y_1, w_2) + b_2)

			with tf.name_scope('out_layer'):
				n_inputs_out = int(y_2.get_shape()[1])
				init_out = tf.random_uniform((n_inputs_out, self.n_outputs), -1.0, 1.0, dtype=self.dtype)
				w_out = tf.Variable(init_out, name='weights_layer_out', dtype=self.dtype)
				b_out = tf.Variable(tf.zeros([self.n_outputs], dtype=self.dtype), name='bias_out', dtype=self.dtype)
				y_out = tf.matmul(y_2, w_out) + b_out
				y_out_sm = tf.nn.softmax(y_out)
				# y_out = tf.nn.sigmoid(y_out1)

		return y_out




	