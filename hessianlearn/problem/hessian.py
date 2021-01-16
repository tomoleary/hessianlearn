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
# import tensorflow as tf
# if int(tf.__version__[0]) > 1:
# 	import tensorflow.compat.v1 as tf
# 	tf.disable_v2_behavior()
from abc import ABC, abstractmethod


class Hessian(ABC):
	"""
	This class implements methods for the neural network training Hessian.

	Must have a problem and a sess in order to be evaluated
	"""
	def __init__(self,problem=None,sess=None):
		"""
		Create a Hessian given:

			- problem: the description of the neural network training problem 
				(hessianlearn.problem.Problem)
			- sess: the tf.Session() needed for evaluation at run time
		"""
		self._problem = problem
		self._sess = sess

	@property
	def problem(self):
		return self._problem
	@property
	def sess(self):
		return self._sess

	@property
	def dimension(self):
		return self.problem.dimension
	

	@property
	def T(self):
		return self._T

	def _T(self):
		return self
	
	def __mult__(self,x):
		return self(x)

	def __call__(self,x,feed_dict,verbose = False):
		"""
		This method implements Hessian action, must have a problem and sess
		set before this method can be evaluated.
			-x: numpy array to be multiplied one at a time
			-feed_dict: data used in finite sum Hessian evaluation
			-verbose: for printing
		"""
		assert self.problem is not None
		assert self.sess is not None

		if len(x.shape) == 1:
			feed_dict[self.problem.dw] = x
			return self.sess.run(self.problem.Hdw,feed_dict)
		elif len(x.shape) == 2:
			n_vectors = x.shape[-1]
			if self.problem._HdW is None:
				if verbose:
					print('Total vectors = ',n_vectors)
					print('Initializing Hessian blocking')
				self.problem._initialize_hessian_blocking(n_vectors)
			# When the block sizes agree
			if n_vectors == self.problem._hessian_block_size:
				feed_dict[self.problem._dW] = x
				HdW = self.sess.run(self.problem.HdW,feed_dict)
				return HdW
			# When the requested block size is smaller
			elif n_vectors < self.problem._hessian_block_size:
				# The speedup is roughly 5x, so in the case that its less 
				# than 1/5 its faster to either reinitialize the blocking
				# or for loop around running problem.Hdw
				if n_vectors < 0.2*self.problem._hessian_block_size:
					# Could reinitialize the blocking or just for loop
					# For looping for now
					HdW = np.zeros_like(x)
					for i in range(n_vectors):
						feed_dict[problem.dw] = x[:,i]
						HdW[:,i] = sess.run(problem.Hdw,feed_dict)
					return HdW
				else:
					dW = np.zeros(self.problem.dimension,self.problem._hessian_block_size)
					dW[:,:n_vectors] = x
					feed_dict[self.problem._dW] = dW
					HdW =  self.sess.run(self.problem.HdW,feed_dict)
					return HdW[:,:n_vectors]
			# When the requested block size is larger
			elif n_vectors > self.problem._hessian_block_size:
				HdW = np.zeros_like(x)
				block_size = self.problem._hessian_block_size
				blocks, remainder = np.divmod(block_size,block_size)
				for i in range(blocks):
					feed_dict[self.problem._dW] = x[:,i*block_size:(i+1)*block_size]
					HdW[:,i*block_size:(i+1)*block_size] = self.sess.run(self.problem.HdW,feed_dict)
				# The last vectors are done as a for loop or a zeroed out array
				if remainder < 0.2*self.problem._hessian_block_size:
					for i in range(n_vectors):
						feed_dict[problem.dw] = x[:,blocks*block_size+i]
						HdW[:,blocks*block_size+i] = sess.run(problem.Hdw,feed_dict)
				else:
					dW = np.zeros(self.problem.dimension,self.problem._hessian_block_size)
					dW[:,:remainder] = x[:,-remainder:]
					feed_dict[self.problem._dW] = dW
					HdW[:,-remainder:] = sess.run(problem.Hdw,feed_dict)
		else:
			# Many different Hessian mat-vecs interpreted as a tensor?
			print('This case is not yet implemented'.center(80))
			raise

	def quadratics(self,x,feed_dict,verbose = False):
		"""
		This method implements Hessian quadratics xTHx. 
		Must have self._problem and self._sess set before this method can be evaluated.
			-x: numpy array to be multiplied one at a time
			-feed_dict: data used in finite sum Hessian evaluation
			-verbose: for printing
		"""
		assert self.problem is not None
		assert self.sess is not None
		if len(x.shape) == 1:
			feed_dict[self.problem.dw] = x
			return self.sess.run(self.problem.H_quadratic,feed_dict)
		elif len(x.shape) == 2:
			number_of_quadratics = x.shape[1]
			H_quads = np.zeros(number_of_quadratics)
			if verbose:
				try:
					from tqdm import tqdm
					for i in tqdm(range(number_of_quadratics)):
						feed_dict[self.problem.dw] = x[:,i]
						H_quads[i] = self.sess.run(self.problem.H_quadratic,feed_dict)
				except:
					print('No progress bar :(')
					for i in range(number_of_quadratics):
						feed_dict[self.problem.dw] = x[:,i]
						H_quads[i] = self.sess.run(self.problem.H_quadratic,feed_dict)
			else:
				for i in range(number_of_quadratics):
					feed_dict[self.problem.dw] = x[:,i]
					H_quads[i] = self.sess.run(self.problem.H_quadratic,feed_dict)
			return H_quads
		else:
			raise


class HessianWrapper:
	
	def __init__(self,hessian,data_dictionary):
		
		self._hessian = hessian
		self._data_dictionary = data_dictionary
		
		
	def __call__(self,x):
		return self._hessian(x,self._data_dictionary)
