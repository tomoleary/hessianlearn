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
		x_shape = x.shape
		if len(x_shape) == 1:
			feed_dict[self.problem.w_hat] = x
			return self.sess.run(self.problem.H_action,feed_dict)
		elif len(x_shape) == 2:
			H_action = np.zeros_like(x)
			if verbose:
				try:
					from tqdm import tqdm
					for i in tqdm(range(x_shape[1])):
						feed_dict[self.problem.w_hat] = x[:,i]
						H_action[:,i] = self.sess.run(self.problem.H_action,feed_dict)
				except:
					print('No progress bar :(')
					for i in range(x_shape[1]):
						feed_dict[self.problem.w_hat] = x[:,i]
						H_action[:,i] = self.sess.run(self.problem.H_action,feed_dict)
			else:
				for i in range(x_shape[1]):
					feed_dict[self.problem.w_hat] = x[:,i]
					H_action[:,i] = self.sess.run(self.problem.H_action,feed_dict)
			return H_action
		else:
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
		x_shape = x.shape
		if len(x_shape) == 1:
			feed_dict[self.problem.w_hat] = x
			return self.sess.run(self.problem.H_quadratic,feed_dict)
		elif len(x_shape) == 2:
			number_of_quadratics = x_shape[1]
			H_quads = np.zeros(number_of_quadratics)
			if verbose:
				try:
					from tqdm import tqdm
					for i in tqdm(range(number_of_quadratics)):
						feed_dict[self.problem.w_hat] = x[:,i]
						H_quads[i] = self.sess.run(self.problem.H_quadratic,feed_dict)
				except:
					print('No progress bar :(')
					for i in range(number_of_quadratics):
						feed_dict[self.problem.w_hat] = x[:,i]
						H_quads[i] = self.sess.run(self.problem.H_quadratic,feed_dict)
			else:
				for i in range(number_of_quadratics):
					feed_dict[self.problem.w_hat] = x[:,i]
					H_quads[i] = self.sess.run(self.problem.H_quadratic,feed_dict)
			return H_quads
		else:
			raise

