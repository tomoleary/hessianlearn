from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from  ..utilities import ParameterList
from abc import ABC, abstractmethod


def ParametersRegularization(dictionary = {}):
	parameters = dictionary
	parameters["beta"] = [1e1, "regularization parameter"]

	return ParameterList(parameters)

class Regularization (ABC):
	def __init__(self):

		# make all the things that get initialized
		# Define loss function and accuracy in __init__

		pass

	@property
	def cost(self):
		return self._cost

	@property
	def gradient(self):
		return self._gradient

	@property
	def H_w_hat(self):
		return self._H_w_hat



class ZeroRegularization(Regularization):
	def __init__(self,problem,beta = None,parameters = ParametersRegularization(),dtype = tf.float32):
		# Must implement hessian apply and gradient
		self._problem = problem
		self.parameters = parameters

		self._cost = 0.0
		# fix this
		self._gradient = 0.0*self.problem._flat_w
		# fix this
		self._H_w_hat = 0.0*self.problem.w_hat




class L2Regularization(Regularization):
	def __init__(self,problem, beta = None,parameters = ParametersRegularization(),dtype = tf.float32):
		# Must implement hessian apply and gradient
		self.problem = problem
		self.parameters = parameters

		if beta is not None:
			self.parameters['beta'] = beta
		# fix this
		self._cost = 0.5*self.parameters['beta']*tf.reduce_sum(self.problem._flat_w*self.problem._flat_w)
		# fix this
		self._gradient = self.parameters['beta']*self.problem._flat_w
		# fix this
		self._H_w_hat = self.parameters['beta']*self.problem.w_hat






