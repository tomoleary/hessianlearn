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
from  ..utilities import ParameterList
from abc import ABC, abstractmethod


def ParametersRegularization(dictionary = {}):
	parameters = dictionary
	parameters["gamma"] = [1e-1, "regularization parameter"]

	return ParameterList(parameters)

class Regularization (ABC):
	"""
	This class describes the components of regularization used during training.

	The child class implements the specifics during construction
	"""

	@property
	def cost(self):
		return self._cost

	@property
	def gradient(self):
		return self._gradient

	@property
	def H_action(self):
		return self._H_action

class L2Regularization(Regularization):
	"""
	This class implements standard Tikhonov (L2) regularization
	with regularization parameter gamma
		(gamma/2)||w||^2
	"""
	def __init__(self,problem, gamma = None,parameters = ParametersRegularization(),dtype = tf.float32):
		"""
		The constructor for this class takes
			-problem: The description of the training problem i.e. hessianlearn.problem.Problem variant
			-gamma: The regularization parameter, can be found via Morozov discrepancy, trial and error etc.
		"""
		self.problem = problem
		self.parameters = parameters

		if gamma is not None:
			self.parameters['gamma'] = gamma

		self._cost = 0.5*self.parameters['gamma']*tf.reduce_sum(self.problem._flat_w*self.problem._flat_w)
		
		self._gradient = self.parameters['gamma']*self.problem._flat_w

		self._H_action = self.parameters['gamma']*self.problem.w_hat













