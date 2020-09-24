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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC, abstractmethod
import numpy as np

from ..utilities.parameterList import ParameterList
from ..problem import Hessian

def ParametersOptimizer(dictionary = {}):
	parameters = dictionary
	parameters['alpha']                         = [1.0, "Initial steplength, or learning rate"]
	parameters['rel_tolerance']                 = [1e-3, "Relative convergence when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
	parameters['abs_tolerance']                 = [1e-4,"Absolute converge when sqrt(g,g) <= abs_tolerance"]
	parameters['globalization']					= [None, 'Choose from trust_region, line_search or none']


	return ParameterList(parameters)


class Optimizer(ABC):
	"""
	This class describes the optimizer used during training

	All children must implement the method minimize, which implements 
	one step of the optimizers weight update scheme
	"""
	def __init__(self,problem = None,regularization = None, sess = None,parameters = ParametersOptimizer(),comm = None):
		"""
		The constructor for this class takes:
			-problem: hessianlearn.problem.Problem class
			-regularization: hessianlearn.problem.Regularization class
			-sess: the tf.Session() used to evaluate the computational graph
			-parameters: the dictionary of hyperparameters for the optimizer.
		"""
		self._problem = problem
		self._regularization = regularization
		self._sess = sess
		self._parameters = parameters
		self._sweeps = 0
		self._comm = comm
		self._iter = 0
		self.H = Hessian(problem=problem,sess=sess)

	@property
	def problem(self):
		return self._problem

	@property
	def sess(self):
		return self._sess

	@property
	def parameters(self):
		return self._parameters

	@property
	def sweeps(self):
		return self._sweeps

	@property
	def comm(self):
		return self._comm

	@property
	def iter(self):
		return self._iter

	@property
	def regularization(self):
		return self._regularization


	def minimize(self):
		r"""
		Implements update rule for the algorithm.
		"""
		raise NotImplementedError("Child class should implement method minimize") 
	
	def initialize_trust_region(self):
		r"""
		Initializes trust region parameters
		"""
		raise NotImplementedError("Child class should implement method minimize") 



	def _loss_at_candidate(self,p,feed_dict):
		"""
		This method implements a function to assist with Armijo line search
			-p: candidate update to be evaluated in Armijo line search producedure
			-feed_dict: data dictionary used to evaluate cost at candidate
		"""
		self.sess.run(self.problem._update_ops,feed_dict = {self.problem._update_placeholder:p})
		# self.sess.run(self.problem._update_w(p))
		misfit = self.sess.run((self.problem.loss),feed_dict)
		self.sess.run(self.problem._update_ops,feed_dict = {self.problem._update_placeholder:-p})
		# self.sess.run(self.problem._update_w(-p))
		return misfit
		


