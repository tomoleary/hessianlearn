from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC, abstractmethod

from ..utilities.parameterList import ParameterList

def ParametersOptimizer(dictionary = {}):
	parameters = dictionary
	parameters['alpha']                         = [1.0, "Initial steplength, or learning rate"]
	parameters['rel_tolerance']                 = [1e-3, "Relative convergence when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
	parameters['abs_tolerance']                 = [1e-4,"Absolute converge when sqrt(g,g) <= abs_tolerance"]
	parameters['max_NN_evals_per_batch']        = [10000, "Scale constant for maximum neural network evaluations per datum"]
	parameters['max_NN_evals']                  = [None, "Maximum number of neural network evaluations"]

	parameters['globalization']					= ['None', 'Choose from trust_region, line_search or none']
	# Reasons for convergence failure
	parameters['reasons'] = [[], 'list of reasons for termination']


	return ParameterList(parameters)


class Optimizer(ABC):
	def __init__(self,problem = None,regularization = None, sess = None,parameters = ParametersOptimizer(),comm = None):
		self._problem = problem
		self._regularization = regularization
		self._sess = sess
		self._parameters = parameters
		self._sweeps = 0
		self._comm = comm
		self._iter = 0

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

	# this will need to become H_w_hat


	def _loss_at_candidate(self,p,feed_dict):
		self.sess.run(self.problem._update_w(p))
		misfit = self.sess.run((self.problem.loss),feed_dict)
		self.sess.run(self.problem._update_w(-p))
		return misfit


	def H_w_hat(self,x,feed_dict):
		assert self.problem is not None
		assert self.sess is not None

		feed_dict[self.problem.w_hat] = x
		H_w_hat = self.sess.run(self.problem.H_w_hat,feed_dict)
		return H_w_hat


