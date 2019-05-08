from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from ..utilities.parameterList import ParameterList
from ..utilities.utilityFunctions import *
from ..algorithms import Optimizer




def ParametersAdam(parameters = {}):
	parameters['alpha']                         = [1e-3, "Initial steplength, or learning rate"]
	parameters['beta_1']                        = [0.9, "Exponential decay rate for first moment"]
	parameters['beta_2']                        = [0.999, "Exponential decay rate for second moment"]
	parameters['epsilon']						= [1e-8, "epsilon for denominator involving square root"]

	parameters['rel_tolerance']                 = [1e-3, "Relative convergence when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
	parameters['abs_tolerance']                 = [1e-4,"Absolute converge when sqrt(g,g) <= abs_tolerance"]
	parameters['max_NN_evals_per_batch']        = [10000, "Scale constant for maximum neural network evaluations per datum"]
	parameters['max_NN_evals']                  = [None, "Maximum number of neural network evaluations"]

	parameters['globalization']					= ['None', 'Choose from trust_region, line_search or none']
	# Reasons for convergence failure
	parameters['reasons'] = [[], 'list of reasons for termination']


	return ParameterList(parameters)


class Adam(Optimizer):
	def __init__(self,problem,regularization = None,sess = None,feed_dict= None,parameters = ParametersAdam()):
		if regularization is None:
			_regularization = ZeroRegularization(problem)
		else:
			_regularization = regularization
		super(Adam,self).__init__(problem,_regularization,sess,parameters)

		self.grad = self.problem.gradient + self.regularization.gradient

		self.m = np.zeros_like(self.grad)
		self.v = np.zeros_like(self.grad)
		self.p = np.zeros_like(self.grad)

		self._iter = 0
		self._sweeps = np.zeros(2)

		self.alpha = parameters['alpha']

	def minimize(self,feed_dict = None):
		r"""
		w-=alpha*g
		"""
		assert self.sess is not None
		assert feed_dict is not None
		self._iter += 1

		alpha = self.parameters['alpha']* np.sqrt(1 - self.parameters['beta_2']**self.iter)/(1 - self.parameters['beta_1']**self.iter)

		gradient = self.sess.run(self.grad,feed_dict = feed_dict)
		
		self.m = self.parameters['beta_1']*self.m + (1-self.parameters['beta_1'])*gradient 
		# m_hat = [m/(1 - self.parameters['beta_1']**self.iter) for m in self.m]

		g_sq_vec = np.square(gradient) 
		self.v = self.parameters['beta_2']*self.v + (1-self.parameters['beta_2'])*g_sq_vec 
		v_root = np.sqrt(self.v)


		update = -alpha*self.m/(v_root +self.parameters['epsilon'])
		self.p = update
		self._sweeps += [1,0]
		return self.problem._update_w(update)
		

