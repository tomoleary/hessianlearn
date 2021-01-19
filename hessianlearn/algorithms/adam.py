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

import numpy as np

from ..utilities.parameterList import ParameterList
from ..algorithms import Optimizer




def ParametersAdam(parameters = {}):
	parameters['alpha']                         = [1e-3, "Initial steplength, or learning rate"]
	parameters['beta_1']                        = [0.9, "Exponential decay rate for first moment"]
	parameters['beta_2']                        = [0.999, "Exponential decay rate for second moment"]
	parameters['epsilon']						= [1e-7, "epsilon for denominator involving square root"]

	parameters['rel_tolerance']                 = [1e-3, "Relative convergence when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
	parameters['abs_tolerance']                 = [1e-4,"Absolute converge when sqrt(g,g) <= abs_tolerance"]
	parameters['max_NN_evals_per_batch']        = [10000, "Scale constant for maximum neural network evaluations per datum"]
	parameters['max_NN_evals']                  = [None, "Maximum number of neural network evaluations"]

	parameters['globalization']					= [None, 'Choose from trust_region, line_search or none']
	# Reasons for convergence failure
	parameters['reasons'] = [[], 'list of reasons for termination']


	return ParameterList(parameters)


class Adam(Optimizer):
	"""
	This class implements the Adam optimizer
	"""
	def __init__(self,problem,regularization = None,sess = None,feed_dict= None,parameters = ParametersAdam()):
		"""
		The constructor for this class takes:
			-problem: hessianlearn.problem.Problem
			-regularization: hessianlearn.problem.Regularization
			-sess: tf.Session()
			-parameters: hyperparameters dictionary
		"""
		if regularization is None:
			_regularization = L2Regularization(problem,gamma = 0.0)
		else:
			_regularization = regularization
		super(Adam,self).__init__(problem,_regularization,sess,parameters)

		self.grad = self.problem.gradient + self.regularization.gradient

		self.m = np.zeros(self.problem.dimension)
		self.v = np.zeros(self.problem.dimension)
		self.p = np.zeros(self.problem.dimension)

		self._iter = 0
		self._sweeps = np.zeros(2)

		self.alpha = self.parameters['alpha']

	def minimize(self,feed_dict = None):
		r"""
		This method implements one step of the Adam algorithm:
			-feed_dict: data dictionary used to evaluate gradient
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
		self.sess.run(self.problem._update_ops,feed_dict = {self.problem._update_placeholder:update})
		

