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
from ..algorithms.globalization import ArmijoLineSearch, TrustRegion




def ParametersGradientDescent(parameters = {}):
	parameters['alpha']                         = [1e-3, "Initial steplength, or learning rate"]
	parameters['rel_tolerance']                 = [1e-3, "Relative convergence when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
	parameters['abs_tolerance']                 = [1e-4,"Absolute converge when sqrt(g,g) <= abs_tolerance"]
	parameters['max_NN_evals_per_batch']        = [10000, "Scale constant for maximum neural network evaluations per datum"]
	parameters['max_NN_evals']                  = [None, "Maximum number of neural network evaluations"]
	parameters['max_backtracking_iter']			= [10, 'max backtracking iterations for line search']

	parameters['globalization']					= [None, 'Choose from trust_region, line_search or none']
	# Reasons for convergence failure
	parameters['reasons'] = [[], 'list of reasons for termination']

	return ParameterList(parameters)


class GradientDescent(Optimizer):
	"""
	This class implements the gradient descent (and stochastic variant) optimizer
	"""
	def __init__(self,problem,regularization,sess = None,feed_dict = None,parameters = ParametersGradientDescent()):
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
		super(GradientDescent,self).__init__(problem,_regularization,sess,parameters)

		self.grad = self.problem.gradient + self.regularization.gradient
		self._sweeps = np.zeros(2)

		self.trust_region_initialized = False
		if self.parameters['globalization'] == 'trust_region':
			self.alpha = 0.0
		else:
			self.alpha = parameters['alpha']




	def minimize(self,feed_dict = None):
		r"""
		Implements the gradient update:
		w-=alpha*g
		Takes the parameter:
			-feed_dict: data to be used to evaluate stochastic gradient and cost
		"""
		assert self.sess is not None
		assert feed_dict is not None
		
		g = self.sess.run(self.grad,feed_dict = feed_dict)


		if self.parameters['globalization'] == 'line_search':
			w_dir = -g
			w_dir_inner_g = np.inner(w_dir,g)
			initial_cost = self.sess.run(self.problem.loss, feed_dict)
			cost_at_candidate = lambda p : self._loss_at_candidate(p,feed_dict)
			self.alpha, line_search, line_search_iter = ArmijoLineSearch(w_dir,w_dir_inner_g,\
																			cost_at_candidate, initial_cost)
			p = self.alpha*w_dir
			self._sweeps += [1+0.5*line_search_iter,0]

		elif self.parameters['globalization'] == None:
			self.alpha = self.parameters['alpha']
			p = -self.parameters['alpha']*g
			self._sweeps += [1,0]

		self.p = p

		self.sess.run(self.problem._update_ops,feed_dict = {self.problem._update_placeholder:p})
		