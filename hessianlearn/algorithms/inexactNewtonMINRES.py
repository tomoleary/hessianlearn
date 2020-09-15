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
from ..algorithms import Optimizer, MINRESSolver, ParametersMINRESSolver
from ..algorithms.globalization import ArmijoLineSearch, TrustRegion
from ..problem import L2Regularization




def ParametersInexactNewtonMINRES(parameters = {}):
	parameters['alpha']                         = [1e-1, "Initial steplength, or learning rate"]
	parameters['rel_tolerance']                 = [1e-3, "Relative convergence when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
	parameters['abs_tolerance']                 = [1e-4,"Absolute converge when sqrt(g,g) <= abs_tolerance"]
	parameters['max_NN_evals_per_batch']        = [20, "Scale constant for maximum neural network evaluations per datum"]
	parameters['max_NN_evals']                  = [None, "Maximum number of neural network evaluations"]

	parameters['minres_parameters']					= [ ParametersMINRESSolver(),'CG Parameters']
	# CG solver parameters
	parameters['cg_coarse_tol']					= [0.5,'CG coarse solve tolerance']
	parameters['cg_max_iter']					= [1000,'CG maximum iterations']
	parameters['eta_mode']						= [0, 'eta mode for E-W conditions:0,1,2']
	parameters['globalization']					= [None, 'Choose from trust_region, line_search or none']
	parameters['max_backtracking_iter']			= [10, 'max backtracking iterations for line search']


	# Reasons for convergence failure
	parameters['reasons'] = [[], 'list of reasons for termination']


	return ParameterList(parameters)


class InexactNewtonMINRES(Optimizer):
	def __init__(self,problem,regularization = None,sess = None,feed_dict = None,parameters = ParametersInexactNewtonMINRES(),preconditioner = None):
		if regularization is None:
			_regularization = ZeroRegularization(problem)
		else:
			_regularization = regularization
		super(InexactNewtonMINRES,self).__init__(problem,_regularization,sess,parameters)

		self._sweeps = np.zeros(2)
		self.grad = self.problem.gradient + self.regularization.gradient
		self.minres_solver = MINRESSolver(self.problem,self.regularization,\
			self.sess,parameters= self.parameters['minres_parameters'])
		self.alpha = (8*'-').center(10)
		

	def minimize(self,feed_dict = None,hessian_feed_dict = None):
		r"""
		w-=alpha*g
		"""
		assert self.sess is not None
		assert feed_dict is not None
		if hessian_feed_dict is None:
			hessian_feed_dict = feed_dict
		
		self.gradient = self.sess.run(self.grad,feed_dict = feed_dict)

		if self.parameters['globalization'] == 'line_search':
			w_dir,_ = self.minres_solver.solve(-self.gradient,hessian_feed_dict)
			w_dir_inner_g = np.inner(w_dir,self.gradient)
			initial_cost = self.sess.run(self.problem.loss,feed_dict = feed_dict)
			cost_at_candidate = lambda p : self._loss_at_candidate(p,feed_dict = feed_dict)
			self.alpha, line_search, line_search_iter = ArmijoLineSearch(w_dir,w_dir_inner_g,\
																cost_at_candidate, initial_cost,\
											max_backtracking_iter = self.parameters['max_backtracking_iter'])
			update = self.alpha*w_dir
			self._sweeps += [1+0.5*line_search_iter,2*self.minres_solver.iter]
			self.sess.run(self.problem._update_ops,feed_dict = {self.problem._update_placeholder:update})
		elif self.parameters['globalization'] == None:
			self.alpha = self.parameters['alpha']
			p,converged = self.minres_solver.solve(-self.gradient,hessian_feed_dict)
			# print(converged)
			# if converged:
			# 	print('Converged!')
			# else:
			# 	print('NOT CONVERGED!!!!!')
			self._sweeps += [1, 4*self.minres_solver.iter]
			self.p = p
			update = self.alpha*p
			self.sess.run(self.problem._update_ops,feed_dict = {self.problem._update_placeholder:update})

				


		
		

		