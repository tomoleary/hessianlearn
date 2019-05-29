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
from ..utilities.utilityFunctions import *
from ..algorithms import Optimizer, CGSolver, ParametersCGSolver
from ..algorithms.globalization import ArmijoLineSearch, TrustRegion
from ..modeling import L2Regularization




def ParametersInexactNewtonCG(parameters = {}):
	parameters['alpha']                         = [1e0, "Initial steplength, or learning rate"]
	parameters['rel_tolerance']                 = [1e-3, "Relative convergence when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
	parameters['abs_tolerance']                 = [1e-4,"Absolute converge when sqrt(g,g) <= abs_tolerance"]
	parameters['max_NN_evals_per_batch']        = [10000, "Scale constant for maximum neural network evaluations per datum"]
	parameters['max_NN_evals']                  = [None, "Maximum number of neural network evaluations"]

	parameters['cg_parameters']					= [ ParametersCGSolver(),'CG Parameters']
	# CG solver parameters
	parameters['cg_coarse_tol']					= [0.5,'CG coarse solve tolerance']
	parameters['cg_max_iter']					= [10,'CG maximum iterations']
	parameters['eta_mode']						= [0, 'eta mode for E-W conditions:0,1,2']
	parameters['globalization']					= ['None', 'Choose from trust_region, line_search or none']
	parameters['max_backtracking_iter']			= [10, 'max backtracking iterations for line search']

	# Reasons for convergence failure
	parameters['reasons'] = [[], 'list of reasons for termination']


	return ParameterList(parameters)






class InexactNewtonCG(Optimizer):
	def __init__(self,problem,regularization = None,sess = None,feed_dict = None,\
			parameters = ParametersInexactNewtonCG(),preconditioner = None):
		if regularization is None:
			_regularization = ZeroRegularization(problem)
		else:
			_regularization = regularization
		super(InexactNewtonCG,self).__init__(problem,_regularization,sess,parameters)

	
		self.grad = self.problem.gradient + self.regularization.gradient
		self.cg_solver = CGSolver(self.problem,self.regularization,self.sess,parameters= self.parameters['cg_parameters'])
		self._sweeps = np.zeros(2)
		self.trust_region_initialized = False
		if self.parameters['globalization'] == 'trust_region':
			self.initialize_trust_region()
		self.alpha = (8*'-').center(10)



	def initialize_trust_region(self):
		if not self.parameters['globalization'] == 'trust_region':
			self.parameters['globalization'] = 'trust_region'
		self.trust_region = TrustRegion()
		self.cg_solver.initialize_trust_region(coarse_tol = self.parameters['cg_coarse_tol'])
		self.cg_solver.set_trust_region_radius(self.trust_region.radius)
		self.trust_region_initialized = True

	def minimize(self,feed_dict = None,hessian_feed_dict = None):
		r"""
		w-=alpha*g
		"""
		assert self.sess is not None
		assert feed_dict is not None
		if hessian_feed_dict is None:
			hessian_feed_dict = feed_dict
		
		self.gradient = self.sess.run(self.grad,feed_dict = feed_dict)



		if self.parameters['globalization'] is 'None':
			alpha = self.parameters['alpha']
			p,on_boundary = self.cg_solver.solve(-self.gradient,hessian_feed_dict)
			self._sweeps += [1,2*self.cg_solver.iter]
			self.p = p
			update = alpha*p
			return self.problem._update_w(update)

		if self.parameters['globalization'] == 'line_search':
			w_dir,on_boundary = self.cg_solver.solve(-self.gradient,hessian_feed_dict)
			w_dir_inner_g = np.inner(w_dir,self.gradient)
			initial_cost = self.sess.run(self.problem.loss,feed_dict = feed_dict)
			cost_at_candidate = lambda p : self._loss_at_candidate(p,feed_dict = feed_dict)
			self.alpha, line_search, line_search_iter = ArmijoLineSearch(w_dir,w_dir_inner_g,\
														cost_at_candidate, initial_cost,\
														max_backtracking_iter = self.parameters['max_backtracking_iter'])
			update = self.alpha*w_dir
			self._sweeps += [1+0.5*line_search_iter,2*self.cg_solver.iter]
			return self.problem._update_w(update)

		elif self.parameters['globalization'] == 'trust_region':
			if not self.trust_region_initialized:
				self.initialize_trust_region()
			# Set trust region radius
			self.cg_solver.set_trust_region_radius(self.trust_region.radius)
			p,on_boundary = self.cg_solver.solve(-self.gradient,feed_dict)
			self._sweeps += [1,2*self.cg_solver.iter]
			self.p = p
			# Solve for candidate step
			p, on_boundary  = self.cg_solver.solve(-self.gradient,hessian_feed_dict)
			pg = np.dot(p,self.gradient)
			# Calculate predicted reduction
			feed_dict[self.cg_solver.problem.w_hat] = p
			Hp 					= self.sess.run(self.cg_solver.Aop,feed_dict)
			pHp = np.dot(p,Hp)
			predicted_reduction = -pg-0.5*pHp
			# Calculate actual reduction
			misfit,reg = self.sess.run((self.problem.loss,self.regularization.cost),\
								feed_dict = feed_dict)
			cost = misfit + reg
			w_copy = self.sess.run(self.problem.w)
			self.sess.run(self.problem._update_w(p))

			misfit,reg = self.sess.run((self.problem.loss,self.regularization.cost),\
								feed_dict = feed_dict)
			cost_new = misfit + reg
			actual_reduction    = cost - cost_new

			# Decide whether or not to accept the step
			accept_step = self.trust_region.evaluate_step(actual_reduction = actual_reduction,\
				predicted_reduction = predicted_reduction,on_boundary = on_boundary)
			if accept_step:
				return self.problem._update_w(zeros_like(w_copy))
			else:
				return self.problem._assign_to_w(w_copy)
				


		
		

		