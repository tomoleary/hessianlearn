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
from scipy.sparse import diags

from ..utilities.parameterList import ParameterList
from ..algorithms import Optimizer
from ..algorithms.globalization import ArmijoLineSearch, TrustRegion
from ..algorithms.randomizedEigensolver import randomized_eigensolver
from ..problem import L2Regularization





def ParametersLowRankSaddleFreeNewton(parameters = {}):
	parameters['alpha']                         = [1e0, "Initial steplength, or learning rate"]
	parameters['rel_tolerance']                 = [1e-3, "Relative convergence when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
	parameters['abs_tolerance']                 = [1e-4,"Absolute converge when sqrt(g,g) <= abs_tolerance"]
	parameters['max_NN_evals_per_batch']        = [10000, "Scale constant for maximum neural network evaluations per datum"]
	parameters['max_NN_evals']                  = [None, "Maximum number of neural network evaluations"]

	parameters['hessian_low_rank']        		= [20, "Scale constant for maximum neural network evaluations per datum"]
	parameters['default_damping']        		= [1e-3, "Levenberg-Marquardt damping when no regularization is used"]

	parameters['globalization']					= ['None', 'Choose from trust_region, line_search or none']
	parameters['max_backtracking_iter']			= [5, 'Max backtracking iterations for armijo line search']

	# Reasons for convergence failure
	parameters['reasons'] = [[], 'list of reasons for termination']


	return ParameterList(parameters)


class LowRankSaddleFreeNewton(Optimizer):
	def __init__(self,problem,regularization = None,sess = None,feed_dict = None,parameters = ParametersLowRankSaddleFreeNewton(),preconditioner = None):
		if regularization is None:
			_regularization = ZeroRegularization(problem)
		else:
			_regularization = regularization
		super(LowRankSaddleFreeNewton,self).__init__(problem,_regularization,sess,parameters)
		self.grad = self.problem.gradient + self.regularization.gradient
		if self.parameters['globalization'] == 'trust_region':
			self.trust_region = TrustRegion()
		self._sweeps = np.zeros(2)
		self.alpha = (8*'-').center(10)


	def minimize(self,feed_dict = None,hessian_feed_dict = None):
		r"""
		Solves the saddle escape problem. Given a misfit (loss) Hessian operator (H)
		Takes:
		H_misfit operator (callable) : \mathbb{R}^n \rightarrow \mathbb{R}^n
		alpha_damping -- if Tikhonov regularization is used then alpha is the regularization 
			coefficient, otherwise it is a Levenberg-Marquardt style damping coefficient
		n: the dimension of the weights
		rank: dimension for low rank spectral decomposition
		1. H = U_r Lambda_r U_r^T
		2. Solve [U_r |Lambda_r| U_r^T + alpha I] p = -g for p via Woodbury formula:

		[U_r Lambda_r U_r^T + alpha I]^{-1} = 1/alpha * I - 1/alpha * UDU^T
		where D = diag(|lambda_i|/(|lambda_i| + alpha))

		"""
		assert self.sess is not None
		assert feed_dict is not None

		if hessian_feed_dict is None:
			hessian_feed_dict = feed_dict
		
		
		gradient = self.sess.run(self.grad,feed_dict = feed_dict)

		alpha = self.parameters['alpha']
		rank = self.parameters['hessian_low_rank']
		H = lambda x: self.H_w_hat(x,hessian_feed_dict)
		n = self.problem.dimension

		Lmbda,U = randomized_eigensolver(H, n, rank,verbose=False)
		self.lambdas = Lmbda
		Lmbda_abs = np.abs(Lmbda)
		Lmbda_diags = diags(Lmbda_abs)

		if self.regularization.parameters['gamma'] < 1e-4:
			alpha_damping = self.parameters['default_damping']
		else:
			alpha_damping = self.regularization.parameters['gamma']
		# Build terms for Woodbury inversion
		D_denominator = Lmbda_abs + alpha_damping*np.ones_like(Lmbda_abs)

		D = np.divide(Lmbda_abs,D_denominator)




		# Invert by applying terms in Woodbury formula:
		UTg = np.dot(U.T,gradient)
		DUTg = np.multiply(D,UTg)
		UDUTg = np.dot(U,DUTg)
		minus_p = (gradient - UDUTg)/alpha_damping
		self.p = -minus_p
		


		if self.parameters['globalization'] is 'None':
			alpha = self.parameters['alpha']
			self._sweeps += [1,2*rank]
			update = alpha*self.p
			self.sess.run(self.problem._update_ops,feed_dict = {self.problem._update_placeholder:update})


		elif self.parameters['globalization'] == 'line_search':
			w_dir_inner_g = np.inner(self.p,gradient)
			initial_cost = self.sess.run(self.problem.loss,feed_dict = feed_dict)
			cost_at_candidate = lambda p : self._loss_at_candidate(p,feed_dict = feed_dict)
			self.alpha, line_search, line_search_iter = ArmijoLineSearch(self.p,w_dir_inner_g,\
														cost_at_candidate, initial_cost,
														max_backtracking_iter = self.parameters['max_backtracking_iter'])
			update = self.alpha*self.p
			self._sweeps += [1+0.5*line_search_iter,2*rank]
			self.sess.run(self.problem._update_ops,feed_dict = {self.problem._update_placeholder:update})



		
		
		