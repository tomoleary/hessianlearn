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
import time

from ..utilities.parameterList import ParameterList
from ..algorithms import Optimizer
from ..algorithms.globalization import ArmijoLineSearch, TrustRegion
from ..algorithms.randomizedEigensolver import randomized_eigensolver, eigensolver_from_range
from ..algorithms.rangeFinders import block_range_finder, noise_aware_adaptive_range_finder
from ..algorithms.varianceBasedNystrom import variance_based_nystrom
from ..problem import L2Regularization, HessianWrapper




def ParametersLowRankSaddleFreeNewton(parameters = {}):
	parameters['alpha']                         = [1e0, "Initial steplength, or learning rate"]
	parameters['rel_tolerance']                 = [1e-3, "Relative convergence when sqrt(g,g)/sqrt(g_0,g_0) <= rel_tolerance"]
	parameters['abs_tolerance']                 = [1e-4,"Absolute converge when sqrt(g,g) <= abs_tolerance"]
	parameters['default_damping']        		= [1e-3, "Levenberg-Marquardt damping when no regularization is used"]
	
	# Hessian approximation parameters
	parameters['range_finding']					= [None,"Range finding, if None then r = hessian_low_rank\
															Choose from None, 'arf', 'naarf', 'vn'"]
	parameters['range_rel_error_tolerance']     = [0.1, "Error tolerance for error estimator in adaptive range finding"]
	parameters['range_abs_error_tolerance']     = [100, "Error tolerance for error estimator in adaptive range finding"]
	parameters['range_block_size']        		= [20, "Block size used in range finder"]
	parameters['rq_samples_for_naarf']        	= [100, "Number of partitions for RQ variance evaluation"]
	parameters['hessian_low_rank']        		= [20, "Fixed rank for randomized eigenvalue decomposition"]
	# Variance Nystrom Parameters
	parameters['max_bad_vectors_nystrom']       = [5, "Number of maximum bad vectors for variance based Nystrom"]
	parameters['max_vectors_nystrom']       	= [40, "Number of maximum vectors for variance based Nystrom"]
	parameters['nystrom_std_tolerance']       	= [0.5, "Noise to eigenvalue ratio used for Nystrom truncation"]
	

	# Globaliziation parameters
	parameters['globalization']					= [None, 'Choose from trust_region, line_search, spectral_step or none']
	parameters['max_backtracking_iter']			= [5, 'Max backtracking iterations for armijo line search']
	parameters['spectral_step_alpha']			= [1e-2, 'Used in min condition for spectral step']

	parameters['verbose']                       = [False, "Printing"]
	parameters['record_last_rq_std']			= [False, "Record the last eigenvector RQ variance"]

	return ParameterList(parameters)


class LowRankSaddleFreeNewton(Optimizer):
	"""
	This class implements the Low Rank Saddle Free Newton (LRSFN) algorithm
	"""
	def __init__(self,problem,regularization = None,sess = None,parameters = ParametersLowRankSaddleFreeNewton(),preconditioner = None):
		"""
		The constructor for this class takes:
			-problem: hessianlearn.problem.Problem
			-regularization: hessianlearn.problem.Regularization
			-sess: tf.Session()
			-parameters: hyperparameters dictionary
			-preconditioner: hessianlearn.problem.Preconditioner
		"""
		if regularization is None:
			_regularization = L2Regularization(problem,gamma = 0.0)
		else:
			_regularization = regularization
		super(LowRankSaddleFreeNewton,self).__init__(problem,_regularization,sess,parameters)

		self.grad = self.problem.gradient + self.regularization.gradient

		if self.parameters['globalization'] == 'trust_region':
			self.trust_region = TrustRegion()
		self._sweeps = np.zeros(2)

		self.alpha = 0.0
		self._rank = 0

		self._rq_std = 0.0

	@property
	def rank(self):
		return self._rank

	@property
	def rq_variance(self):
		return self._rq_variance
	
	


	def minimize(self,feed_dict = None,hessian_feed_dict = None,rq_estimator_dict = None):
		r"""
		Solves the saddle escape problem. Given a misfit (loss) Hessian operator (H)
		1. H = U_r Lambda_r U_r^T
		2. Solve [U_r |Lambda_r| U_r^T + gamma I] p = -g for p via Woodbury formula:

		[U_r Lambda_r U_r^T + gamma I]^{-1} = 1/gamma * I - 1/gamma * UDU^T
		where D = diag(|lambda_i|/(|lambda_i| + gamma))
			-feed_dict: data dictionary used for evaluating gradient and cost
			-hessian_feed_dict: dictionary used for stochastic Hessian
			-rq_estimator_dict: dictionary used for RQ variance calculations

		"""
		self._iter += 1
		assert self.sess is not None
		assert feed_dict is not None

		assert self.parameters['range_finding'] in [None,'arf','naarf','vn']

		if hessian_feed_dict is None:
			hessian_feed_dict = feed_dict
		
		
		gradient = self.sess.run(self.grad,feed_dict = feed_dict)

		alpha = self.parameters['alpha']
		
		if self.parameters['range_finding'] == 'arf':
			H = lambda x: self.H(x,hessian_feed_dict,verbose = self.parameters['verbose'])
			n = self.problem.dimension
			# norm_g = np.linalg.norm(gradient)
			# tolerance = self.parameters['range_rel_error_tolerance']*norm_g
			tolerance = self.parameters['range_rel_error_tolerance']
			Q = block_range_finder(H,n,tolerance,self.parameters['range_block_size'])
			self._rank = Q.shape[1]
			print('Shape Q = ',Q.shape)
			Lmbda,U = eigensolver_from_range(H,Q)

		elif self.parameters['range_finding'] == 'naarf':
			norm_g = np.linalg.norm(gradient)
			tolerance = self.parameters['range_rel_error_tolerance']*norm_g
			if rq_estimator_dict is None:
				rq_estimator_dict_list = self.problem._partition_dictionaries(feed_dict,self.parameters['rq_samples_for_naarf'])
			elif type(rq_estimator_dict) == list:
				rq_estimator_dict_list = rq_estimator_dict
			elif type(rq_estimator_dict) == dict:
				rq_estimator_dict_list = self.problem._partition_dictionaries(rq_estimator_dict,self.parameters['rq_samples_for_naarf'])
			else:
				raise
			Q = noise_aware_adaptive_range_finder(self.H,hessian_feed_dict,rq_estimator_dict_list,block_size = self.parameters['range_block_size'],epsilon = tolerance)
			self._rank = Q.shape[1]
			H = lambda x: self.H(x,hessian_feed_dict,verbose = self.parameters['verbose'])
			Lmbda,U = eigensolver_from_range(H,Q)

		elif self.parameters['range_finding'] == 'vn':
			if rq_estimator_dict is None:
				rq_estimator_dict_list = self.problem._partition_dictionaries(feed_dict,self.parameters['rq_samples_for_naarf'])
			elif type(rq_estimator_dict) == list:
				rq_estimator_dict_list = rq_estimator_dict
			elif type(rq_estimator_dict) == dict:
				rq_estimator_dict_list = self.problem._partition_dictionaries(rq_estimator_dict,self.parameters['rq_samples_for_naarf'])
			else:
				raise
			nystrom_t0 = time.time()
			apply_H_list = [HessianWrapper(self.H,dictionary) for dictionary in rq_estimator_dict_list]
			[Lmbda, U, all_std_good],[Lmbda_all,U_all,all_std] = variance_based_nystrom(apply_H_list, self.H.dimension,\
																std_tol = self.parameters['nystrom_std_tolerance'],\
																max_vectors = self.parameters['max_vectors_nystrom'],\
																max_bad_vectors=self.parameters['max_bad_vectors_nystrom'],\
																verbose = self.parameters['verbose'])
			self._rank = U_all.shape[1]
			if self.parameters['verbose']:
				print('Nystrom method took ',time.time() - nystrom_t0, 's')

		else:
			H = lambda x: self.H(x,hessian_feed_dict,verbose = self.parameters['verbose'])
			n = self.problem.dimension
			self._rank = self.parameters['hessian_low_rank']
			Lmbda,U = randomized_eigensolver(H, n, self._rank,verbose=False)
		
		self.eigenvalues = Lmbda
		# Log the variance of the last eigenvector
		if self.parameters['record_last_rq_std'] :
			try:
				rq_direction = U[:,-1]
				if rq_estimator_dict is None:
					rq_estimator_dict_list = self.problem._partition_dictionaries(feed_dict,self.parameters['rq_samples_for_naarf'])
				elif type(rq_estimator_dict) == list:
					rq_estimator_dict_list = rq_estimator_dict
				elif type(rq_estimator_dict) == dict:
					rq_estimator_dict_list = self.problem._partition_dictionaries(rq_estimator_dict,self.parameters['rq_samples_for_naarf'])
				else:
					raise	
				
				try:
					RQ_samples = np.zeros((len(rq_estimator_dict_list),rq_direction.shape[1]))
				except:
					RQ_samples = np.zeros(len(rq_estimator_dict_list))

				for samp_i,sample_dictionary in enumerate(rq_estimator_dict_list):
					RQ_samples[samp_i] = self.H.quadratics(rq_direction,sample_dictionary)
				self._rq_std = np.std(RQ_samples)
			except:
				self._rq_std = None
				print(80*'#')
				print('U is [], taking gradient step, fix this later?'.center(80))

		# Saddle free inversion via Woodbury
		if self.regularization.parameters['gamma'] < 1e-4:
			gamma_damping = self.parameters['default_damping']
			# Using this condition instead of fixed gamma allows one to take larger step sizes
			# but does not appear to improve accuracy
			# gamma_damping = max(0.9*Lmbda_abs[-1],self.parameters['default_damping'])
		else:
			gamma_damping = self.regularization.parameters['gamma']
		Lmbda_abs = np.abs(Lmbda)
		Lmbda_diags = diags(Lmbda_abs)
		# Build terms for Woodbury inversion
		D_denominator = Lmbda_abs + gamma_damping*np.ones_like(Lmbda_abs)
		D = np.divide(Lmbda_abs,D_denominator)
		# Invert by applying terms in Woodbury formula:
		UTg = np.dot(U.T,gradient)
		DUTg = np.multiply(D,UTg)
		UDUTg = np.dot(U,DUTg)
		minus_p = (gradient - UDUTg)/gamma_damping
		self.p = -minus_p
		

		# Globalization: compute alpha and update the weights
		if self.parameters['globalization'] is None:
			self.alpha = self.parameters['alpha']
			self._sweeps += [1,2*self._rank]
			update = self.alpha*self.p
			self.sess.run(self.problem._update_ops,feed_dict = {self.problem._update_placeholder:update})

		elif self.parameters['globalization'] is 'spectral_step':
			self.alpha = min(self.parameters['spectral_step_alpha'],0.1/Lmbda_abs[0])
			self._sweeps += [1,2*self._rank]
			update = self.alpha*self.p
			self.sess.run(self.problem._update_ops,feed_dict = {self.problem._update_placeholder:update})

		elif self.parameters['globalization'] == 'line_search':
			w_dir_inner_g = np.inner(self.p,gradient)
			initial_cost = self.sess.run(self.problem.loss,feed_dict = feed_dict)
			cost_at_candidate = lambda p : self._loss_at_candidate(p,feed_dict = feed_dict)
			self.alpha, line_search, line_search_iter = ArmijoLineSearch(self.p,w_dir_inner_g,\
														cost_at_candidate, initial_cost,
														max_backtracking_iter = self.parameters['max_backtracking_iter'])
			update = self.alpha*self.p
			self._sweeps += [1+0.5*line_search_iter,2*self._rank]
			self.sess.run(self.problem._update_ops,feed_dict = {self.problem._update_placeholder:update})



		
		
		