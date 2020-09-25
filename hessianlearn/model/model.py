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
from abc import ABC, abstractmethod

import sys, os, pickle, time, datetime
# sys.path.append( os.environ.get('HESSIANLEARN_PATH', "../../"))
# from hessianlearn import *

from ..utilities.parameterList import ParameterList

# from ..algorithms import *

from ..algorithms.adam import Adam
from ..algorithms.gradientDescent import GradientDescent
# from ..algorithms.cgSolver import CGSolver
from ..algorithms.inexactNewtonCG import InexactNewtonCG
# from ..algorithms.gmresSolver import GMRESSolver 
# from ..algorithms.inexactNewtonGMRES import InexactNewtonGMRES
# from ..algorithms.minresSolver import MINRESSolver
# from ..algorithms.inexactNewtonMINRES import InexactNewtonMINRES
from ..algorithms.randomizedEigensolver import *
from ..problem.regularization import L2Regularization
from ..algorithms.lowRankSaddleFreeNewton import LowRankSaddleFreeNewton



def HessianlearnModelSettings(settings = {}):
	settings['problem_name']         		= ['', "string for name used in file naming"]
	settings['title']         				= [None, "string for name used in plotting"]
	settings['logger_outname']         		= [None, "string for name used in logger file naming"]
	settings['printing_items']				= [{'sweeps':'sweeps','Loss':'loss_train','acc train':'accuracy_train',\
												'||g||':'||g||','Loss test':'loss_test','acc test':'accuracy_test',\
												'maxacc test':'max_accuracy_test','alpha':'alpha'},\
																			"Dictionary of items for printing"]

	settings['verbose']         			= [True, "Boolean for printing"]

	settings['intra_threads']         		= [2, "Setting for intra op parallelism"]
	settings['inter_threads']         		= [2, "Setting for inter op parallelism"]

	# Optimizer settings
	settings['optimizer']                	= ['lrsfn', "String to denote choice of optimizer"]
	settings['alpha']                		= [5e-2, "Initial steplength, or learning rate"]
	settings['hessian_low_rank']			= [20, "Low rank to be used for LRSFN / SFN"]
	settings['fixed_step']					= [False, " True means steps of length alpha will be taken at each iteration"]
	settings['max_backtrack']				= [10, "Maximum number of backtracking iterations for each line search"]

	# Range finding settings for LRSFN
	settings['range_finding']				= [None,"Range finding, if None then r = hessian_low_rank\
	 														Choose from None, 'arf', 'naarf'"]
	settings['range_rel_error_tolerance']   = [5, "Error tolerance for error estimator in adaptive range finding"]
	settings['range_abs_error_tolerance']   = [50, "Error tolerance for error estimator in adaptive range finding"]
	settings['range_block_size']        	= [10, "Block size used in range finder"]


	settings['max_sweeps']					= [10,"Maximum number of times through the data (measured in epoch equivalents"]

	#Settings for recording spectral information during training
	settings['record_spectrum']         	= [False, "Boolean for recording spectrum during training"]
	settings['record_last_rq_std']         	= [False, "Boolean for recording last RQ std for last eigenvector of LRSFN"]
	settings['spec_frequency'] 				= [10, "Frequency for recording of spectrum"]
	settings['rayleigh_quotients']         	= [True, "Boolean for recording of spectral variance during training"]
	settings['rq_data_size'] 				= [None,"Amount of training data to be used, None means all"]
	settings['rq_samps']					= [100,"Number of partitions used for sample average statistics of RQs"]
	settings['target_rank']					= [100,"Target rank for randomized eigenvalue solver"]
	settings['oversample']					= [10,"Oversampling for randomized eigenvalue solver"]

	return ParameterList(settings)


class HessianlearnModel(ABC):
	def __init__(self,problem,regularization,data,settings = HessianlearnModelSettings({})):

		self._problem = problem
		self._regularization = regularization
		self._data = data

		self.settings = settings

		if self.settings['verbose']:
			print(80*'#')
			print(('Size of configuration space:  '+str(self.problem.dimension)).center(80))
			print(('Size of training data: '+str(self.data.train_data_size)).center(80))
			# Approximate data needed is d_W / output
			if len(self.problem.y_prediction.shape) >2:
				output_dimension = 1.
				for shape in self.problem.y_prediction.shape[1:]:
					output_dimension *= shape.value
				# print('Shape = ',self.problem.y_prediction.shape[1:].value)
				# output_dimension = None
			else:
				output_dimension = float(self.problem.y_prediction.shape[-1].value)
			print(('Approximate data cardinality needed: '\
				+str(int(float(self.problem.dimension)/output_dimension	))).center(80))
			print(80*'#')

		# self._sess = None
		self._optimizer = None




	@property
	def sess(self):
		return self._sess
	
	@property
	def optimizer(self):
		return self._optimizer

	@property
	def fit(self):
		return self._fit

	@property
	def problem(self):
		return self._problem
	
	@property
	def regularization(self):
		return self._regularization
	
	@property
	def data(self):
		return self._data

	@property
	def logger(self):
		return self._logger

	def _initialize_optimizer(self, sess,settings = None):
		if settings == None:
			settings = self.settings
		if 'rank' in self.settings['printing_items'].keys():
			_ = self.settings['printing_items'].pop('rank',None)

		self._logger['optimizer'] = settings['optimizer']
		# assert self.sess is not None
		if settings['optimizer'] == 'adam':
			print(('Using Adam optimizer').center(80))
			print(('Batch size = '+str(self.data._batch_size)).center(80))
			optimizer = Adam(self.problem,self.regularization,sess)
			optimizer.parameters['alpha'] = settings['alpha']
			optimizer.alpha = settings['alpha']
			self._logger['alpha'][0] = settings['alpha']

		elif settings['optimizer'] == 'gd':
			print('Using gradient descent optimizer with line search'.center(80))
			print(('Batch size = '+str(self.data._batch_size)).center(80))
			optimizer = GradientDescent(self.problem,self.regularization,sess)
			optimizer.parameters['globalization'] = 'line_search'
			self._logger['globalization'] = 'line_search'
			optimizer.parameters['max_backtracking_iter'] = 8
		elif settings['optimizer'] == 'incg':
			if not settings['fixed_step']:
				print('Using inexact Newton CG optimizer with line search'.center(80))
				print(('Batch size = '+str(self.data._batch_size)).center(80))
				print(('Hessian batch size = '+str(self.data._hessian_batch_size)).center(80))
				optimizer = InexactNewtonCG(self.problem,self.regularization,sess)
				optimizer.parameters['globalization'] = 'line_search'
				self._logger['globalization'] = 'line_search'
				optimizer.parameters['max_backtracking_iter'] = settings['max_backtrack']
			else:
				print('Using inexact Newton CG optimizer with fixed step'.center(80))
				print(('Batch size = '+str(self.data._batch_size)).center(80))
				print(('Hessian batch size = '+str(self.data._hessian_batch_size)).center(80))
				optimizer = InexactNewtonCG(self.problem,LowRankSaddleFreeNewton.regularization,sess)
				optimizer.parameters['globalization'] = None
				optimizer.alpha = settings['alpha']
				self._logger['alpha'][0] = settings['alpha']
		elif settings['optimizer'] == 'lrsfn':
			self.settings['printing_items']['rank'] = 'hessian_low_rank'
			if not settings['fixed_step']:
				print('Using low rank SFN optimizer with line search'.center(80))
				print(('Batch size = '+str(self.data._batch_size)).center(80))
				print(('Hessian batch size = '+str(self.data._hessian_batch_size)).center(80))
				print(('Hessian low rank = '+str(settings['hessian_low_rank'])).center(80))
				optimizer = LowRankSaddleFreeNewton(self.problem,self.regularization,sess)
				optimizer.parameters['globalization'] = 'line_search'
				self._logger['globalization'] = 'line_search'
				optimizer.parameters['max_backtracking_iter'] = settings['max_backtrack']
				optimizer.parameters['hessian_low_rank'] = settings['hessian_low_rank']
				optimizer.parameters['range_finding'] = settings['range_finding']
				optimizer.parameters['range_rel_error_tolerance'] =	settings['range_rel_error_tolerance']
				optimizer.parameters['range_block_size'] =	settings['range_block_size']

			else:
				print('Using low rank SFN optimizer with fixed step'.center(80))
				print(('Batch size = '+str(self.data._batch_size)).center(80))
				print(('Hessian batch size = '+str(self.data._hessian_batch_size)).center(80))
				print(('Hessian low rank = '+str(settings['hessian_low_rank'])).center(80))
				optimizer = LowRankSaddleFreeNewton(self.problem,self.regularization,sess)
				optimizer.parameters['globalization'] = None
				optimizer.parameters['hessian_low_rank'] = settings['hessian_low_rank']
				optimizer.parameters['alpha'] = settings['alpha']
				optimizer.parameters['range_finding'] = settings['range_finding']
				optimizer.parameters['range_rel_error_tolerance'] =	settings['range_rel_error_tolerance']
				optimizer.parameters['range_block_size'] =	settings['range_block_size']
				optimizer.alpha = settings['alpha']
				self._logger['alpha'][0] = settings['alpha']
			if self.settings['range_finding'] is None:
				self._logger['hessian_low_rank'][0] = settings['hessian_low_rank']
		elif settings['optimizer'] == 'sgd':
			print(('Using stochastic gradient descent optimizer').center(80))
			print(('Batch size = '+str(self._data._batch_size)).center(80))
			optimizer = GradientDescent(self.problem,self.regularization,sess)
			optimizer.parameters['alpha'] = settings['alpha']
			optimizer.alpha = settings['alpha']
			self._logger['alpha'][0] = settings['alpha']
		else:
			raise
		self._optimizer = optimizer



	def _initialize_logging(self):
		# Initialize Logging 
		logger = {}
		logger['dimension'] = self.problem.dimension
		logger['problem_name'] = self.settings['problem_name']
		logger['title'] = self.settings['title']
		logger['batch_size'] = self.data._batch_size
		logger['hessian_batch_size'] = self.data._hessian_batch_size
		logger['loss_train'] = {}
		logger['loss_test'] = {}
		logger['||g||'] ={}
		logger['sweeps'] = {}
		logger['time'] = {}
		logger['best_weight'] = []
		logger['optimizer'] = None
		logger['alpha'] = None
		logger['globalization'] = None
		logger['hessian_low_rank'] = {}

		logger['accuracy_test'] = {}
		logger['accuracy_train'] = {}

		logger['max_accuracy_test'] = {}
		logger['alpha'] = {}

		if self.settings['record_spectrum']:
			logger['full_train_eigenvalues'] = {}
			logger['train_eigenvalues'] = {}
			logger['test_eigenvalues'] = {}
			logger['rq_std'] = {}
		elif self.settings['record_last_rq_std']:
			logger['last_rq_std'] = {}


		self._logger = logger

		if not os.path.isdir(self.settings['problem_name']+'_logging/'):
			os.makedirs(self.settings['problem_name']+'_logging/')

		# Set outname for logging file
		if self.settings['logger_outname'] is None:
			logger_outname = str(datetime.date.today())+'-'+self.settings['optimizer']+'-dW='+str(self.problem.dimension)
			if self.settings['optimizer'] in ['lrsfn','incg','gd']:
				if self.settings['fixed_step']:
					logger_outname += '-alpha='+str(self.settings['alpha'])
				if self.settings['optimizer'] == 'lrsfn':
					logger_outname += '-rank='+str(self.settings['hessian_low_rank'])
			else:
				logger_outname += '-alpha='+str(self.settings['alpha'])

			if self.settings['problem_name'] is not None:
				logger_outname = self.settings['problem_name']+'-'+logger_outname
		else:
			logger_outname = self.settings['logger_outname']
		self.logger_outname = logger_outname
	


	def _fit(self,options = None, w_0 = None):
		# Consider doing scope managed sess
		# For now I will use the sess as a member variable
		# self._sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=self.settings['intra_threads'],\
		# 									inter_op_parallelism_threads=self.settings['inter_threads']))
		with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=self.settings['intra_threads'],\
											inter_op_parallelism_threads=self.settings['inter_threads'])) as sess:
			# Initialize logging:
			self._initialize_logging()
			# Initialize the optimizer
			self._initialize_optimizer(sess)
			# After optimizer is instantiated, we call the global variables initializer
			sess.run(tf.global_variables_initializer())
			# Load initial guess if requested:
			if w_0 is not None:
				if type(w_0) is list:
					self._problem._NN.set_weights(w_0)
				else:
					try:
						sess.run(self.problem._assignment_ops,feed_dict = {self.problem._assignment_placeholder:w_0})
					except:
						print(80*'#')
						print('Issue setting weights manually'.center(80))
						print('tf.global_variables_initializer used to initial instead'.center(80))		
			else:
				pass
				# random_state = np.random.RandomState(seed = 0)
				# w_0 = random_state.randn(problem.dimension)
				# sess.run(problem._assignment_ops,feed_dict = {problem._assignment_placeholder:w_0})

			if self.settings['verbose']:
				self.print(first_print = True)

			x_test, y_test = next(iter(self.data.test))
			if self.problem.is_autoencoder:
				test_dict = {self.problem.x: x_test}
			else:
				test_dict = {self.problem.x: x_test,self.problem.y_true: y_test}

			# Iteration Loop
			max_sweeps = self.settings['max_sweeps']
			train_data = iter(self.data.train)
			x_batch,y_batch = next(train_data)
			sweeps = 0
			min_test_loss = np.inf
			max_test_acc = -np.inf
			t0 = time.time()
			for iteration, (data_g,data_H) in enumerate(zip(self.data.train,self.data.hess_train)):
				# Unpack data pairs
				x_batch,y_batch = data_g
				x_hess, y_hess = data_H
				# Instantiate data dictionaries for this iteration
				if self.problem.is_autoencoder:
					train_dict = {self.problem.x: x_batch}
					hess_dict = {self.problem.x: x_hess}
				else:
					train_dict = {self.problem.x: x_batch, self.problem.y_true: y_batch}
					hess_dict = {self.problem.x: x_hess, self.problem.y_true: y_hess}
				# Log time / sweep number
				# Every element of dictionary is 
				# keyed by the optimization iteration
				self._logger['time'][iteration] = time.time() - t0
				self._logger['sweeps'][iteration] = sweeps
				# Log information for training data
				if hasattr(self.problem,'accuracy'):
					norm_g, loss_train, accuracy_train = sess.run([self.problem.norm_g,self.problem.loss,self.problem.accuracy],train_dict)
					self._logger['accuracy_train'][iteration] = accuracy_train
				else:
					norm_g, loss_train = sess.run([self.problem.norm_g,self.problem.loss],train_dict)
				self._logger['||g||'][iteration] = norm_g
				self._logger['loss_train'][iteration] = loss_train
				# Log for test data
				if hasattr(self.problem,'accuracy'):
					loss_test,	accuracy_test = sess.run([self.problem.loss,self.problem.accuracy],test_dict)
					self._logger['accuracy_test'][iteration] = accuracy_test
				else:
					loss_test = sess.run(self.problem.loss,test_dict)
				self._logger['loss_test'][iteration] = loss_test
				min_test_loss = min(min_test_loss,loss_test)
				max_test_acc = max(max_test_acc,accuracy_test)
				self._logger['max_accuracy_test'][iteration] = max_test_acc
				self._logger['alpha'][iteration] = self.optimizer.alpha
				if self.settings['optimizer'] == 'lrsfn':
					self._logger['hessian_low_rank'][iteration] = self.optimizer.rank

				if accuracy_test == max_test_acc:
					self._best_weights = sess.run(self.problem._w)
					if len(self._logger['best_weight']) > 2:
						self._logger['best_weight'].pop(0)
					acc_weight_tuple = (accuracy_test,accuracy_train,sess.run(self.problem._flat_w))
					self._logger['best_weight'].append(acc_weight_tuple) 

				sweeps = np.dot(self.data.batch_factor,self.optimizer.sweeps)
				if self.settings['verbose'] and iteration % 1 == 0:
					# Print once each epoch
					self.print(iteration = iteration)

				if np.isnan(loss_train) or np.isnan(norm_g):
					print(80*'#')
					print('Encountered nan, exiting'.center(80))
					print(80*'#')
					return
				try:
					self.optimizer.minimize(train_dict,hessian_feed_dict=hess_dict)
				except:
					self.optimizer.minimize(train_dict)



				# If LRSFN used with range_finding need to record rank at each iteration
				# If LRSFN used with range_finding need to record rank at each iteration
				# If LRSFN used with range_finding need to record rank at each iteration
				# If LRSFN used with range_finding need to record rank at each iteration
				# If LRSFN used with range_finding need to record rank at each iteration
				# If LRSFN used with range_finding need to record rank at each iteration
				# If LRSFN used with range_finding need to record rank at each iteration
				# If LRSFN used with range_finding need to record rank at each iteration
				# If LRSFN used with range_finding need to record rank at each iteration
				# If LRSFN used with range_finding need to record rank at each iteration
				# If LRSFN used with range_finding need to record rank at each iteration
				# If LRSFN used with range_finding need to record rank at each iteration
				# If LRSFN used with range_finding need to record rank at each iteration
				# If LRSFN used with range_finding need to record rank at each iteration
				# print('Fix this recording thing for LRSFN with adaptive range finding')

				if self.settings['record_spectrum'] and iteration%self.settings['spec_frequency'] ==0:
					self._record_spectrum(iteration)
				elif self.settings['record_last_rq_std'] and self.settings['optimizer'] == 'lrsfn':
					logger['last_rq_std'][iteration] = self.optimizer._rq_std
				with open(self.settings['problem_name']+'_logging/'+ self.logger_outname +'.pkl', 'wb+') as f:
					pickle.dump(self.logger, f, pickle.HIGHEST_PROTOCOL)

				if sweeps > max_sweeps:
					break

				

		# The weights need to be manually set once the session scope is closed.
		try:
			self._problem._NN.set_weights(self._best_weights)
		except:
			pass

	
	def _record_spectrum(self,iteration):
		k_rank = self.settings['target_rank']
		p_oversample = self.settings['oversample']
		
		
		if self.settings['rayleigh_quotients']:
			train_data_xs = self.data.train._data.x[0:self.settings['rq_data_size']]
			train_data_ys = self.data.train._data.y[0:self.settings['rq_data_size']]
			test_data_xs = self.data.test._data.x
			test_data_ys = self.data.test._data.y
			if self.problem.is_autoencoder:
				full_train_dict = {self.problem.x:train_data_xs}
				full_test_dict = {self.problem.x:test_data_xs}
			else:
				full_train_dict = {self.problem.x:train_data_xs,self.problem.y_true:train_data_ys}
				full_test_dict = {self.problem.x:test_data_xs,self.problem.y_true:test_data_ys}

			d_full_train, U_full_train = low_rank_hessian(self.optimizer,full_train_dict,k_rank,p_oversample,verbose=True)
			self._logger['full_train_eigenvalues'][iteration] = d_full_train
			# Initialize array for Rayleigh quotient samples
			RQ_samples = np.zeros((self.settings['rq_samps'],U_full_train.shape[1]))

			partitioned_dictionaries_train = self.problem._partition_dictionaries(full_train_dict,self.settings['rq_samps'])

			try:
				from tqdm import tqdm
				for samp_i,sample_dictionary in enumerate(tqdm(partitioned_dictionaries_train)):
					RQ_samples[samp_i] = self.optimizer.H.quadratics(U_full_train,sample_dictionary)
			except:
				print('Issue with tqdm')
				for samp_i,sample_dictionary in enumerate(partitioned_dictionaries_train):
					RQ_samples[samp_i] = self.optimizer.H.quadratics(U_full_train,sample_dictionary)

			RQ_sample_std = np.std(RQ_samples,axis = 0)
			self._logger['rq_std'][iteration] = RQ_sample_std

		else:
			d_full,_ = low_rank_hessian(self.optimizer,train_dict,k_rank,p_oversample)
			self._logger['train_eigenvalues'][iteration] = d_full
			d_test,_ = low_rank_hessian(self.optimizer,test_dict,k)
			self._logger['test_eigenvalues'][iteration] = d_test



	def print(self,first_print = False,iteration = None):
		for key in self.settings['printing_items'].keys():
			assert self.settings['printing_items'][key] in self._logger.keys(), 'item '+str(self.settings['printing_items'][key])+' not in logger'
		if first_print:
			print(80*'#')
			format_string = ''
			for i in range(len(self.settings['printing_items'].keys())):
				if i == 0:
					format_string += '{0:7} '
				else:
					format_string += '{'+str(i)+':10} '
			string_tuples = (print_string.center(8) for print_string in self.settings['printing_items'].keys())
			print(format_string.format(*string_tuples))
		else:
			format_string = ''
			for i,key in enumerate(self.settings['printing_items'].keys()):
				if 'sweeps' in key:
					format_string += '{'+str(i)+':^8.2f} '
				elif 'acc' in key:
					if False:
						pass
					else:
						format_string += '{'+str(i)+':.3%} '
				elif 'rank' in key: 
					format_string += '{'+str(i)+':10} '
				else:
					format_string += '{'+str(i)+':1.4e} '

			value_tuples = (self._logger[self.settings['printing_items'][item]][iteration] for item in self.settings['printing_items'])


			print(format_string.format(*value_tuples))


