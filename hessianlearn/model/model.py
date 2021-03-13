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
# tf.compat.v1.enable_eager_execution()
if int(tf.__version__[0]) > 1:
	import tensorflow.compat.v1 as tf
	tf.disable_v2_behavior()
	# tf.enable_eager_execution()

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

from ..problem.hessian import Hessian, HessianWrapper
from ..algorithms.varianceBasedNystrom import variance_based_nystrom



def HessianlearnModelSettings(settings = {}):
	settings['problem_name']         		= ['', "string for name used in file naming"]
	settings['title']         				= [None, "string for name used in plotting"]
	settings['logger_outname']         		= [None, "string for name used in logger file naming"]
	settings['printing_items']				= [{'sweeps':'sweeps','Loss':'train_loss','acc train':'train_acc',\
												'||g||':'||g||','Loss val':'val_loss','acc val':'val_acc',\
												'maxacc val':'max_val_acc','alpha':'alpha'},\
																			"Dictionary of items for printing"]
	settings['printing_sweep_frequency']    = [1, "Print only every this many sweeps"]
	settings['validate_frequency']			= [1, "Only compute validation quantities every X sweeps"]
	settings['save_weights']				= [True, "Whether or not to save the best weights"]


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
															Choose from None, 'arf', 'naarf','vn' "]
	settings['range_rel_error_tolerance']   = [5, "Error tolerance for error estimator in adaptive range finding"]
	settings['range_abs_error_tolerance']   = [50, "Error tolerance for error estimator in adaptive range finding"]
	settings['range_block_size']        	= [10, "Block size used in range finder"]
	settings['max_sweeps']					= [10,"Maximum number of times through the data (measured in epoch equivalents"]

	settings['max_bad_vectors_nystrom']     = [5, "Number of maximum bad vectors for variance based Nystrom"]
	settings['max_vectors_nystrom']       	= [40, "Number of maximum vectors for variance based Nystrom"]
	settings['nystrom_std_tolerance']       = [0.5, "Noise to eigenvalue ratio used for Nystrom truncation"]


	# Initial weights for specific layers 
	settings['layer_weights'] 				= [{},"Dictionary of layer name key and weight \
													values for weights set after global variable initialization "]

	# Settings for recording spectral information during training
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
	def __init__(self,problem,regularization,data,derivative_data = None,settings = HessianlearnModelSettings({})):

		self._problem = problem
		self._regularization = regularization
		self._data = data

		self._derivative_data = derivative_data

		self.settings = settings

		if self.settings['verbose']:
			print(80*'#')
			print(('Size of configuration space:  '+str(self.problem.dimension)).center(80))
			print(('Size of training data: '+str(self.data.train_data_size)).center(80))
			print(('Approximate data cardinality needed: '\
				+str(int(float(self.problem.dimension)/self.problem.output_dimension	))).center(80))
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
	def derivative_data(self):
		return self._derivative_data

	@property
	def logger(self):
		return self._logger

	def _initialize_optimizer(self, sess,settings = None):
		assert sess is not None
		if settings == None:
			settings = self.settings
		if 'rank' in self.settings['printing_items'].keys():
			_ = self.settings['printing_items'].pop('rank',None)

		self._logger['optimizer'] = settings['optimizer']
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
				optimizer = InexactNewtonCG(self.problem,self.regularization,sess)
				optimizer.parameters['globalization'] = None
				optimizer.parameters['alpha'] = settings['alpha']
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
				optimizer.parameters['max_bad_vectors_nystrom'] = settings['max_bad_vectors_nystrom'] 
				optimizer.parameters['max_vectors_nystrom']  = settings['max_vectors_nystrom'] 
				optimizer.parameters['nystrom_std_tolerance']  = settings['nystrom_std_tolerance'] 

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
				optimizer.parameters['max_bad_vectors_nystrom'] = settings['max_bad_vectors_nystrom'] 
				optimizer.parameters['max_vectors_nystrom']  = settings['max_vectors_nystrom'] 
				optimizer.parameters['nystrom_std_tolerance']  = settings['nystrom_std_tolerance'] 
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
			raise NotImplementedError('Unsupported choice of optimizer')
		self._optimizer = optimizer



	def _initialize_logging(self):
		# Initialize Logging 
		logger = {}
		logger['dimension'] = self.problem.dimension
		logger['problem_name'] = self.settings['problem_name']
		logger['title'] = self.settings['title']
		logger['batch_size'] = self.data._batch_size
		logger['hessian_batch_size'] = self.data._hessian_batch_size
		logger['train_loss'] = {}
		logger['val_loss'] = {}
		logger['||g||'] ={}
		logger['sweeps'] = {}
		logger['total_time'] = {}
		logger['time'] = {}
		logger['best_weights'] = None
		logger['optimizer'] = None
		logger['alpha'] = None
		logger['globalization'] = None
		logger['hessian_low_rank'] = {}

		logger['val_acc'] = {}
		logger['train_acc'] = {}


		logger['max_val_acc'] = {}
		logger['alpha'] = {}

		if self.settings['record_spectrum']:
			logger['full_train_eigenvalues'] = {}
			logger['train_eigenvalues'] = {}
			logger['val_eigenvalues'] = {}
			logger['rq_std'] = {}
		elif self.settings['record_last_rq_std']:
			logger['last_rq_std'] = {}

		if hasattr(self.problem,'_variance_reduction'):
			logger['val_variance_reduction'] = {}


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
		with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=self.settings['intra_threads'],\
											inter_op_parallelism_threads=self.settings['inter_threads'])) as sess:
			# Re initialize data
			self.data.reset()
			# Initialize logging:
			self._initialize_logging()
			# Initialize the optimizer
			self._initialize_optimizer(sess)
			# After optimizer is instantiated, we call the global variables initializer
			sess.run(tf.global_variables_initializer())
			################################################################################
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
						print('tf.global_variables_initializer() used to initial instead'.center(80))
			# This handles a corner case for weights that are not trainable,
			# but still get set by the tf.global_variables_initializer()
			for layer_name,weight in self.settings['layer_weights'].items():
				self.problem._NN.get_layer(layer_name).set_weights(weight)
			################################################################################
			# First print
			if self.settings['verbose']:
				self.print(first_print = True)
			################################################################################
			# Load validation data
			try:
				x_val, y_val = next(iter(self.data.validation))
				if self.problem.is_autoencoder:
					val_dict = {self.problem.x: x_val}
				elif self.problem.is_gan:
					random_state_gan = np.random.RandomState(seed = 0)
					# Should the first dimension here agree with the size of the validation data?
					noise = random_state_gan.normal(size = (self.data.batch_size, self.problem.noise_dimension))
					val_dict = {self.problem.x: x_val,self.problem.noise : noise}
				else:
					val_dict = {self.problem.x: x_val,self.problem.y_true: y_val}
			except:
				val_data = next(iter(self.data.validation))
				if self.problem.is_autoencoder:
					val_dict = {self.problem.x: val_data[self.problem.x]}
				elif self.problem.is_gan:
					random_state_gan = np.random.RandomState(seed = 0)
					# Should the first dimension here agree with the size of the validation data?
					noise = random_state_gan.normal(size = (self.data.batch_size, self.problem.noise_dimension))
					val_dict = {self.problem.x: val_data[self.problem.x],self.problem.noise : noise}
				else:
					val_dict = {self.problem.x: val_data[self.problem.x],self.problem.y_true: val_data[self.problem.y_true]}

			################################################################################
			# Prepare for iteration
			max_sweeps = self.settings['max_sweeps']
			train_data = iter(self.data.train)
			sweeps = 0
			min_val_loss = np.inf
			max_val_acc = -np.inf
			validation_duration = 0.0
			t0 = time.time()
			for iteration, (data_g,data_H) in enumerate(zip(self.data.train,self.data.hess_train)):
				################################################################################
				# Unpack data pairs and update dictionary as needed
				assert type(data_g) is dict and type(data_H) is dict, 'Old hessianlearn data object has been deprecated, use dictionary iterator now'
				train_dict = data_g
				hess_dict = data_H
				if self.problem.is_autoencoder:
					assert not hasattr(self.problem,'y_true')
				elif self.problem.is_gan:
					noise = random_state_gan.normal(size = (self.data.batch_size, self.problem.noise_dimension))
					train_dict[self.problem.noise] = noise
					noise_hess = random_state_gan.normal(size = (self.data.hessian_batch_size, self.problem.noise_dimension))
					hess_dict[self.problem.noise] = noise_hess 
				################################################################################
				# Log time / sweep number
				# Every element of dictionary is 
				# keyed by the optimization iteration
				self._logger['total_time'][iteration] = time.time() - t0 - validation_duration 
				self._logger['sweeps'][iteration] = sweeps
				if iteration-1 not in self._logger['time'].keys():
					self._logger['time'][iteration] = self._logger['total_time'][iteration]
				else:
					self._logger['time'][iteration] = self._logger['total_time'][iteration] - self._logger['total_time'][iteration-1]
				self._logger['sweeps'][iteration] = sweeps
				# Log information for training data
				# Much more efficient to have the actual optimizer / minimize() function
				# return this information since it has to query the graph
				# This is a place to cut down on computational graph queries
				if hasattr(self.problem,'accuracy'):
					norm_g, train_loss, train_acc = sess.run([self.problem.norm_g,self.problem.loss,self.problem.accuracy],train_dict)
					self._logger['train_acc'][iteration] = train_acc
				else:
					norm_g, train_loss = sess.run([self.problem.norm_g,self.problem.loss],train_dict)
				self._logger['||g||'][iteration] = norm_g
				self._logger['train_loss'][iteration] = train_loss
				# Logging of optimization hyperparameters 
				# These can change at each iteration when using adaptive range finding
				# or globalization like line search
				self._logger['alpha'][iteration] = self.optimizer.alpha
				if self.settings['optimizer'] == 'lrsfn':
					self._logger['hessian_low_rank'][iteration] = self.optimizer.rank

				# Update the sweeps
				sweeps = np.dot(self.data.batch_factor,self.optimizer.sweeps)
				################################################################################
				# Log for validation data
				validate_this_iteration = False
				validate_frequency = self.settings['validate_frequency']
				if self.settings['validate_frequency'] is None or iteration == 0:
					validate_this_iteration = True
				else:
					validate_this_iteration = self._check_sweep_remainder_condition(iteration,self.settings['validate_frequency'])

				if hasattr(self.problem,'accuracy'):
					if validate_this_iteration:
						validation_start = time.time()
						if hasattr(self.problem,'_variance_reduction'):
							val_loss,	val_acc, val_var_red =\
							 sess.run([self.problem.loss,self.problem.accuracy,self.problem.variance_reduction],val_dict)
							self._logger['val_variance_reduction'][iteration] = val_var_red
						else:
							val_loss,	val_acc = sess.run([self.problem.loss,self.problem.accuracy],val_dict)
						self._logger['val_acc'][iteration] = val_acc
						max_val_acc = max(max_val_acc,val_acc)
						validation_duration += time.time() - validation_start
					self._logger['max_val_acc'][iteration] = max_val_acc
				else:
					if validate_this_iteration:
						validation_start = time.time()
						val_loss = sess.run(self.problem.loss,val_dict)
						validation_duration += time.time() - validation_start
					self._logger['val_loss'][iteration] = val_loss

				min_val_loss = min(min_val_loss,val_loss)

				################################################################################
				# Save the best weights based on validation accuracy or loss
				if hasattr(self.problem,'accuracy') and val_acc == max_val_acc:
					weight_dictionary = {}
					for layer in self.problem._NN.layers:
						weight_dictionary[layer.name] = self.problem._NN.get_layer(layer.name).get_weights()
					self._best_weights = weight_dictionary
					if self.settings['save_weights']:
						self._logger['best_weights'] = weight_dictionary
				elif val_loss == min_val_loss:
					weight_dictionary = {}
					if self.problem.is_gan:
						weight_dictionary['generator'] = {}
						for layer in self.problem._generator.layers:
							weight_dictionary['generator'][layer.name] = self.problem._generator.get_layer(layer.name).get_weights()
						weight_dictionary['discriminator'] = {}
						for layer in self.problem._discriminator.layers:
							weight_dictionary['discriminator'][layer.name] = self.problem._discriminator.get_layer(layer.name).get_weights()
					else:
						for layer in self.problem._NN.layers:
							weight_dictionary[layer.name] = self.problem._NN.get_layer(layer.name).get_weights()
					self._best_weights = weight_dictionary
					if self.settings['save_weights']:
						self._logger['best_weights'] = weight_dictionary
				################################################################################
				# Printing
				if self.settings['verbose']:
					# Print once each epoch
					self.print(iteration = iteration)
				################################################################################
				# Checking for nans!
				if np.isnan(train_loss) or np.isnan(norm_g):
					print(80*'#')
					print('Encountered nan, exiting'.center(80))
					print(80*'#')
					return
				################################################################################
				# Actual optimization takes place here
				try:
					self.optimizer.minimize(train_dict,hessian_feed_dict=hess_dict)
				except:
					self.optimizer.minimize(train_dict)
				################################################################################
				# Recording the spectrum
				if self.settings['record_spectrum'] and iteration%self.settings['spec_frequency'] ==0:
					self._record_spectrum(iteration)
				elif self.settings['record_last_rq_std'] and self.settings['optimizer'] == 'lrsfn':
					logger['last_rq_std'][iteration] = self.optimizer._rq_std
				with open(self.settings['problem_name']+'_logging/'+ self.logger_outname +'.pkl', 'wb+') as f:
					pickle.dump(self.logger, f, pickle.HIGHEST_PROTOCOL)
				################################################################################
				# Check if max_sweeps condition has been met
				if sweeps > max_sweeps:
					# One last print
					self.print(iteration = iteration,force_print = True)
					break
		################################################################################
		# Post optimization
		# The weights need to be manually set once the session scope is closed.
		try:
			if self.problem.is_gan:
				for layer_name in self._best_weights['generator']:
					self.problem._generator.get_layer(layer_name).set_weights(self._best_weights['generator'][layer_name])
				for layer_name in self._best_weights['discriminator']:
					self.problem._discriminator.get_layer(layer_name).set_weights(self._best_weights['discriminator'][layer_name])
			else:
				for layer_name in self._best_weights:
					self._problem._NN.get_layer(layer_name).set_weights(self._best_weights[layer_name])
		except:
			print('Error setting the weights after training')

	
	def _record_spectrum(self,iteration):
		k_rank = self.settings['target_rank']
		p_oversample = self.settings['oversample']
		
		
		if self.settings['rayleigh_quotients']:
			print('It is working')
			my_t0 = time.time()

			train_data = self.data.train._data
			val_data = self.data.val._data

			if self.problem.is_autoencoder:
				full_train_dict = {self.problem.x:train_data[self.problem.x]}
				full_val_dict = {self.problem.x:val_data[self.problem.x]}
			else:
				full_train_dict = {self.problem.x:train_data[self.problem.x],self.problem.y_true:train_data[self.problem.y_true]}
				full_val_dict = {self.problem.x:val_data[self.problem.x],self.problem.y_true:val_data[self.problem.y_true]}

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
			d_val,_ = low_rank_hessian(self.optimizer,val_dict,k)
			self._logger['val_eigenvalues'][iteration] = d_val



	def print(self,first_print = False,iteration = None,force_print = False):
		################################################################################
		# Check to make sure everything requested to print exists
		for key in self.settings['printing_items'].keys():
			assert self.settings['printing_items'][key] in self._logger.keys(), 'item '+str(self.settings['printing_items'][key])+' not in logger'
		################################################################################
		# First print : column names
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
		################################################################################
		# Iteration prints
		else:
			format_string = ''
			for i,key in enumerate(self.settings['printing_items'].keys()):
				if iteration not in self._logger[self.settings['printing_items'][key]].keys():
					format_string += '{'+str(i)+':7} '
				elif 'sweeps' in key:
					format_string += '{'+str(i)+':^8.2f} '
				elif 'acc' in key:
					value = self._logger[self.settings['printing_items'][key]][iteration]
					if value < 0.0:
						format_string += '{'+str(i)+':.2%} '
					else:
						format_string += '{'+str(i)+':.3%} '
				elif 'rank' in key: 
					format_string += '{'+str(i)+':10} '
				else:
					format_string += '{'+str(i)+':1.4e} '
			################################################################################
			# Check sweep remainder condition here
			every_sweep = self.settings['printing_sweep_frequency']
			if every_sweep is None or not ('sweeps' in self.settings['printing_items'].keys()):
				print_this_time = True
			elif iteration == 0 or force_print:
				print_this_time = True
			elif every_sweep is not None:
				assert iteration is not None
				print_this_time = self._check_sweep_remainder_condition(iteration,every_sweep)
			################################################################################
			# Actual printing
			if print_this_time:
				value_list = []
				for item in self.settings['printing_items']:
					if iteration in self._logger[self.settings['printing_items'][item]]:
						value_list.append(self._logger[self.settings['printing_items'][item]][iteration])
					else:
						value_list.append(8*' ')
				# value_tuples = (self._logger[self.settings['printing_items'][item]][iteration] for item in self.settings['printing_items'])
				print(format_string.format(*value_list))


	def _check_sweep_remainder_condition(self,iteration, sweeps_divisor):

		last_sweep_floor_div,last_sweep_rem = np.divmod(self._logger['sweeps'][iteration-1], sweeps_divisor)
		this_sweep_floor_div,this_sweep_rem = np.divmod(self._logger['sweeps'][iteration], sweeps_divisor)

		if this_sweep_floor_div > last_sweep_floor_div:
			return True
		else:
			return False
