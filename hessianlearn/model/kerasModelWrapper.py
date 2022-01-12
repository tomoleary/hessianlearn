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
import warnings

import sys, os, pickle, time, datetime


from ..utilities.parameterList import ParameterList

from ..problem.problem import KerasModelProblem

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



def KerasModelWrapperSettings(settings = {}):
	settings['problem_name']         		= ['', "string for name used in file naming"]
	settings['title']         				= [None, "string for name used in plotting"]
	settings['logger_outname']         		= [None, "string for name used in logger file naming"]
	settings['printing_items']				= [{'sweeps':'sweeps','Loss':'train_loss','acc ':'train_acc',\
												'||g||':'||g||','Lossval':'val_loss','accval':'val_acc',\
												'maxacc':'max_val_acc','alpha':'alpha'},\
																			"Dictionary of items for printing"]
	settings['printing_sweep_frequency']    = [1, "Print only every this many sweeps"]
	settings['validate_frequency']			= [1, "Only compute validation quantities every X sweeps"]
	settings['save_weights']				= [True, "Whether or not to save the best weights"]
	settings['max_sweeps']					= [10,"Maximum number of times through the data (measured in epoch equivalents"]


	settings['verbose']         			= [True, "Boolean for printing"]

	settings['intra_threads']         		= [2, "Setting for intra op parallelism"]
	settings['inter_threads']         		= [2, "Setting for inter op parallelism"]



	# Initial weights for specific layers 
	settings['layer_weights'] 				= [{},"Dictionary of layer name key and weight \
													values for weights set after global variable initialization "]

	# Settings for recording spectral information during training
	settings['record_spectrum']         	= [False, "Boolean for recording spectrum during training"]
	settings['target_rank']					= [100,"Target rank for randomized eigenvalue solver"]
	settings['oversample']					= [10,"Oversampling for randomized eigenvalue solver"]

	return ParameterList(settings)


class KerasModelWrapper(ABC):
	def __init__(self,kerasModel,regularization= None,optimizer = None,\
		optimizer_parameters = None,hessian_block_size = None, settings = KerasModelWrapperSettings({})):
		warnings.warn('Experimental Class! Be Wary')
		# Check hessian blocking condition here?
		if optimizer_parameters is not None:
			if ('hessian_low_rank' in optimizer_parameters.data.keys()) and (hessian_block_size is not None):
				hessian_block_size = max(optimizer_parameters['hessian_low_rank'],hessian_block_size)


		self._problem = KerasModelProblem(kerasModel,hessian_block_size = hessian_block_size)
		if regularization is None:
			# If regularization is not passed in, default to zero Tikhonov
			self._regularization = L2Regularization(self._problem, 0.0)
		else:
			self._regularization = regularization

		self.settings = settings

		if optimizer is not None:
			if optimizer_parameters is None:
				self.set_optimizer(optimizer)
			else:
				self.set_optimizer(optimizer,parameters = optimizer_parameters)
		


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
	def set_optimizer(self):
		return self._set_optimizer
	
	

	@property
	def logger(self):
		return self._logger

	def _set_optimizer(self,optimizer,parameters = None):
		if parameters is None:
			self._optimizer = optimizer(self.problem, regularization = self.regularization,sess = None)
		else:
			self._optimizer = optimizer(self.problem, regularization = self.regularization,sess = None,parameters = parameters)
		# If larger Hessian spectrum is requested, reinitialize blocking for faster Hessian evaluations
		if 'hessian_low_rank' in self._optimizer.parameters.data.keys():
			if self.problem._hessian_block_size is None:
				self.problem._initialize_hessian_blocking(self.optimizer.parameters['hessian_low_rank'])
			elif self.problem._hessian_block_size < self.optimizer.parameters['hessian_low_rank']:
				self.problem._initialize_hessian_blocking(self.optimizer.parameters['hessian_low_rank'])




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

		if hasattr(self.problem, 'metric_dict'):
			for metric_name in self.problem.metric_dict.keys():
				logger[metric_name] = {}

		if self.settings['record_spectrum']:
			logger['full_train_eigenvalues'] = {}
			logger['train_eigenvalues'] = {}
			logger['val_eigenvalues'] = {}

		elif 'eigenvalues' in dir(self._optimizer):
			logger['train_eigenvalues'] = {}


		self._logger = logger

		os.makedirs(self.settings['problem_name']+'_logging/',exist_ok = True)
		os.makedirs(self.settings['problem_name']+'_best_weights/',exist_ok = True)
		

		# Set outname for logging file
		if self.settings['logger_outname'] is None:
			logger_outname = str(datetime.date.today())+'-dW='+str(self.problem.dimension)
		else:
			logger_outname = self.settings['logger_outname']
		self.logger_outname = logger_outname
	


	def _fit(self,data,options = None, w_0 = None):
		self.data = data
		if self.settings['verbose']:
			print(80*'#')
			print(('Size of configuration space:  '+str(self.problem.dimension)).center(80))
			print(('Size of training data: '+str(self.data.train_data_size)).center(80))
			print(('Approximate data cardinality needed: '\
				+str(int(float(self.problem.dimension)/self.problem.output_dimension	))).center(80))
			print(80*'#')

		with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=self.settings['intra_threads'],\
											inter_op_parallelism_threads=self.settings['inter_threads'])) as sess:
			# Re initialize data
			self.data.reset()
			# Initialize logging:
			self._initialize_logging()
			# Initialize the optimizer
			self._optimizer.set_sess(sess)
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
			val_dict = next(iter(self.data.validation))
			if self.problem.is_autoencoder:
				assert not hasattr(self.problem,'y_true')
			elif self.problem.is_gan:
				random_state_gan = np.random.RandomState(seed = 0)
				# Should the first dimension here agree with the size of the validation data?
				noise = random_state_gan.normal(size = (self.data.batch_size, self.problem.noise_dimension))
				val_dict[self.problem.noise] = noise

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
					assert not hasattr(self.problem,'y_true')
					noise = random_state_gan.normal(size = (self.data.batch_size, self.problem.noise_dimension))
					train_dict[self.problem.noise] = noise
					noise_hess = random_state_gan.normal(size = (self.data.hessian_batch_size, self.problem.noise_dimension))
					hess_dict[self.problem.noise] = noise_hess 

				try:
					self.problem.NN.reset_metrics()
				except:
					pass

				# metric_names = [metric.name for metric in self.problem.NN.metrics]
				# metric_evals = sess.run(self.problem.metrics_list,train_dict)

				# for name,evalu in zip(metric_names,metric_evals):
				# 	print('For metric',name,' we have: ',evalu)

				metric_names = list(self.problem.metric_dict.keys())
				metric_evals = sess.run(list(self.problem.metric_dict.values()),train_dict)

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
				try:
					self.problem.NN.reset_metrics()
				except:
					pass
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
				if hasattr(self.optimizer,'rank'):
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
				try:
					self.problem.NN.reset_metrics()
				except:
					pass
				if hasattr(self.problem,'accuracy'):
					if validate_this_iteration:
						validation_start = time.time()
						if hasattr(self.problem,'metric_dict'):
							metric_names = list(self.problem.metric_dict.keys())
							metric_values = sess.run(list(self.problem.metric_dict.values()),train_dict)
							for metric_name,metric_value in zip(metric_names,metric_values):
								self.logger[metric_name][iteration] = metric_value
						if hasattr(self.problem,'_variance_reduction'):
							if self.problem.has_derivative_loss:
								val_loss,	val_acc, val_h1_acc, val_var_red =\
									 sess.run([self.problem.loss,self.problem.accuracy,\
									 	self.problem.h1_accuracy,self.problem.variance_reduction],val_dict)
								self._logger['val_h1_acc'][iteration] = val_h1_acc
							else:
								val_loss,	val_acc, val_var_red =\
								 sess.run([self.problem.loss,self.problem.accuracy,self.problem.variance_reduction],val_dict)
							self._logger['val_variance_reduction'][iteration] = val_var_red
						else:
							if self.problem.has_derivative_loss:
								val_loss,	val_acc, val_h1_acc = sess.run([self.problem.loss,self.problem.accuracy,\
																			self.problem.h1_accuracy],val_dict)
								self._logger['val_h1_acc'][iteration] = val_h1_acc
							else:
								val_loss,	val_acc = sess.run([self.problem.loss,self.problem.accuracy],val_dict)
						self._logger['val_acc'][iteration] = val_acc
						self._logger['val_loss'][iteration] = val_loss
						max_val_acc = max(max_val_acc,val_acc)
						min_val_loss = min(min_val_loss,val_loss)
						validation_duration += time.time() - validation_start
					self._logger['max_val_acc'][iteration] = max_val_acc
				else:
					if validate_this_iteration:
						validation_start = time.time()
						val_loss = sess.run(self.problem.loss,val_dict)
						validation_duration += time.time() - validation_start
						min_val_loss = min(min_val_loss,val_loss)
						self._logger['val_loss'][iteration] = val_loss

				################################################################################
				# Save the best weights based on validation accuracy or loss
				if hasattr(self.problem,'accuracy') and val_acc == max_val_acc:
					weight_dictionary = {}
					for layer in self.problem._NN.layers:
						weight_dictionary[layer.name] = self.problem._NN.get_layer(layer.name).get_weights()
					self._best_weights = weight_dictionary
					if self.settings['save_weights']:
						# Save the weights individually, not in the logger

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
					break
				################################################################################
				# Actual optimization takes place here
				try:
					self.optimizer.minimize(train_dict,hessian_feed_dict=hess_dict)
				except:
					self.optimizer.minimize(train_dict)
				################################################################################
				# Recording the spectrum
				if not self.settings['record_spectrum'] and 'eigenvalues' in dir(self._optimizer):
					try:
						self._logger['train_eigenvalues'][iteration] = self.optimizer.eigenvalues
					except:
						pass
				elif self.settings['record_spectrum'] and iteration%self.settings['spec_frequency'] ==0:
					self._record_spectrum(iteration)
				with open(self.settings['problem_name']+'_logging/'+ self.logger_outname +'.pkl', 'wb+') as f:
					pickle.dump(self.logger, f, pickle.HIGHEST_PROTOCOL)
				with open(self.settings['problem_name']+'_best_weights/'+ self.logger_outname +'.pkl', 'wb+') as f:
					pickle.dump(self._best_weights, f, pickle.HIGHEST_PROTOCOL)
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
				if not (type(self.problem.x) is list):
					full_train_dict = {self.problem.x:train_data[self.problem.x]}
					full_val_dict = {self.problem.x:val_data[self.problem.x]}
				else:
					full_train_dict,full_val_dict = {},{}
					for input_key in self.problem.x:
						full_train_dict[input_key] = train_data[input_key]
						full_val_dict[input_key] = val_data[input_key]
			else:
				if not (type(self.problem.x) is list) and not (type(self.problem.y_true) is list):
					full_train_dict = {self.problem.x:train_data[self.problem.x],self.problem.y_true:train_data[self.problem.y_true]}
					full_val_dict = {self.problem.x:val_data[self.problem.x],self.problem.y_true:val_data[self.problem.y_true]}
				else:
					full_train_dict, full_val_dict = {}, {}
					if type(self.problem.x) is list:
						for input_key in self.problem.x:
							full_train_dict[input_key] = train_data[input_key]
							full_val_dict[input_key] = val_data[input_key]
					else:
						full_train_dict[self.problem.x] = train_data[self.problem.x]
						full_val_dict[self.problem.x] = val_data[self.problem.x]
					if type(self.problem.y_true) is list:
						for output_key in self.problem.y_true:
							full_train_dict[output_key] = train_data[output_key]
							full_val_dict[output_key] = val_data[output_key]
					else:
						full_train_dict[self.problem.y_true] = train_data[self.problem.y_true]
						full_val_dict[self.problem.y_true] = val_data[self.problem.y_true]




			d_full_train, U_full_train = low_rank_hessian(self.optimizer,full_train_dict,k_rank,p_oversample,verbose=True)
			self._logger['full_train_eigenvalues'][iteration] = d_full_train

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
					format_string += '{'+str(i)+':7} '
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
					format_string += '{'+str(i)+':5} '
				else:
					format_string += '{'+str(i)+':1.2e} '
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
