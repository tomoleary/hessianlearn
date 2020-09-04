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
import time

# import sys, os
# sys.path.append( os.environ.get('HESSIANLEARN_PATH', "../../"))
# from hessianlearn import *

from ..utilities.parameterList import ParameterList

# from ..algorithms import *

from ..algorithms.adam import Adam
from ..algorithms.gradientDescent import GradientDescent
# from ..algorithms.cgSolver import CGSolver
# from ..algorithms.inexactNewtonCG import InexactNewtonCG
# from ..algorithms.gmresSolver import GMRESSolver 
# from ..algorithms.inexactNewtonGMRES import InexactNewtonGMRES
# from ..algorithms.minresSolver import MINRESSolver
# from ..algorithms.inexactNewtonMINRES import InexactNewtonMINRES
# from ..algorithms.randomizedEigensolver import *
from ..problem.regularization import L2Regularization
from ..algorithms.lowRankSaddleFreeNewton import LowRankSaddleFreeNewton



def HessianlearnModelSettings(settings = {}):
	settings['verbose']         			= [True, "Boolean for printing"]

	settings['intra_threads']         		= [2, "Setting for intra op parallelism"]
	settings['inter_threads']         		= [2, "Setting for inter op parallelism"]

	# Optimizer settings
	settings['optimizer']                	= ['lrsfn', "String to denote choice of optimizer"]
	settings['alpha']                		= [5e-2, "Initial steplength, or learning rate"]
	settings['sfn_lr']						= [10, "Low rank to be used for LRSFN / SFN"]
	settings['fixed_step']					= [False, " True means steps of length alpha will be taken at each iteration"]
	settings['max_backtrack']				= [10, "Maximum number of backtracking iterations for each line search"]


	settings['max_sweeps']					= [10,"Maximum number of times through the data (measured in epoch equivalents"]



	return ParameterList(settings)


class HessianlearnModel(ABC):
	def __init__(self,problem,regularization,data,settings = HessianlearnModelSettings()):

		self._problem = problem
		self._regularization = regularization
		self._data = data

		self.settings = settings

		if self.settings['verbose']:
			print(80*'#')
			print(('Size of configuration space: '+str(self.problem.dimension)).center(80))
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
	


	def _fit(self,options = None):
		# Consider doing scope managed sess
		# For now I will use the sess as a member variable
		# self._sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=self.settings['intra_threads'],\
		# 									inter_op_parallelism_threads=self.settings['inter_threads']))
		with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=self.settings['intra_threads'],\
											inter_op_parallelism_threads=self.settings['inter_threads'])) as sess:
			self._initialize_optimizer(sess)

			# After optimizer is instantiated, we call the global variables initializer
			sess.run(tf.global_variables_initializer())

			# random_state = np.random.RandomState(seed = 0)
			# w_0 = random_state.randn(problem.dimension)
			# sess.run(problem._assignment_ops,feed_dict = {problem._assignment_placeholder:w_0})

			if self.settings['verbose']:
				print(80*'#')
				# First print
				print('{0:8} {1:11} {2:11} {3:11} {4:11} {5:11} {6:11} {7:11}'.format(\
										'Sweeps'.center(8),'Loss'.center(8),'acc train'.center(8),'||g||'.center(8),\
															'Loss_test'.center(8), 'acc test'.center(8),'max test'.center(8), 'alpha'.center(8)))

			x_test, y_test = next(iter(self.data.test))

			test_dict = {self.problem.x: x_test}

			# Iteration Loop
			max_sweeps = self.settings['max_sweeps']
			train_data = iter(self.data.train)
			x_batch,y_batch = next(train_data)
			sweeps = 0
			min_test_loss = np.inf
			max_test_acc = -np.inf
			t0 = time.time()
			for i, (data_g,data_H) in enumerate(zip(self.data.train,self.data.hess_train)):
				x_batch,y_batch = data_g
				x_hess, y_hess = data_H

				feed_dict = {self.problem.x: x_batch}
				hess_dict = {self.problem.x: x_hess}

				norm_g, loss_train, accuracy_train = sess.run([self.problem.norm_g,self.problem.loss,self.problem.accuracy],feed_dict)
				# logger['||g||'][i] = norm_g
				# logger['loss'][i] = loss_train
				# logger['accuracy_train'][i] = accuracy_train
				# logger['time'][i] = time.time() - t0
				
				# logger['sweeps'][i] = sweeps
				loss_test,	accuracy_test = sess.run([self.problem.loss,self.problem.accuracy],test_dict)
				# logger['accuracy_test'][i] = accuracy_test
				# logger['loss_test'][i] = loss_test
				min_test_loss = min(min_test_loss,loss_test)
				max_test_acc = max(max_test_acc,accuracy_test)
				if accuracy_test == max_test_acc:
					self._best_weights = sess.run(self.problem._w)
				# 	if len(logger['best_weight']) > 2:
				# 		logger['best_weight'].pop(0)
				# 	acc_weight_tuple = (accuracy_test,accuracy_train,sess.run(problem._flat_w))
				# 	logger['best_weight'].append(acc_weight_tuple) 

				sweeps = np.dot(self.data.batch_factor,self.optimizer.sweeps)
				if self.settings['verbose'] and i % 1 == 0:
					# Print once each epoch
					try:
						print(' {0:^8.2f} {1:1.4e} {2:.3%} {3:1.4e} {4:1.4e} {5:.3%} {6:.3%} {7:1.4e}'.format(\
							sweeps, loss_train,accuracy_train,norm_g,loss_test,accuracy_test,max_test_acc,self.optimizer.alpha))
					except:
						print(' {0:^8.2f} {1:1.4e} {2:.3%} {3:1.4e} {4:1.4e} {5:.3%} {6:.3%} {7:11}'.format(\
							sweeps, loss_train,accuracy_train,norm_g,loss_test,accuracy_test,max_test_acc,self.optimizer.alpha))
				try:
					self.optimizer.minimize(feed_dict,hessian_feed_dict=hess_dict)
				except:
					self.optimizer.minimize(feed_dict)

				if sweeps > max_sweeps:
					break
		# The weights need to be manually set once the session scope is closed.
		try:
			self._problem._NN.set_weights(self._best_weights)
		except:
			pass

	# try:
	# 	os.makedirs('results/')
	# except:
	# 	pass
	# with open('results/'+ outname +'.pkl', 'wb+') as f:
	# 	pickle.dump(logger, f, pickle.HIGHEST_PROTOCOL)


		

	def _initialize_optimizer(self, sess):
		settings = self.settings
		# assert self.sess is not None
		if settings['optimizer'] == 'adam':
			print(('Using Adam optimizer').center(80))
			print(('Batch size = '+str(self.data._batch_size)).center(80))
			optimizer = Adam(self.problem,self.regularization,sess)
			optimizer.parameters['alpha'] = settings['alpha']
			optimizer.alpha = settings['alpha']

		elif settings['optimizer'] == 'gd':
			print('Using gradient descent optimizer with line search'.center(80))
			print(('Batch size = '+str(self.data._batch_size)).center(80))
			optimizer = GradientDescent(self.problem,self.regularization,sess)
			optimizer.parameters['globalization'] = 'line_search'
			optimizer.parameters['max_backtracking_iter'] = 8
		elif settings['optimizer'] == 'incg':
			if not settings['fixed_step']:
				print('Using inexact Newton CG optimizer with line search'.center(80))
				print(('Batch size = '+str(self.data._batch_size)).center(80))
				print(('Hessian batch size = '+str(self.data._hessian_batch_size)).center(80))
				optimizer = InexactNewtonCG(self.problem,self.regularization,sess)
				optimizer.parameters['globalization'] = 'line_search'
				optimizer.parameters['max_backtracking_iter'] = settings['max_backtrack']
			else:
				print('Using inexact Newton CG optimizer with fixed step'.center(80))
				print(('Batch size = '+str(self.data._batch_size)).center(80))
				print(('Hessian batch size = '+str(self.data._hessian_batch_size)).center(80))
				optimizer = InexactNewtonCG(self.problem,self.regularization,sess)
				optimizer.parameters['globalization'] = 'None'
				optimizer.alpha = settings['alpha']
		elif settings['optimizer'] == 'lrsfn':
			if not settings['fixed_step']:
				print('Using low rank SFN optimizer with line search'.center(80))
				print(('Batch size = '+str(self.data._batch_size)).center(80))
				print(('Hessian batch size = '+str(self.data._hessian_batch_size)).center(80))
				print(('Hessian low rank = '+str(settings['sfn_lr'])).center(80))
				optimizer = LowRankSaddleFreeNewton(self.problem,self.regularization,sess)
				optimizer.parameters['globalization'] = 'line_search'
				optimizer.parameters['max_backtracking_iter'] = settings['max_backtrack']
				optimizer.parameters['hessian_low_rank'] = settings['sfn_lr']
			else:
				print('Using low rank SFN optimizer with fixed step'.center(80))
				print(('Batch size = '+str(self.data._batch_size)).center(80))
				print(('Hessian batch size = '+str(self.data._hessian_batch_size)).center(80))
				print(('Hessian low rank = '+str(settings['sfn_lr'])).center(80))
				optimizer = LowRankSaddleFreeNewton(self.problem,self.regularization,sess)
				optimizer.parameters['hessian_low_rank'] = settings['sfn_lr']
				optimizer.parameters['alpha'] = settings['alpha']
				optimizer.alpha = settings['alpha']
		elif settings['optimizer'] == 'sgd':
			print(('Using stochastic gradient descent optimizer').center(80))
			print(('Batch size = '+str(self._data._batch_size)).center(80))
			optimizer = GradientDescent(self.problem,self.regularization,sess)
			optimizer.parameters['alpha'] = settings['alpha']
			optimizer.alpha = settings['alpha']
		else:
			raise
		self._optimizer = optimizer



		def _initialize_logging(self):
			pass












