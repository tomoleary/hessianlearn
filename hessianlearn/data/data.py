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
# along with stein variational inference methods class project.  If not, see <http://www.gnu.org/licenses/>.
#
# Author: Tom O'Leary-Roseberry
# Contact: tom.olearyroseberry@utexas.edu

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
import random
import math
import time
from abc import ABC, abstractmethod


class Data(ABC):
	# Must pass in data = [x,y] where x and y are numpy arrays of the same length (but possibly different shapes)

	def __init__(self,data,train_data_size, batch_size,validation_data_size = 500,\
					test_data_size = 500,total_population_size = None,max_epochs = 1000,\
					variable_batch = False,hessian_batch_size = -1,batch_increment = None,
					shuffle = True,verbose = False):
		self._train_data_size = train_data_size
		self._batch_size = batch_size
		self._validation_data_size = validation_data_size
		self._test_data_size = test_data_size
		self._max_epochs = max_epochs
		self.verbose = verbose
		self._variable_batch = variable_batch
		self._hessian_batch_size = hessian_batch_size
		self._batch_increment = batch_increment
		self._shuffle = shuffle

		self._input_shape = [None] + list(data[0][0].shape)
		self._output_shape = [None] + list(data[1][0].shape)
		if len(self._input_shape) == 3:
			self._input_shape += [1]
		if len(self._output_shape) == 3:
			self._output_shape += [1]
		data_size = len(data[0])
		if total_population_size is None:
			self._total_population_size = data_size
		else:
			self._total_population_size = min(data_size,total_population_size)
		# Partition data and instantiate iterables for train, val test etc.
		self._partition(data)

	@property
	def test(self):
		return self._test

	@property
	def validation(self):
		return self._validation

	@property
	def train(self):
		return self._train

	@property
	def hess_train(self):
		return self._hess_train

	def _load_data(self):
		# Collect Data in a reproducible way (same ordering)
		# Put in numpy array
		# return two numpy arrays x (input), y (output)
		raise NotImplementedError('Child class must implement method _load_data')

	def _partition(self,data,seed = 0):
		# Shuffle and parition, instantiate self.train, self.validation ,self.test

		assert self._train_data_size + self._test_data_size + self._validation_data_size <= self._total_population_size,\
													 'sizes for train/validation/test data sets exceed available data:'+\
													 str(self._train_data_size + self._test_data_size + self._validation_data_size)+'<'+\
													 str(self._total_population_size)
		indices = range(self._total_population_size)
		test_indices,\
		validation_indices,\
		train_indices 	= np.split(indices,[self._test_data_size,self._test_data_size + self._validation_data_size])
		if self.verbose:
			print('Shuffling data')
			t0 = time.time()
		all_x,all_y = data
		if self._shuffle:
			random.Random(seed).shuffle(all_x)
			random.Random(seed).shuffle(all_y)
		if self.verbose:
			duration = time.time() - t0
			print('Shuffling took ', duration,' s')
			print('Partitioning data')
			t0 = time.time()
		test_data = xyData([all_x[test_indices],all_y[test_indices]])
		validation_data = xyData([all_x[validation_indices],all_y[validation_indices]])
		train_data = xyData([all_x[train_indices],all_y[train_indices]])
		if self.verbose:
			duration = time.time() - t0
			print('Partitioning took ', duration,' s')			
			print('Instantiating data iterables')
			t0 = time.time()
		self._test = StaticIterator(test_data)
		self._validation = StaticIterator(validation_data)
		if self._variable_batch:
			self._train = VariableBatchIterator(train_data,self._batch_size,\
							batch_increment = self._batch_increment,max_epochs = self._max_epochs,\
							verbose = self.verbose)
			if self._hessian_batch_size > 0:
				self._hess_train = VariableBatchIterator(train_data,self._hessian_batch_size,\
							batch_increment = self._hessian_batch_size,max_epochs = np.inf)
			else:
				self._hess_train = None
		else:
			self._train = BatchIterator(train_data,self._batch_size,max_epochs = self._max_epochs,\
										verbose = self.verbose)
			if self._hessian_batch_size > 0:
				self._hess_train = BatchIterator(train_data,self._hessian_batch_size,\
													max_epochs = np.inf,verbose = self.verbose)
			else:
				self._hess_train = None

		if self.verbose:
			duration = time.time() - t0
			print('Instantiating iterables took ', duration,' s')		



class BatchIterator(object):
	def __init__(self,data,batch_size,max_epochs = 1000, seed = 0,verbose = False):
		self._data = data
		self._batch_size = batch_size
		self._max_epochs = max_epochs
		self._index = 0
		self._epoch = 0
		self.verbose = verbose


	def __iter__(self):
		return self
		

	def __next__(self):
		if self._epoch >= self._max_epochs:
			print('Maximum epochs reached')
			raise StopIteration
		if self._index + self._batch_size > self._data.size:
			amount_left = self._index + self._batch_size - self._data.size
			pref_x = self._data.x[self._index:]
			pref_y = self._data.y[self._index:]
			if self.verbose:
				print('Reshuffling')
			# Seeding on the epoch number for now
			random.Random(self._epoch).shuffle(self._data.x)
			random.Random(self._epoch).shuffle(self._data.y)
			self._epoch +=1
			self._index = amount_left
			suff_x = self._data.x[:self._index]
			suff_y = self._data.y[:self._index]
			next_x = np.concatenate((pref_x,suff_x))
			next_y = np.concatenate((pref_y,suff_y))
		else:
			next_x = self._data.x[self._index:self._index+self._batch_size]
			next_y = self._data.y[self._index:self._index+self._batch_size]
			self._index += self._batch_size
		return next_x, next_y

class VariableBatchIterator(object):
	def __init__(self,data,batch_size,max_epochs = 1000,batch_increment = None,seed = 0,verbose = False):
		self._data = data
		self._batch_size = batch_size
		self._max_epochs = max_epochs
		self._index = 0
		self._epoch = 0
		self.verbose = verbose
		if batch_increment is None:
			self._batch_increment = batch_size
		else:
			self._batch_increment = batch_increment


	def __iter__(self):
		return self
		

	def __next__(self):
		if self._epoch >= self._max_epochs:
			print('Maximum epochs reached')
			raise StopIteration
		if self._index + self._batch_size > self._data.size:
			amount_left = self._index + self._batch_size - self._data.size
			pref_x = self._data.x[self._index:]
			pref_y = self._data.y[self._index:]
			if self.verbose:
				print('Reshuffling')
			# Seeding on the epoch number for now
			random.Random(self._epoch).shuffle(self._data.x)
			random.Random(self._epoch).shuffle(self._data.y)
			self._epoch +=1
			self._index = amount_left
			suff_x = self._data.x[:self._index]
			suff_y = self._data.y[:self._index]
			next_x = np.concatenate((pref_x,suff_x))
			next_y = np.concatenate((pref_y,suff_y))
			if self._batch_size + self._batch_increment < self._data.size:
				self._batch_size += self._batch_increment
			else:
				self._batch_size = self._data.size
		else:
			next_x = self._data.x[self._index:self._index+self._batch_size]
			next_y = self._data.y[self._index:self._index+self._batch_size]
			self._index += self._batch_size
		return next_x, next_y


class StaticIterator(object):
	def __init__(self,data):
		self.data = data
		self.index = 0

	def __iter__(self):
		return self

	def __next__(self):
		if self.index > 0:
			raise StopIteration
		next_x = self.data.x
		next_y = self.data.y
		self.index += 1
		return next_x,next_y



class xyData (object):
	def __init__(self,data):
		self.x = data[0]
		self.y = data[1]
		self.size = len(self.x)



class DataIterator(object):
	def __init__(self):
		pass

	def __iter__(self):
		return self

	def __next__(self):
		pass


