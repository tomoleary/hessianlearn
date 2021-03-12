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
import os
import sys
import numpy as np
import random
import math
import time
from abc import ABC, abstractmethod


class Data(ABC):
	"""
	This class implements the data iterator construct used in hessianlearn
	It takes data already prepartitioned into validation and training, or partitions 
	the data as such and then implements iterators that are used in the training loop
	"""
	def __init__(self,data, batch_size,validation_data = None,
					validation_data_size = None, max_epochs = np.inf,hessian_batch_size = -1,\
					variable_batch = False,batch_increment = None,
					shuffle = True,verbose = False,seed = 0):
		""" 
		The constructor for this class takes (x,y) data and partitions it into member iterators
			-data: a dictionary containing keys for names of data and values for the data itself
			-batch_size: the initial batch size to be used during training
			-validation_data: if none then validation and training data will be sampled and partitioned
				from data, otherwise data will be used for training and validation_data for validation
			-max_epochs: maximum numbers of times through the the data during iteration
			-hessian_batch_size: if positive then a Hessian data batch iterator will be instantiated,
				otherwise it will not
			-variable_batch: Boolean for variable batch size iterator
			-shuffle: Boolean for whether the data should be shuffled or not during partitioning
			-verbose: for printing
			-seed: the random seed used for shuffling.
		"""
		assert type(data) is dict, 'Data takes a dictionary in the constructor'
		data_keys = list(data.keys())

		n_data_objects = len(data.keys())

		data_cardinality = data[data_keys[0]].shape[0]


		# Make sure all data have the same cardinality (first index shape)
		for key in data_keys:
			assert data[key].shape[0] == data_cardinality, 'Cardinality mismatch within data'

		print('Data dimension agree')

		if validation_data is not None:
			assert data.keys() == validation_data.keys(), 'Validation data do not agree with train data'
			self._validation_data_size = validation_data[data_keys[0]].shape[0]
			# Make sure all validation data have the same cardinality
			for key in data_keys:
				assert validation_data[key].shape[0] == self._validation_data_size, 'Cardinality mismatch within validation data'
			# Make sure that all of the validation data shapes agree with training data
			for key in data_keys:
				assert data[key][0].shape == validation_data[key][0].shape, 'Shape mismatch between train and validation'

		else:
			assert validation_data_size is not None
			self._validation_data_size = validation_data_size

		self._batch_size = batch_size

		self._train_data_size = None

		self._max_epochs = max_epochs
		self.verbose = verbose
		self._variable_batch = variable_batch
		self._hessian_batch_size = hessian_batch_size
		self._batch_increment = batch_increment
		self._shuffle = shuffle


		if validation_data is None:
			validation_data_cardinality = 0
		else:
			validation_data_cardinality = validation_data[data_keys[0]].shape[0]
		self._total_data_cardinality = data_cardinality + validation_data_cardinality

		# Partition data and instantiate iterables for training and validation data
		self._partition(data,validation_data = validation_data,seed = seed)

	@property
	def validation(self):
		return self._validation

	@property
	def train(self):
		return self._train

	@property
	def hess_train(self):
		return self._hess_train

	@property
	def batch_factor(self):
		return self._batch_factor

	@property
	def validation_data_size(self):
		return self._validation_data_size
	
	@property
	def train_data_size(self):
		return self._train_data_size

	@property
	def batch_size(self):
		return self._batch_size

	@property
	def hessian_batch_size(self):
		return self._hessian_batch_size
	
	
	

	def _partition(self,data,validation_data = None, seed = 0):
		"""
		This method partitions the data, if validation_data is none
		the method will shuffle and partition data when the boolean
		self._shuffle is True, otherwise it will partition the data
		as it is passed in.
			-data: the corpus of data, if validation_data is not None, then data
			is just the training data
			-validation_data: Optional, pre determined validationing data partition
			-seed: random seed used for shuffling
		This method instantiates the self.train and self.validation data iterators
		"""
		if validation_data is not None:
			# Then the partition is giving implicitly by the user 
			_validation_data = DictData(validation_data)
			_train_data = DictData(data)
		else:
			# In this case we parititon from the entire dataset
			print('self._total_data_cardinality = ',self._total_data_cardinality)
			indices = range(self._total_data_cardinality)
			validation_indices,train_indices 	= np.split(indices,[self._validation_data_size])

			if self.verbose:
				print('Shuffling data')
				t0 = time.time()
			if self._shuffle:
				for key in data:
					random.Random(seed).shuffle(data[key])

			if self.verbose:
				duration = time.time() - t0
				print('Shuffling took ', duration,' s')
				print('Partitioning data')
				t0 = time.time()

			train_dict = {}
			validation_dict = {}
			for key in data:
				train_dict[key] = data[key][train_indices]
				validation_dict[key] = data[key][validation_indices]

			_validation_data = DictData(validation_dict)
			_train_data = DictData(train_dict)
			if self.verbose:
				duration = time.time() - t0
				print('Partitioning took ', duration,' s')			
				print('Instantiating data iterables')
				t0 = time.time()		

		# Instantiate the validation data static iterator
		self._validation = StaticIterator(_validation_data)
		# Instantiate the training data iterator
		if self._variable_batch:
			self._train = VariableBatchIterator(_train_data,self._batch_size,\
							batch_increment = self._batch_increment,max_epochs = self._max_epochs,\
							verbose = self.verbose)
			if self._hessian_batch_size > 0:
				self._hess_train = VariableBatchIterator(_train_data,self._hessian_batch_size,\
							batch_increment = self._hessian_batch_size,max_epochs = np.inf)
			else:
				self._hess_train = None
		else:
			self._train = BatchIterator(_train_data,self._batch_size,max_epochs = self._max_epochs,\
										verbose = self.verbose)
			if self._hessian_batch_size > 0:
				self._hess_train = BatchIterator(_train_data,self._hessian_batch_size,\
													max_epochs = np.inf,verbose = self.verbose)
			else:
				self._hess_train = None

		if self.verbose:
			duration = time.time() - t0
			print('Instantiating iterables took ', duration,' s')

		self._train_data_size = _train_data.size

		self._batch_factor = [float(self._batch_size)/float(self._train_data_size),\
					 float(self._hessian_batch_size)/float(self._train_data_size)]		


	def reset(self):
		self._train._index = 0
		self._validation._index = 0



class BatchIterator(object):
	"""
	This class implements a batch iterator object.
	"""
	def __init__(self,data,batch_size,max_epochs = np.inf, seed = 0,verbose = False):
		"""
		The constructor for this class takes pre-partitioned data and instantiates a 
		batch data iterator
			-data: the pre partitioned data
			-batch_size: the fixed batch size for each iteration
			-max_epochs: The maximum number of times to go through the data
			-seed: the seed for shuffling during iteration
			-verbose: Boolean for printing
		"""
		self._data = data
		self._batch_size = batch_size
		self._max_epochs = max_epochs
		self._index = 0
		self._epoch = 0
		self.verbose = verbose

	@property
	def index(self):
		return self._index
	

	def __iter__(self):
		"""
		Returns self (the iterator object)
		"""
		return self
		

	def __next__(self):
		"""
		This method defines the shuffling scheme for the iterator
		"""
		next_dictionary = {}
		if self._epoch >= self._max_epochs:
			print('Maximum epochs reached')
			raise StopIteration
		if self._index + self._batch_size > self._data.size:
			prefix_dictionary = {}
			amount_left = self._index + self._batch_size - self._data.size
			for key in self._data.keys():
				prefix_dictionary[key] = self._data[key][self._index:]
			if self.verbose:
				print('Reshuffling')
			# Seeding on the epoch number for now
			for key in self._data.keys():
				random.Random(self._epoch).shuffle(self._data[key])
			self._epoch +=1
			self._index = amount_left
			suffix_dictionary = {}
			for key in self._data.keys():
				suffix_dictionary[key] = self._data[key][:self._index]
			next_data = {}
			for key in self._data.keys():
				next_data[key] = np.concatenate((prefix_dictionary[key],suffix_dictionary[key]))
		else:
			next_data = {}
			for key in self._data.keys():
				next_data[key] = self._data[key][self._index:self._index+self._batch_size]
			self._index += self._batch_size
		return next_data


# class VariableBatchIterator(object):
# 	"""
# 	This class implements a variable batch iterator object
# 	"""
# 	def __init__(self,data,batch_size,max_epochs = 1000,batch_increment = None,seed = 0,verbose = False):
# 		"""
# 		The constructor for this class takes pre-partitioned data and instantiates a 
# 		batch data iterator
# 			-data: the pre partitioned data
# 			-batch_size: the fixed batch size for each iteration
# 			-max_epochs: The maximum number of times to go through the data
# 			-seed: the seed for shuffling during iteration
# 			-verbose: Boolean for printing
# 		"""
# 		self._data = data
# 		self._batch_size = batch_size
# 		self._max_epochs = max_epochs
# 		self._index = 0
# 		self._epoch = 0
# 		self.verbose = verbose
# 		if batch_increment is None:
# 			self._batch_increment = batch_size
# 		else:
# 			self._batch_increment = batch_increment

# 	@property
# 	def index(self):
# 		return self._index
	

# 	def __iter__(self):
# 		"""
# 		Returns self (the iterator object)
# 		"""
# 		return self
		

# 	def __next__(self):
# 		"""
# 		This method defines the shuffling scheme for the iterator
# 		"""
# 		if self._epoch >= self._max_epochs:
# 			print('Maximum epochs reached')
# 			raise StopIteration
# 		if self._index + self._batch_size > self._data.size:
# 			amount_left = self._index + self._batch_size - self._data.size
# 			pref_x = self._data.x[self._index:]
# 			pref_y = self._data.y[self._index:]
# 			if self.verbose:
# 				print('Reshuffling')
# 			# Seeding on the epoch number for now
# 			random.Random(self._epoch).shuffle(self._data.x)
# 			random.Random(self._epoch).shuffle(self._data.y)
# 			self._epoch +=1
# 			self._index = amount_left
# 			suff_x = self._data.x[:self._index]
# 			suff_y = self._data.y[:self._index]
# 			next_x = np.concatenate((pref_x,suff_x))
# 			next_y = np.concatenate((pref_y,suff_y))
# 			if self._batch_size + self._batch_increment < self._data.size:
# 				self._batch_size += self._batch_increment
# 			else:
# 				self._batch_size = self._data.size
# 		else:
# 			next_x = self._data.x[self._index:self._index+self._batch_size]
# 			next_y = self._data.y[self._index:self._index+self._batch_size]
# 			self._index += self._batch_size
# 		return next_x, next_y


class StaticIterator(object):
	"""
	This class implements a static data iterator object
	"""
	def __init__(self,data):
		"""
		The constructor for this class just takes the data (xyData object)
		"""
		self._data = data
		self._index = 0

	@property
	def index(self):
		return self._index
	

	def __iter__(self):
		"""
		Returns self (the iterator object)
		"""
		return self

	def __next__(self):
		"""
		This method defines the shuffling scheme for the iterator
		Which just returns all of the data since its a static iterator
		"""
		# if self.index > 0:
		# 	raise StopIteration
		self._index += 1
		return self._data


class DictData (object):
	"""
	This class implements a simple xy data pair object
	"""
	def __init__(self,data):
		"""
		The constructor for this class takes a list of xy data
			-data: List of [x,y] data
		"""
		self.data = data
		self.size = len(self.data[list(self.data.keys())[0]])

	def __getitem__(self,key):
		return self.data[key]

	def __setitem__(self,key,item):
		self.data[key] = item

	def keys(self):
		return self.data.keys()




