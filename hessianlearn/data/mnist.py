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

from ..data.data import *

import math
import time

# from statsmodels import robust

def dir_check(dir):
	try:
		os.stat(dir)
	except:
		os.mkdir(dir)

def load_mnist():
	try: 
		# read from file
		images = np.load('mnist_all_images.npy')
		labels = np.load('mnist_all_labels.npy')
	except:
		# write to file
		print(80*'#')
		print('Did not load'.center(80))
		print(80*'#')
		from tensorflow.examples.tutorials.mnist import input_data
		# Can I access mnist data from tf v2 using compat??
		mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

		all_data = mnist.train.next_batch(60000)
		images, labels = all_data

		images = np.array(images)
		for i in range(2):
			images = np.expand_dims(images,axis = -1)
		np.save('mnist_all_images.npy',images)
		labels = np.array(labels)
		np.save('mnist_all_labels.npy',labels)
	return [images,labels]





