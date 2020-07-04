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

import struct

import gzip

# from statsmodels import robust

def dir_check(dir):
	try:
		os.stat(dir)
	except:
		os.mkdir(dir)

def read_idx(filename):
	with gzip.open(filename, 'rb') as f:
		zero, data_type, dims = struct.unpack('>HBB', f.read(4))
		shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
		return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def load_mnist(normalize = True):
	try: 
		# read from file
		if normalize:
			images = np.load('mnist_all_normalized_images.npy')
		else:
			images = np.load('mnist_all_images.npy')
		labels = np.load('mnist_all_labels.npy')
	except:
		# write to file
		print(80*'#')
		print('Did not load locally'.center(80))
		print(80*'#')

		tarballs = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',\
						't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
		for tarball in tarballs:
			try:
				os.stat(tarball)
			except:
				print('Downloading '+tarball+' from source, and saving to disk.')

				import urllib.request
				urllib.request.urlretrieve("http://yann.lecun.com/exdb/mnist/"+tarball,\
									  tarball)
		print(80*'#')

		test_images = read_idx('t10k-images-idx3-ubyte.gz')
		test_labels = read_idx('t10k-labels-idx1-ubyte.gz')
		train_images = read_idx('train-images-idx3-ubyte.gz')
		train_labels = read_idx('train-labels-idx1-ubyte.gz')

		images = np.concatenate((test_images,train_images))
		images = images.astype(np.float64)
		if normalize:
			images *= 1./(np.max(images))
		labels_temp = np.concatenate((test_labels,train_labels))

		labels = np.zeros((labels_temp.shape[0],10))
		for i,label in enumerate(labels_temp):
			labels[i,label] = 1


		for i in range(1):
			images = np.expand_dims(images,axis = -1)
		if normalize:
			np.save('mnist_all_normalized_images.npy',images)
		else:
			np.save('mnist_all_images.npy',images)
		np.save('mnist_all_labels.npy',labels)
	return [images,labels]





