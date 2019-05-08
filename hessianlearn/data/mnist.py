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





