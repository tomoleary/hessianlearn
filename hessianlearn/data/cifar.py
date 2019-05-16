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
from scipy import signal
import random



import math
import time

# from statsmodels import robust

def dir_check(dir):
	try:
		os.stat(dir)
	except:
		os.mkdir(dir)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        try:
            dict = pickle.load(fo, encoding='bytes')
        except:
            dict = pickle.load(fo)
    return dict

def label_to_vector(input_list):
    labels = []
    for element in input_list:
        labels.append(signal.unit_impulse(10,element))
    return np.array(labels)


def load_cifar10():
	try: 
		# read from file
		images = np.load('cifar10_all_images.npy')
		labels = np.load('cifar10_all_labels.npy')
		return [images, labels]
	except:
		# write to file
		print(80*'#')
		print('Did not load locally.')
		print(80*'#')
		try:
			os.stat("cifar-10-python.tar.gz")
		except:
			print('Downloading from source, and saving to disk.')
			print(80*'#')
			import urllib.request
			urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "cifar-10-python.tar.gz")
		try:
			os.stat('cifar-10-batches-py/')
		except:
			import subprocess
			subprocess.run(['tar','zxvf',"cifar-10-python.tar.gz"])
		folder_name = 'cifar-10-batches-py/'
		data_raw = {}
		for file in ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']:
		    data_raw[file] = unpickle(folder_name+file)

		images = np.empty(shape = (60000,32,32,3))
		labels = np.empty(shape = (60000,10))
		index = 0
		increment = 10000
		for file in ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']:
		    images[index:index+increment,:] = data_raw[file][b'data'].reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
		    labels[index:index+increment,:] = label_to_vector(data_raw[file][b'labels'])
		    index += increment

		assert(labels.shape[0]==images.shape[0])

		print(labels.shape)

		images = np.array(images)
		np.save('cifar10_all_images.npy',images)
		labels = np.array(labels)
		np.save('cifar10_all_labels.npy',labels)

		return [images,labels]



	# def view_random_pair(self):
	# 	try:
	# 	    labelkey = ['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
	# 	    i = np.random.choice(range(60000))
	# 	    index = self.all_data[1][i]
	# 	    label = labelkey[index]
	# 	    import matplotlib.pyplot as plt
	# 	    fig, ax = plt.subplots(figsize = (3,3))
	# 	    ax.set_title(str(label))
	# 	    data = self.all_data[0][i,:,:,:].astype(np.uint8)
	# 	    ax.imshow(data)
	# 	    plt.show()
	# 	except:
	# 		pass
	

