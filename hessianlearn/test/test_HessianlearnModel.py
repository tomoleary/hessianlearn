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

import unittest 
import numpy as np
import tensorflow as tf
if int(tf.__version__[0]) > 1:
	import tensorflow.compat.v1 as tf
	tf.disable_v2_behavior()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["KMP_WARNINGS"] = "FALSE" 



import sys
sys.path.append('../../')
from hessianlearn import (HessianlearnModel, HessianlearnModelSettings,
								ClassificationProblem,Data, L2Regularization)

tf.set_random_seed(0)

class TestHessianlearnModel(unittest.TestCase):

	def test_all_optimizers(self):
		# Instantiate data
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
		# Normalize the data
		x_train = x_train.astype('float32') / 255.
		x_test = x_test.astype('float32') / 255.
		def one_hot_vectors(labels_temp):
			labels = np.zeros((labels_temp.shape[0],10))
			for i,label in enumerate(labels_temp):
				labels[i,label] = 1
			return labels
		y_train = one_hot_vectors(y_train)
		y_test = one_hot_vectors(y_test)
		# Instantiate neural network
		classifier = tf.keras.Sequential([
		    tf.keras.layers.Flatten(input_shape=(28, 28)),
		    tf.keras.layers.Dense(128, activation='relu'),
		    tf.keras.layers.Dense(10)
		])
		# Instantiate the problem, regularization.
		problem = ClassificationProblem(classifier,loss_type = 'least_squares',dtype=tf.float32)
		regularization = L2Regularization(problem,gamma =0.001)
		# Instante the data object
		train_dict = {problem.x:x_train, problem.y_true:y_train}
		validation_dict = {problem.x:x_test, problem.y_true:y_test}
		data = Data(train_dict,256,validation_data = validation_dict,hessian_batch_size = 32)
		# Instantiate the model object
		HLModelSettings = HessianlearnModelSettings()
		HLModelSettings['max_sweeps'] = 1.
		HLModel = HessianlearnModel(problem,regularization,data,settings = HLModelSettings)

		for optimizer in ['lrsfn','adam','gd','incg','sgd']:
			HLModel.settings['optimizer'] = optimizer
			HLModel.fit()
			first_loss = HLModel.logger['loss_train'][0]
			last_iteration = max(HLModel.logger['loss_train'].keys())
			last_loss = HLModel.logger['loss_train'][last_iteration]
			print('first loss = ',first_loss)
			print('last_loss = ',last_loss)


		assert True

if __name__ == '__main__':
    unittest.main()