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

################################################################################
# Uses some code from https://blog.keras.io/building-autoencoders-in-keras.html
################################################################################

import numpy as np
import os
import tensorflow as tf
import time
# if int(tf.__version__[0]) > 1:
# 	import tensorflow.compat.v1 as tf
# 	tf.disable_v2_behavior()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["KMP_WARNINGS"] = "FALSE" 
import sys
sys.path.append( os.environ.get('HESSIANLEARN_PATH', "../../"))
from hessianlearn import *

tf.set_random_seed(0)

settings = {}
# Set run specifications
# Data specs
settings['batch_size'] = 100
settings['hess_batch_size'] = 10


################################################################################
# Instantiate data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Normalize the data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
# Reshape the data
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Instante the data object
data = Data([x_train,y_train],settings['batch_size'],test_data = [x_test,y_test],hessian_batch_size = settings['hess_batch_size'])

# settings['input_shape'] = data._input_shape
# settings['output_shape'] = data._output_shape


################################################################################
# Create the neural network in keras

encoding_dim = 32  
input_img = tf.keras.layers.Input(shape=(784,))
encoded = tf.keras.layers.Dense(encoding_dim, activation='softplus')(input_img)
decoded = tf.keras.layers.Dense(784, activation='sigmoid')(encoded)
autoencoder = tf.keras.models.Model(input_img, decoded)


################################################################################
# Instantiate the problem, regularization.

problem = AutoencoderProblem(autoencoder,dtype=tf.float32)

settings['tikhonov_gamma'] = 0.0

regularization = L2Regularization(problem,gamma = settings['tikhonov_gamma'])


################################################################################
# Instantiate the model object
HLModelSettings = HessianlearnModelSettings()

HLModelSettings['optimizer'] = 'lrsfn'
HLModelSettings['alpha'] = 1e-2
HLModelSettings['fixed_step'] = False
HLModelSettings['sfn_lr'] = 20
HLModelSettings['max_backtrack'] = 16
HLModelSettings['max_sweeps'] = 50

HLModelSettings['problem_name'] = 'mnist_ae'
HLModelSettings['record_spectrum'] = False
HLModelSettings['rq_data_size'] = 100


HLModel = HessianlearnModel(problem,regularization,data,settings = HLModelSettings)

HLModel.fit()

################################################################################
# Postprocessing with the trained autoencoder

encoder = tf.keras.models.Model(input_img, encoded)

encoded_input = tf.keras.layers.Input(shape=(encoding_dim,))

decoder_layer = autoencoder.layers[-1]

decoder = tf.keras.models.Model(encoded_input, decoder_layer(encoded_input))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

try:
	import matplotlib.pyplot as plt

	n = 10  # how many digits we will display
	plt.figure(figsize=(20, 4))
	for i in range(n):
	    # display original
	    ax = plt.subplot(2, n, i + 1)
	    plt.imshow(x_test[i].reshape(28, 28))
	    plt.gray()
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)

	    # display reconstruction
	    ax = plt.subplot(2, n, i + 1 + n)
	    plt.imshow(decoded_imgs[i].reshape(28, 28))
	    plt.gray()
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)
	plt.show()
except:
	pass

