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

import pickle


settings = {}
# Set run specifications
settings['batch_size'] = 100
settings['hess_batch_size'] = 10
settings['tikhonov_gamma'] = 0.0
settings['intra_threads'] = 2
settings['inter_threads'] = 2

# Optimizer specifications
settings['optimizer'] = 'lrsfn'
settings['sfn_lr'] = 10
settings['fixed_step'] = 1
settings['alpha'] = 5e-2
settings['max_backtrack'] = 4

settings['max_sweeps'] = 20

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
data = Data([x_train,y_train],settings['batch_size'],test_data = [x_test,y_test],hessian_batch_size = settings['hess_batch_size'],test_data_size = testing_data_size)

settings['input_shape'] = data._input_shape
settings['output_shape'] = data._output_shape

training_data_size = len(x_train)
testing_data_size = len(x_test)
batch_size = settings['batch_size']
hess_batch_size = settings['hess_batch_size']
settings['batch_factor'] = [float(batch_size)/float(training_data_size),\
					 float(hess_batch_size)/float(training_data_size)]

################################################################################
# Create the neural network in keras

encoding_dim = 32  
input_img = tf.keras.layers.Input(shape=(784,))
encoded = tf.keras.layers.Dense(encoding_dim, activation='softplus')(input_img)
decoded = tf.keras.layers.Dense(784, activation='sigmoid')(encoded)
autoencoder = tf.keras.models.Model(input_img, decoded)

# encoder = tf.keras.models.Model(input_img, encoded)

# encoded_input = tf.keras.layers.Input(shape=(encoding_dim,))

# decoder_layer = autoencoder.layers[-1]

# decoder = tf.keras.models.Model(encoded_input, decoder_layer(encoded_input))


################################################################################
# Instantiate the problem, regularization.

problem = AutoencoderProblem(autoencoder,dtype=tf.float32)

regularization = L2Regularization(problem,beta = settings['tikhonov_gamma'])

print(80*'#')
print(('Size of configuration space: '+str(problem.dimension)).center(80))
print(80*'#')

settings['dimension'] = problem.dimension

################################################################################
# Instantiate the model object

# Model takes problem regularization and the specified optimizer in its constructor

sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=settings['intra_threads'],\
											inter_op_parallelism_threads=settings['inter_threads']))


if settings['optimizer'] == 'adam':
	print(('Using Adam optimizer').center(80))
	print(('Batch size = '+str(batch_size)).center(80))
	optimizer = Adam(problem,regularization,sess)
	optimizer.parameters['alpha'] = settings['alpha']
	optimizer.alpha = settings['alpha']

elif settings['optimizer'] == 'gd':
	print('Using gradient descent optimizer with line search'.center(80))
	print(('Batch size = '+str(batch_size)).center(80))
	optimizer = GradientDescent(problem,regularization,sess)
	optimizer.parameters['globalization'] = 'line_search'
	optimizer.parameters['max_backtracking_iter'] = 8
elif settings['optimizer'] == 'incg':
	if not settings['fixed_step']:
		print('Using inexact Newton CG optimizer with line search'.center(80))
		print(('Batch size = '+str(batch_size)).center(80))
		print(('Hessian batch size = '+str(hess_batch_size)).center(80))
		optimizer = InexactNewtonCG(problem,regularization,sess)
		optimizer.parameters['globalization'] = 'line_search'
		optimizer.parameters['max_backtracking_iter'] = settings['max_backtrack']
	else:
		print('Using inexact Newton CG optimizer with fixed step'.center(80))
		print(('Batch size = '+str(batch_size)).center(80))
		print(('Hessian batch size = '+str(hess_batch_size)).center(80))
		optimizer = InexactNewtonCG(problem,regularization,sess)
		optimizer.parameters['globalization'] = 'None'
		optimizer.alpha = settings['alpha']
elif settings['optimizer'] == 'lrsfn':
	if not settings['fixed_step']:
		print('Using low rank SFN optimizer with line search'.center(80))
		print(('Batch size = '+str(batch_size)).center(80))
		print(('Hessian batch size = '+str(hess_batch_size)).center(80))
		print(('Hessian low rank = '+str(settings['sfn_lr'])).center(80))
		optimizer = LowRankSaddleFreeNewton(problem,regularization,sess)
		optimizer.parameters['globalization'] = 'line_search'
		optimizer.parameters['max_backtracking_iter'] = settings['max_backtrack']
		optimizer.parameters['hessian_low_rank'] = settings['sfn_lr']
	else:
		print('Using low rank SFN optimizer with fixed step'.center(80))
		print(('Batch size = '+str(batch_size)).center(80))
		print(('Hessian batch size = '+str(hess_batch_size)).center(80))
		print(('Hessian low rank = '+str(settings['sfn_lr'])).center(80))
		optimizer = LowRankSaddleFreeNewton(problem,regularization,sess)
		optimizer.parameters['hessian_low_rank'] = settings['sfn_lr']
		optimizer.parameters['alpha'] = settings['alpha']
		optimizer.alpha = settings['alpha']
else:
	raise


# After optimizer is instantiated, we call the global variables initializer
init = tf.global_variables_initializer()
sess.run(init)

# random_state = np.random.RandomState(seed = 0)
# w_0 = random_state.randn(problem.dimension)
# sess.run(problem._assignment_ops,feed_dict = {problem._assignment_placeholder:w_0})

print(80*'#')
# First print
print('{0:8} {1:11} {2:11} {3:11} {4:11} {5:11} {6:11} {7:11}'.format(\
						'Sweeps'.center(8),'Loss'.center(8),'acc train'.center(8),'||g||'.center(8),\
											'Loss_test'.center(8), 'acc test'.center(8),'max test'.center(8), 'alpha'.center(8)))
x_test, y_test = next(iter(data.test))

test_dict = {problem.x: x_test}

# Iteration Loop
max_sweeps = settings['max_sweeps']
train_data = iter(data.train)
x_batch,y_batch = next(train_data)
sweeps = 0
min_test_loss = np.inf
max_test_acc = -np.inf
t0 = time.time()
for i, (data_g,data_H) in enumerate(zip(data.train,data.hess_train)):
	x_batch,y_batch = data_g
	x_hess, y_hess = data_H

	feed_dict = {problem.x: x_batch}
	hess_dict = {problem.x: x_hess}


	norm_g, loss_train, accuracy_train = sess.run([problem.norm_g,problem.loss,problem.accuracy],feed_dict)
	# logger['||g||'][i] = norm_g
	# logger['loss'][i] = loss_train
	# logger['accuracy_train'][i] = accuracy_train
	# logger['time'][i] = time.time() - t0
	
	# logger['sweeps'][i] = sweeps
	loss_test,	accuracy_test = sess.run([problem.loss,problem.accuracy],test_dict)
	# logger['accuracy_test'][i] = accuracy_test
	# logger['loss_test'][i] = loss_test
	min_test_loss = min(min_test_loss,loss_test)
	max_test_acc = max(max_test_acc,accuracy_test)
	# if accuracy_test == max_test_acc:
	# 	if len(logger['best_weight']) > 2:
	# 		logger['best_weight'].pop(0)
	# 	acc_weight_tuple = (accuracy_test,accuracy_train,sess.run(problem._flat_w))
	# 	logger['best_weight'].append(acc_weight_tuple) 

	sweeps = np.dot(settings['batch_factor'],optimizer.sweeps)
	if i % 10 == 0:
		# Print once each epoch
		try:
			print(' {0:^8.2f} {1:1.4e} {2:.3%} {3:1.4e} {4:1.4e} {5:.3%} {6:.3%} {7:1.4e}'.format(\
				sweeps, loss_train,accuracy_train,norm_g,loss_test,accuracy_test,max_test_acc,optimizer.alpha))
		except:
			print(' {0:^8.2f} {1:1.4e} {2:.3%} {3:1.4e} {4:1.4e} {5:.3%} {6:.3%} {7:11}'.format(\
				sweeps, loss_train,accuracy_train,norm_g,loss_test,accuracy_test,max_test_acc,optimizer.alpha))
	try:
		optimizer.minimize(feed_dict,hessian_feed_dict=hess_dict)
	except:
		optimizer.minimize(feed_dict)

	if sweeps > max_sweeps:
		break


	# try:
	# 	os.makedirs('results/')
	# except:
	# 	pass
	# with open('results/'+ outname +'.pkl', 'wb+') as f:
	# 	pickle.dump(logger, f, pickle.HIGHEST_PROTOCOL)
