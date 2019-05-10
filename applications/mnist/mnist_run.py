import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
sys.path.append('../../')
from hessianlearn import *

import pickle


# Parse run specifications
from argparse import ArgumentParser

parser = ArgumentParser(add_help=True, description="-b batch size (int) \
														-p population_size (int) -alpha (float)")
# parser.add_argument('-hl',dest = 'path_to_hl',required=True, help="path to hippylearn, required!",type=str)
parser.add_argument("-optimizer", dest='optimizer',required=False, default = 'incg', help="optimizer type",type=str)
parser.add_argument('-alpha',dest = 'alpha',required = False,default = 5e-2,help= 'learning rate alpha',type=float)
parser.add_argument('-sfn_lr',dest = 'sfn_lr',required= False,default = 20,help='low rank for sfn',type = int)
parser.add_argument('-record_spectrum',dest = 'record_spectrum',\
					required= False,default = 0,help='boolean for recording spectrum',type = int)
parser.add_argument('-weight_burn_in',dest = 'weight_burn_in',\
					required= False,default = 0,help='',type = int)
parser.add_argument('-n_threads',dest = 'n_threads',required= False,default = 2,help='threads',type = int)
parser.add_argument('-batch_ratio',dest = 'batch_ratio',required= False,default = 0.1,help='threads',type = float)

args = parser.parse_args() #
# Check command line arguments
optimizers = ['adam','gd','incg','ingmres','lrsfn','sgd']
assert args.optimizer in optimizers,\
 '+\n'+80*'#'+'\n'+'Error: choose optimizer from adam, gd, incg, ingmres, lrsfn'.center(80)+'\n'+80*'#'+'\n'
# try:
#     sys.path.append(args.path_to_hl)
#     from hessianlearn import *
# except:
#     print('Error loading!!!')
#     exit()
#     pass

# Instantiate data
training_data_size = 10000
batch_size = 10000
hess_batch_size = 1000
testing_data_size = 10000

batch_factor = [1, float(hess_batch_size)/float(batch_size)]
data_func = load_mnist
raw_data = data_func()

data = Data(raw_data,training_data_size,\
		batch_size,hessian_batch_size = hess_batch_size,test_data_size = testing_data_size)

# Define network and instantiate problem and regularization
architecture = {}
architecture['input_shape'] = data._input_shape
architecture['n_filters'] = [4, 4]
architecture['filter_sizes'] = [8,4]

CAE = GenericCAE(architecture)
problem = AutoencoderProblem(CAE,dtype=tf.float32)
regularization = L2Regularization(problem,beta = 1e-1)
print(80*'#')
print(('Size of configuration space: '+str(problem.dimension)).center(80))
print(80*'#')

# Initialize Logging 
logger = {}
logger['dimension'] = problem.dimension
logger['loss'] = {}
logger['loss_test'] = {}
logger['||g||'] ={}
logger['sweeps'] = {}

if args.record_spectrum:
	logger['lambdases'] = {}
	logger['lambdases_test'] = {}

# Instantiate session, initialize variables
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=args.n_threads))
init = tf.global_variables_initializer()
sess.run(init)

# Burn in random number generator, draw and assign initial weights
random_state = np.random.RandomState(seed = 0)
for _ in range(args.weight_burn_in):
	__ = random_state.randn(problem.dimension)
w_0 = random_state.randn(problem.dimension)
sess.run(problem._assign_to_w(w_0))

name_appendage = ''

# Optimizer selection
if args.optimizer == 'adam':
	batch_size = int(args.batch_ratio*training_data_size)
	data = Data(raw_data,training_data_size,\
		batch_size,hessian_batch_size = hess_batch_size,test_data_size = testing_data_size)
	print(('Using Adam optimizer, with '+str(100*args.batch_ratio)+'% mini-batches').center(80))
	print(('Batch size = '+str(batch_size)).center(80))
	optimizer = Adam(problem,regularization,sess)
	optimizer.parameters['alpha'] = args.alpha
	optimizer.alpha = args.alpha
	batch_factor = [args.batch_ratio,0]
elif args.optimizer == 'gd':
	print('Using gradient descent optimizer with line search'.center(80))
	print(('Batch size = '+str(batch_size)).center(80))
	optimizer = GradientDescent(problem,regularization,sess)
	optimizer.parameters['globalization'] = 'line_search'
elif args.optimizer == 'incg':
	print('Using inexact Newton CG optimizer with line search'.center(80))
	print(('Batch size = '+str(batch_size)).center(80))
	print(('Hessian batch size = '+str(hess_batch_size)).center(80))
	optimizer = InexactNewtonCG(problem,regularization,sess)
	optimizer.parameters['globalization'] = 'line_search'
elif args.optimizer == 'ingmres':
	print('Using inexact Newton GMRES optimizer with line search'.center(80))
	print(('Batch size = '+str(batch_size)).center(80))
	print(('Hessian batch size = '+str(hess_batch_size)).center(80))
	optimizer = InexactNewtonGMRES(problem,regularization,sess)
	optimizer.parameters['globalization'] = 'line_search'
elif args.optimizer == 'lrsfn':
	if False:
		print('Using low rank SFN optimizer with line search'.center(80))
		print(('Batch size = '+str(batch_size)).center(80))
		print(('Hessian batch size = '+str(hess_batch_size)).center(80))
		print(('Hessian low rank = '+str(args.sfn_lr)).center(80))
		optimizer = LowRankSaddleFreeNewton(problem,regularization,sess)
		optimizer.parameters['globalization'] = 'line_search'
		optimizer.parameters['hessian_low_rank'] = args.sfn_lr
	if True:
		print('Using low rank SFN optimizer with fixed step'.center(80))
		print(('Batch size = '+str(batch_size)).center(80))
		print(('Hessian batch size = '+str(hess_batch_size)).center(80))
		print(('Hessian low rank = '+str(args.sfn_lr)).center(80))
		optimizer = LowRankSaddleFreeNewton(problem,regularization,sess)
		optimizer.parameters['hessian_low_rank'] = args.sfn_lr
		optimizer.parameters['alpha'] = args.alpha
		optimizer.alpha = args.alpha
		name_appendage += 'fixed_step'
elif args.optimizer == 'sgd':
	batch_size = int(args.batch_ratio*training_data_size)
	data = Data(raw_data,training_data_size,\
		batch_size,hessian_batch_size = hess_batch_size,test_data_size = testing_data_size)
	print(('Using stochastic gradient descent optimizer, with '+str(100*args.batch_ratio)+'% mini-batches').center(80))
	print(('Batch size = '+str(batch_size)).center(80))
	optimizer = GradientDescent(problem,regularization,sess)
	optimizer.parameters['alpha'] = args.alpha
	optimizer.alpha = args.alpha
	batch_factor = [args.batch_ratio,0]
	

alpha = args.alpha
optimizer.parameters['alpha'] = alpha

print(80*'#')
# First print
print('{0:8} {1:11} {2:11} {3:11} {4:11}'.format(\
						'Sweeps'.center(8),'Loss'.center(8),'||g||'.center(8),\
											'Loss_test'.center(8), 'alpha'.center(8)))


x_test, y_test = next(iter(data.test))
test_dict = {problem.x: x_test}
# Iteration Loop
max_sweeps = 100
train_data = iter(data.train)
x_batch,y_batch = next(train_data)
sweeps = 0

for i, (data_g,data_H) in enumerate(zip(data.train,data.hess_train)):
	x_batch,y_batch = data_g
	x_hess, y_hess = data_H
	feed_dict = {problem.x: x_batch}
	hess_dict = {problem.x: x_hess}
	if i%1==0:
		norm_g = sess.run(problem.norm_g,feed_dict)
		logger['||g||'][i] = norm_g
		loss_train = sess.run(problem.loss, feed_dict)
		logger['loss'][i] = loss_train
		sweeps = np.dot(batch_factor,optimizer.sweeps)
		logger['sweeps'][i] = sweeps
		loss_test = sess.run(problem.loss,test_dict)
		loss_test = sess.run(problem.loss,test_dict)
		logger['loss_test'][i] = loss_test
		try:
			print(' {0:^8.2f} {1:1.4e} {2:1.4e} {3:1.4e} {4:1.4e}'.format(\
				sweeps, loss_train,norm_g,loss_test,optimizer.alpha))
		except:
			print(' {0:^8.2f} {1:1.4e} {2:1.4e} {3:1.4e} {4:11}'.format(\
				sweeps, loss_train,norm_g,loss_test,optimizer.alpha))
		try:
			sess.run(optimizer.minimize(feed_dict,hessian_feed_dict=hess_dict))
		except:
			sess.run(optimizer.minimize(feed_dict))
		if args.record_spectrum:
			k = 100
			d,_ = low_rank_hessian(optimizer,hess_dict,k)
			logger['lambdases'][i] = d
			d_test,_ = low_rank_hessian(optimizer,test_dict,k)
			logger['lambdases_test'][i] = d_test
	if sweeps > max_sweeps:
		break

outname =  str(data_func.__name__)+\
	str(optimizer.__class__.__name__)+str(alpha)+'_'+str(training_data_size)+\
				'_'+str(batch_factor[-1])+'_burn'+str(args.weight_burn_in)
outname += name_appendage
try:
	os.makedirs('results/')
except:
	pass
with open('results/'+ outname +'.pkl', 'wb+') as f:
	pickle.dump(logger, f, pickle.HIGHEST_PROTOCOL)


