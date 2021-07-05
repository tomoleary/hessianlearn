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
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["KMP_WARNINGS"] = "FALSE" 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pickle
import tensorflow as tf
import time, datetime
# if int(tf.__version__[0]) > 1:
#   import tensorflow.compat.v1 as tf
#   tf.disable_v2_behavior()


# Memory issue with GPUs
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
# Load hessianlearn library
import sys
sys.path.append( os.environ.get('HESSIANLEARN_PATH', "../../"))
from hessianlearn import *

# Parse run specifications
from argparse import ArgumentParser

parser = ArgumentParser(add_help=True)
parser.add_argument("-optimizer", dest='optimizer',required=False, default = 'lrsfn', help="optimizer type",type=str)
parser.add_argument('-fixed_step',dest = 'fixed_step',\
					required= False,default = 1,help='boolean for fixed step vs globalization',type = int)
parser.add_argument('-alpha',dest = 'alpha',required = False,default = 1e-5,help= 'learning rate alpha',type=float)
parser.add_argument('-hessian_low_rank',dest = 'hessian_low_rank',required= False,default = 40,help='low rank for sfn',type = int)
parser.add_argument('-record_spectrum',dest = 'record_spectrum',\
					required= False,default = 0,help='boolean for recording spectrum',type = int)

parser.add_argument("-resnet_weights", dest='resnet_weights',required=False, default = 'imagenet', help="initialization for network weights",type=str)

parser.add_argument('-batch_size',dest = 'batch_size',required= False,default = 32,help='batch size',type = int)
parser.add_argument('-hess_batch_size',dest = 'hess_batch_size',required= False,default = 8,help='hess batch size',type = int)
parser.add_argument('-keras_epochs',dest = 'keras_epochs',required= False,default = 50,help='keras_epochs',type = int)
parser.add_argument("-keras_opt", dest='keras_opt',required=False, default = 'adam', help="optimizer type for keras",type=str)
parser.add_argument('-keras_alpha',dest = 'keras_alpha',required= False,default = 1e-3,help='keras learning rate',type = float)
parser.add_argument('-max_sweeps',dest = 'max_sweeps',required= False,default = 2,help='max sweeps',type = float)

parser.add_argument("-loss_type", dest='loss_type',required=False, default = 'mixed', help="loss type either cross_entrop or mixed",type=str)
parser.add_argument('-seed',dest = 'seed',required= False,default = 0,help='seed',type = int)


args = parser.parse_args()

try:
  tf.set_random_seed(args.seed)
except:
  tf.random.set_seed(args.seed)

# GPU Environment Details
gpu_availabe = tf.test.is_gpu_available()
built_with_cuda = tf.test.is_built_with_cuda()
print(80*'#')
print(('IS GPU AVAILABLE: '+str(gpu_availabe)).center(80))
print(('IS BUILT WITH CUDA: '+str(built_with_cuda)).center(80))
print(80*'#')

settings = {}
# Set run specifications
# Data specs
settings['batch_size'] = args.batch_size
settings['hess_batch_size'] = args.hess_batch_size


################################################################################
# Instantiate data
(x_train, y_train), (_x_test, _y_test) = tf.keras.datasets.cifar100.load_data()

# # Normalize the data
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.

x_train = tf.keras.applications.resnet50.preprocess_input(x_train)
x_test_full = tf.keras.applications.resnet50.preprocess_input(_x_test)
x_val = x_test_full[:2000]
x_test = x_test_full[2000:]

y_train = tf.keras.utils.to_categorical(y_train)
y_test_full = tf.keras.utils.to_categorical(_y_test)
y_val = y_test_full[:2000]
y_test = y_test_full[2000:]

################################################################################
# Create the neural network in keras

# tf.keras.backend.set_floatx('float64')

resnet_input_shape = (200,200,3)
input_tensor = tf.keras.Input(shape = resnet_input_shape)

if args.resnet_weights == 'None':
    pretrained_resnet50 = tf.keras.applications.resnet50.ResNet50(weights = None,include_top=False,input_tensor=input_tensor)
else:
    pretrained_resnet50 = tf.keras.applications.resnet50.ResNet50(weights = 'imagenet',include_top=False,input_tensor=input_tensor)

for layer in pretrained_resnet50.layers[:143]:
    layer.trainable = False

classifier = tf.keras.models.Sequential()
classifier.add(tf.keras.layers.Input(shape=(32,32,3)))
classifier.add(tf.keras.layers.Lambda(lambda image: tf.image.resize(image, resnet_input_shape[:2])))
classifier.add(pretrained_resnet50)
classifier.add(tf.keras.layers.Flatten())
classifier.add(tf.keras.layers.BatchNormalization())
classifier.add(tf.keras.layers.Dense(128, activation='relu'))
classifier.add(tf.keras.layers.Dropout(0.5))
classifier.add(tf.keras.layers.BatchNormalization())
classifier.add(tf.keras.layers.Dense(100, activation='softmax'))


if args.keras_opt == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate = args.keras_alpha,epsilon = 1e-8)
elif args.keras_opt == 'sgd':
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.keras_alpha)
else: 
    raise

if args.loss_type == 'mixed':
    def mixed(y_true, y_pred):
        squared_difference = tf.square(y_true - y_pred)
        return tf.reduce_mean(squared_difference, axis=-1) +tf.keras.losses.CategoricalCrossentropy(from_logits = True)(y_true, y_pred)
    loss = mixed
else:
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True)


classifier.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])


loss_train_0, acc_train_0 = classifier.evaluate(x_train,y_train,verbose=2)
print('acc_train = ',acc_train_0)
loss_test_0, acc_test_0 = classifier.evaluate(x_test,y_test,verbose=2)
print('acc_test = ',acc_test_0)
loss_val_0, acc_val_0 = classifier.evaluate(x_val,y_val,verbose=2)
print('acc_val = ',acc_val_0)

aux_keras_data = {'loss_train_0':loss_train_0,'acc_traun_0':acc_train_0,\
                    'loss_test_0':loss_test_0,'acc_test_0':acc_test_0,\
                    'loss_val_0':loss_val_0, 'acc_val_0':acc_val_0}

no_callback = True
if no_callback:
    callbacks = []
else:
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_acc',restore_best_weights = True)]

keras_directory = 'keras_logging_cifar100/'
# CSV logging
if not os.path.exists(keras_directory):
    os.makedirs(keras_directory)
keras_logger_name = keras_directory+str(datetime.date.today())+args.keras_opt+str(args.keras_alpha)+'_'+str(args.keras_epochs)+'_seed'str(args.seed)+'.csv'
callbacks.append(tf.keras.callbacks.CSVLogger(keras_logger_name, append=True, separator=';'))

classifier.fit(x_train[:], y_train[:], epochs=args.keras_epochs,batch_size = 32,\
               callbacks = callbacks ,verbose = True,validation_data = (x_val,y_val))


# Grab the weights and check the accuracy post process
set_weights = {}

for layer in classifier.layers:
    set_weights[layer.name] = classifier.get_layer(layer.name).get_weights()

# Post process and save additional information from keras training
loss_test_keras_final, acc_test_keras_final = classifier.evaluate(x_test,y_test,verbose=2)
loss_val_keras_final, acc_val_keras_final = classifier.evaluate(x_val,y_val,verbose=2)
print(80*'#')
print('After keras training'.center(80))
print('acc_test = ',acc_test_keras_final)
print('acc_val = ',acc_val_keras_final)
aux_keras_data['loss_test_final'] = loss_test_keras_final
aux_keras_data['acc_test_final'] = acc_test_keras_final
aux_keras_data['loss_val_final'] = loss_val_keras_final
aux_keras_data['acc_val_final'] = acc_val_keras_final
keras_aux_logger_name = keras_logger_name.split('.cvs')[0]+'aux_data.pkl'
with open(keras_aux_logger_name,'wb+') as f:
    pickle.dump(aux_keras_data,f,pickle.HIGHEST_PROTOCOL)


################################################################################
# Instantiate the data, problem, regularization.

t0_problem_construction = time.time()
problem = ClassificationProblem(classifier,loss_type=args.loss_type,dtype=tf.float32)
print('Finished constructing the problem, and it took ',time.time() - t0_problem_construction , 's')


# Instante the data object
data = Data({problem.x:x_train,problem.y_true:y_train},settings['batch_size'],\
  validation_data = {problem.x:x_val,problem.y_true:y_val},hessian_batch_size = settings['hess_batch_size'],seed=args.seed)

settings['tikhonov_gamma'] = 0.0

regularization = L2Regularization(problem,gamma = settings['tikhonov_gamma'])


################################################################################
# Instantiate the model object
HLModelSettings = HessianlearnModelSettings()

HLModelSettings['optimizer'] = args.optimizer
HLModelSettings['alpha'] = args.alpha
HLModelSettings['globalization'] = None
HLModelSettings['hessian_low_rank'] = args.hessian_low_rank
HLModelSettings['max_backtrack'] = 20
HLModelSettings['max_sweeps'] = args.max_sweeps
HLModelSettings['layer_weights'] = set_weights

HLModelSettings['problem_name'] = 'cifar100_resnet_classification_seed'+str(args.seed)
if args.resnet_weights == 'None':
    HLModelSettings['problem_name'] += '_random_guess'
HLModelSettings['record_spectrum'] = bool(args.record_spectrum)
HLModelSettings['rq_data_size'] = 100
HLModelSettings['printing_sweep_frequency'] = None
HLModelSettings['printing_items']               = {'time':'time','sweeps':'sweeps','Loss':'train_loss','acc train':'train_acc',\
                                                      '||g||':'||g||','Loss val':'val_loss','acc val':'val_acc',\
                                                      'maxacc val':'max_val_acc','alpha':'alpha'}


HLModel = HessianlearnModel(problem,regularization,data,settings = HLModelSettings)

if args.max_sweeps > 0:
    HLModel.fit()


loss_test_final, acc_test_final = classifier.evaluate(x_test,y_test,verbose=2)
loss_val_final, acc_val_final = classifier.evaluate(x_val,y_val,verbose=2)

hl_aux_data = {'loss_test_0':loss_test_0,'acc_test_0':acc_test_0,\
                'loss_val_0':loss_val_0,'acc_val_0':acc_val_0,\
                'loss_test_final':loss_test_final,'acc_test_final':acc_test_final,\
                'loss_val_final':loss_val_final,'acc_val_final':acc_val_final}

with open(HLModel.settings['problem_name']+'_logging/'+ HLModel.logger_outname +'aux_data.pkl', 'wb+') as f:
    pickle.dump(hl_aux_data, f, pickle.HIGHEST_PROTOCOL)

################################################################################
# Evaluate again on all the data.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

# # Normalize the data
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.

x_train = tf.keras.applications.resnet50.preprocess_input(x_train)
x_test = tf.keras.applications.resnet50.preprocess_input(x_test)

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

loss_test_total, acc_test_total = classifier.evaluate(x_test,y_test,verbose=2)
print(80*'#')
print('After hessianlearn training'.center(80))
print('acc_test_total = ',acc_test_total)


