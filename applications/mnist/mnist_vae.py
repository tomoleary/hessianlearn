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
flattened_dimension = np.prod(x_train.shape[1:])
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Instante the data object
data = Data([x_train,y_train],settings['batch_size'],test_data = [x_test,y_test],hessian_batch_size = settings['hess_batch_size'])

# settings['input_shape'] = data._input_shape
# settings['output_shape'] = data._output_shape


################################################################################
# Build the variational autoencoder neural network model here

# network parameters
input_shape = (flattened_dimension, )
intermediate_dim = 512
latent_dim = 2

# VAE model = encoder + decoder
# build encoder model
inputs = tf.keras.layers.Input(shape=input_shape)
x_encoder = tf.keras.layers.Dense(intermediate_dim, activation='softplus')(inputs)
z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x_encoder)
z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x_encoder)

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = tf.keras.backend.shape(z_mean)[0]
    dim = tf.keras.backend.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon
# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = tf.keras.models.Model(inputs, [z_mean, z_log_var, z], name='encoder')

# build decoder model
latent_inputs = tf.keras.layers.Input(shape=(latent_dim,), name='z_sampling')
x_decoder = tf.keras.layers.Dense(intermediate_dim, activation='softplus')(latent_inputs)
outputs = tf.keras.layers.Dense(flattened_dimension, activation='sigmoid')(x_decoder)

# instantiate decoder model
decoder = tf.keras.models.Model(latent_inputs, outputs, name='decoder')

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = tf.keras.models.Model(inputs, outputs, name='vae_mlp')



################################################################################
# Instantiate the problem, regularization.

# problem = AutoencoderProblem(vae,inputs = inputs,dtype=tf.float32)
problem = VariationalAutoencoderProblem(vae,z_mean,z_log_var,inputs,dtype=tf.float32)

settings['tikhonov_gamma'] = 1e-2
regularization = L2Regularization(problem,gamma = settings['tikhonov_gamma'])


################################################################################
# Instantiate the model object
HLModelSettings = HessianlearnModelSettings()

HLModelSettings['optimizer'] = 'sgd'
HLModelSettings['alpha'] = 5e-4
HLModelSettings['fixed_step'] = False
HLModelSettings['sfn_lr'] = 20
HLModelSettings['max_backtrack'] = 16
HLModelSettings['max_sweeps'] = 50

HLModelSettings['problem_name'] = 'mnist_vae'


HLModel = HessianlearnModel(problem,regularization,data,settings = HLModelSettings)


# Can pass in an initial guess for the weights w_0 to the method fit, if desired.
HLModel.fit(w_0 = None)

################################################################################
# Post processing
import matplotlib.pyplot as plt
def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as a function of the 2D latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = (n - 1) * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


models = (encoder, decoder)
data = (x_test, y_test)
plot_results(models,
             data,
             batch_size=settings['batch_size'],
             model_name="vae_mlp")


