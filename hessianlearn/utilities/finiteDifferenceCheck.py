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
import numpy as np
from numpy.linalg import norm
import tensorflow as tf
if int(tf.__version__[0]) > 1:
	import tensorflow.compat.v1 as tf
	tf.disable_v2_behavior()


def finite_difference_check(sess,problem, feed_dict, w = None, w_hat=None,verbose = False):

	if w is None:
		w = sess.run(problem.w)
		# w_zeros = []
		# for w_i in w:
		# 	w_zeros.append(np.zeros_like(w))
	if w_hat is None:
		w_hat = []
		for w_i in w:
			# print('Shape',w_i.shape)
			w_hat.append(np.ones_like(w_i))
		# w_hat = [np.ones_like(w_i) for w_i in w]

	eps = np.power(2., np.linspace(-32, 0, 33))
	
	initial_loss       = sess.run(problem.loss,feed_dict)
	

	initial_g       = sess.run(problem.gradient,feed_dict)

	feed_dict[problem.w_hat] = w_hat
	initial_g_w_hat = np.sum(sess.run(problem.g_inner_w_hat,feed_dict))
	
	initial_h_w_hat = sess.run(problem.H_w_hat,feed_dict)

	error_g = np.zeros_like(eps)
	error_H = np.zeros_like(eps)

	# We will need to modify w during this process so we copy 
	# the initial values of w so we can replace them later
	print('Copying initial w since it will be modified during this check')
	w_array = sess.run(problem.w)	
	w_changed = True

	if verbose:
		print('Initial loss:',initial_loss)
		# print('Initial gradient:',initial_g)
		print('Initial g_w_hat',initial_g_w_hat)
		print('{0:10} {1:10} {2:10} {3:10}'.format('epsilon','loss','error_g','error_H'))
	

	for i in np.arange(eps.shape[0]):


		eps_i  = eps[i]
		# Momentarily assign w
		# w_update = [eps_i*w_hat_i for w_hat_i in w_hat]
		# # w_plus = w + eps_i*w_hat 
		# problem._update_w(w_update)
		new_w = []
		for w_i,w_hat_i in zip(w,w_hat):
			new_w.append(w_i + eps_i*w_hat_i)
		sess.run(problem._assign_to_w(new_w))
		#Evaluate new loss and calculate gradient error
		loss_plus = sess.run(problem.loss,feed_dict)
		error_g_i = np.abs( (loss_plus - initial_loss)/eps_i - initial_g_w_hat)
		error_g[i] = error_g_i
		# Evaluate new gradient and calculate Hessian error
		g_plus = sess.run(problem.gradient,feed_dict)
		error_H_i_ = []
		for g_plus_i,initial_g_i,initial_h_w_hat_i in zip(g_plus,initial_g,initial_h_w_hat):
			error_H_i_.append((g_plus_i - initial_g_i)/eps_i-initial_h_w_hat_i)
		error_H_i = np.sqrt(np.sum([np.linalg.norm(e)**2 for e in error_H_i_]))
		error_H[i] = error_H_i

		if verbose:
			print('{0:1.4e} {1:1.4e} {2:1.4e} {3:1.4e}'.format(eps_i,loss_plus,error_g_i,error_H_i))
		
	if w_changed:
		problem._assign_to_w(w_array)
		print('Succesfully re-assigned w')

	out = {}
	out['epsilon'] = eps
	out['error_g'] = error_g
	out['error_H'] = error_H

	return out









