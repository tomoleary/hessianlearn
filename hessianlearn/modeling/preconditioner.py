from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf


class Preconditioner(object):
	def __init__(self):



		# Define loss function and accuracy in __init__
		pass







class IdentityPreconditioner(Preconditioner):
	def __init__(self,problem,dtype = tf.float32):
		self.x = tf.placeholder(dtype,problem.gradient.shape,name='vec_for_prec_apply')


	def __call__(self):
		return self.x



