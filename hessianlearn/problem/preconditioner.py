
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
import tensorflow as tf
if int(tf.__version__[0]) > 1:
	import tensorflow.compat.v1 as tf
	tf.disable_v2_behavior()


class Preconditioner(object):
	"""
	This class describes a preconditioner, currently it is empty

	Child class should implement method __call__ which implements
	the preconditioner approximation of the (Hessian) inverse
	"""


class IdentityPreconditioner(Preconditioner):
	"""
	This class describes identity preconditioning, which means doing nothing
	"""
	def __init__(self,problem,dtype = tf.float32):
		"""
		The constructor for this class takes:
			-problem: hessianlearn.problem.Problem class
			-dtype: data type
		"""
		# Rethink this later and improve for Krylov methods.
		self.x = tf.placeholder(dtype,problem.gradient.shape,name='vec_for_prec_apply')


	def __call__(self):
		"""
		The call method simply returns vector which must be passed to
		the sess at runtime. self.x is a placeholder variable.
		"""
		return self.x



