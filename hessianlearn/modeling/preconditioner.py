
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


class Preconditioner(object):
	def __init__(self):



		# Define loss function and accuracy in __init__
		pass







class IdentityPreconditioner(Preconditioner):
	def __init__(self,problem,dtype = tf.float32):
		self.x = tf.placeholder(dtype,problem.gradient.shape,name='vec_for_prec_apply')


	def __call__(self):
		return self.x



