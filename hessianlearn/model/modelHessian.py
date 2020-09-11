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
from abc import ABC, abstractmethod


class ModelHessian(ABC):

	def __init__(self,Model):
		self.data_dict = None
		pass

	@property
	def T(self):
		return self._T
	

	def __mult__(self,x):


	def _T(self):
		return self


	def __call__(self,x):
		pass
		assert self.problem is not None
		assert self.sess is not None

		feed_dict[self.problem.w_hat] = x
		H_w_hat = self.sess.run(self.problem.H_w_hat,feed_dict)
		return H_w_hat

