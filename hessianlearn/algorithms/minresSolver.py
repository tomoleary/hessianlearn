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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import numpy as np
import tensorflow as tf
if int(tf.__version__[0]) > 1:
	import tensorflow.compat.v1 as tf
	tf.disable_v2_behavior()

from ..utilities.parameterList import ParameterList
from ..algorithms import Optimizer
from .. modeling import IdentityPreconditioner
from ..modeling import L2Regularization
from abc import ABC, abstractmethod

class Identity(object):
	def __init__(self):

		pass
	
	def __call__(self, x):
		return x



def ParametersMINRESSolver(dictionary = {}):
	parameters = dictionary
	parameters["rel_tolerance"] = [1e-9, "the relative tolerance for the stopping criterion"]
	parameters["abs_tolerance"] = [1e-12, "the absolute tolerance for the stopping criterion"]
	parameters["max_iter"]      = [20, "the maximum number of iterations"]
	parameters["zero_initial_guess"] = [True, "if True we start with a 0\
						 initial guess; if False we use the x as initial guess."]
	parameters["print_level"] = [-1, "verbosity level: -1 --> no output on \
			screen; 0 --> only final residual at convergence or reason for not not convergence"]
	
	parameters['coarse_tol'] = [0.5,'coarse tolerance used in calculation \
									of relative tolerances for E-W conditions']
	return ParameterList(parameters)


class MINRESSolver(ABC):




	reason = ["Maximum Number of Iterations Reached",
			  "Relative/Absolute residual less than tol",
			  "Reached a negative direction",
			  "Reached trust region boundary"
			  ]
	def __init__(self,problem,regularization,sess = None,preconditioner = None,\
		x = None,parameters = ParametersMINRESSolver()):
		self.sess = sess
		self.problem = problem
		self.regularization = regularization
		if x is None:
			# self.x = tf.Variable(self.problem.gradient.initialized_value())
			self.x = self.problem.gradient
		else:
			self.x = x
		self.parameters = parameters

		
		self.Aop = self.problem.H_w_hat + self.regularization.H_w_hat

		# # Define preconditioner 
		# if preconditioner is None:
		# 	self.Minv = IdentityPreconditioner(problem,self.problem.dtype)
		# else:
		# 	self.Minv = preconditioner






	def solve(self,b,feed_dict = None,x_0 = None):
		r"""
		Solve Ax=b by the mines method
		as defined in Iterative Methods Ed. 2 by Youssef Saad p 140
		"""
		assert self.sess is not None
		assert feed_dict is not None

		self.iter = 0
		self.converged = False
		self.reason_id = 0
		x = np.zeros_like(b)

		feed_dict[self.problem.w_hat] = x
		Ax_0 = self.sess.run(self.Aop,feed_dict = feed_dict)
		# Calculate initial residual r = Ax_0 -b
		r = b - Ax_0
		# Calculate tolerance for Eisenstat Walker conditions
		rr = np.dot(r,r)
		rtol2 = rr * self.parameters["rel_tolerance"] * self.parameters["rel_tolerance"]
		atol2 = self.parameters["abs_tolerance"] * self.parameters["abs_tolerance"]
		tol = max(rtol2, atol2)
		import scipy
		from scipy.sparse.linalg import LinearOperator

		def Ap(p):
			feed_dict[self.problem.w_hat] = p
			return self.sess.run(self.Aop,feed_dict = feed_dict)

		n = self.problem.dimension

		A = LinearOperator((n,n), matvec=Ap)

		x = np.zeros_like(b)
		p = A(r)

		converged = False
		while not converged :
			self.iter +=1
			alpha = np.dot(p,r)/rr
			x_old = x 
			x += alpha*r
			r -= alpha*p

			p = A(r)
			# This is the extra query of the network to see if the direction
			# is about to rescale gradient components in indefinite directions
			# towards saddle points in which case one can break before 
			# updating
			pAp = np.dot(p,A(p))
			if pAp < 0:
				return x_old, converged

			converged = (np.dot(r,r) < tol)
			if self.iter < self.parameters["max_iter"]:
				return x, converged

		return x, converged




