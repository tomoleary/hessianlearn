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
from .. problem import IdentityPreconditioner
from ..problem import L2Regularization
from abc import ABC, abstractmethod

class Identity(object):
	def __init__(self):

		pass
	
	def __call__(self, x):
		return x



def ParametersCGSolver(dictionary = {}):
	parameters = dictionary
	parameters["rel_tolerance"] = [1e-9, "the relative tolerance for the stopping criterion"]
	parameters["abs_tolerance"] = [1e-12, "the absolute tolerance for the stopping criterion"]
	parameters["max_iter"]      = [10, "the maximum number of iterations"]
	parameters["zero_initial_guess"] = [True, "if True we start with a 0 initial guess; if False we use the x as initial guess."]
	parameters["print_level"] = [-1, "verbosity level: -1 --> no output on screen; 0 --> only final residual at convergence or reason for not not convergence"]
	
	parameters['coarse_tol'] = [0.5,'coarse tolerance used in calculation of relative tolerances for E-W conditions']

	parameters['default_damping']  = [1e-3, "Levenberg-Marquardt damping when no regularization is used"]
	return ParameterList(parameters)


class CGSolver(ABC):
	"""
	This class implements a custom CG solver to be used with Inexact Newton CG
	"""
	reason = ["Maximum Number of Iterations Reached",
			  "Relative/Absolute residual less than tol",
			  "Reached a negative direction",
			  "Reached trust region boundary"
			  ]
	def __init__(self,problem,regularization,sess = None,Aop = None,preconditioner = None,x = None,parameters = ParametersCGSolver()):
		"""
		The constructor for this class takes:
			-problem: hessianlearn.problem.Problem
			-regularization: hessianlearn.problem.Regularization
			-sess: tf.Session()
			-Aop: matrix vector product callable
			-precondition: hessianlearn.problem.Preconditioner
			-parameters: solver hyperparameters
		"""
		self.sess = sess
		self.problem = problem
		if regularization.parameters['gamma'] < 1e-4:
			regularization = L2Regularization(self.problem,gamma = parameters['default_damping'])
		self.regularization = regularization
		if x is None:
			# self.x = tf.Variable(self.problem.gradient.initialized_value())
			self.x = self.problem.gradient
		else:
			self.x = x
		self.parameters = parameters
		if Aop is None:
			self.Aop = self.problem.Hdw + self.regularization.Hdw
		else:
			# be careful to note what the operator requires be passed into feed_dict 
			self.Aop = Aop
		# Define preconditioner 
		if preconditioner is None:
			self.Minv = IdentityPreconditioner(problem,self.problem.dtype)
		else:
			self.Minv = preconditioner

		self.update_x = self.update_without_trust_region
		self.B_op = None

	def initialize_trust_region(self,coarse_tol = None):
		"""
		This method initializes the trust region parameters
			-coarse_tol: coarse tolerance
		"""
		self.update_x = self.update_with_trust_region
		if coarse_tol is not None:
			self.parameters['coarse_tol'] = coarse_tol

	def set_trust_region_radius(self,radius,operator = Identity()):
		"""
		This method sets the trust region radius when trust region is used
		for globalization
			-radius: trust region radius
			-operator: for use in TR calculations
		"""
		assert self.parameters['zero_initial_guess']
		self.trust_region_radius_squared = radius**2
		self.B_op = operator

	def update_without_trust_region(self,x,alpha,p):
		"""
		This method updates the approximation of x^* and returns False when
		TR is not used
			-x: solution at given iteration
			-alpha: step length
			-p: search direction
		"""
		x = x + alpha*p
		return False, x

	def update_with_trust_region(self,x,alpha,p):
		"""
		This method returns a Boolean delineating whether the point was placed
		on the trust region boundary or not, as well as the updated x
			-x: solution at given iteration
			-alpha: step length
			-p: search direction
		"""
		step = x + alpha*p
		assert self.B_op is not None
		step_length = np.dot(x,self.B_op(step))
		if step_length < self.trust_region_radius_squared:
			return False, step
		else:
			# Move the point to the boundary of the trust region
			Bp = self.B_op(p)
			xBp = np.dot(x,Bp)
			pBp = np.dot(p,Bp)
			Bx = self.B_op(x)
			xBx = np.dot(x,Bx)
			a_tau = alpha*alpha*pBp
			b_tau = 2* alpha * xBp
			c_tau = xBx - self.trust_region_radius_squared
			discriminant = (b_tau - 4*a_tau*c_tau)
			if discriminant < 0:
				print('Issue with the discriminant')
				discriminant *= -1
			tau = 0.5*(-b_tau + math.sqrt(discriminant))/a_tau
			alpha_tau = alpha*tau
			return True, x + alpha*p

	def solve(self,b,feed_dict = None,x_0 = None):
		r"""
		Solve Ax=b by the preconditioned conjugate gradients method
		as defined in Iterative Methods Ed. 2 by Yousef Saad p 263
			-b: the right hand side
			-feed_dict: the data dictionary used to evaluate stochastic 
				operators
			-x_0: the initial guess for CG
		"""
		assert self.sess is not None
		assert feed_dict is not None

		self.iter = 0
		self.converged = False
		self.reason_id = 0
		x = np.zeros_like(b)

		feed_dict[self.problem.dw] = x
		Ax_0 = self.sess.run(self.Aop,feed_dict = feed_dict)
		# Calculate initial residual r = Ax_0 -b
		r = b - Ax_0
		# Apply preconditioner z = M^{-1}r
		feed_dict[self.Minv.x] = r
		# fix me!!!!! Preconditioner not working for now?

		z = self.sess.run(self.Minv(),feed_dict = feed_dict)


		# Calculate p (copy array)
		p = z.copy()
		# Calculate tolerance for Eisenstat Walker conditions
		rz_0 = np.dot(r,z)
		rtol2 = rz_0 * self.parameters["rel_tolerance"] * self.parameters["rel_tolerance"]
		atol2 = self.parameters["abs_tolerance"] * self.parameters["abs_tolerance"]
		tol = max(rtol2, atol2)
		# Check convergence and initialize for solve:
		converged = (rz_0 < tol)
		if converged:
			self.converged  = True
			self.reason_id   = 1
			self.final_norm = math.sqrt(rz_0)
			if(self.parameters["print_level"] >= 0):
				print( self.reason[self.reason_id])
				print( "Converged in ", self.iter, " iterations with final norm ", self.final_norm)
			return x, False
		# Check if the direction is negative before taking a step.
		feed_dict[self.problem.dw] = p
		Ap = self.sess.run(self.Aop,feed_dict = feed_dict)
		pAp = np.dot(p,Ap)
		negative_direction = (pAp <= 0.0)
		if negative_direction:
			self.converged = True
			self.reason_id = 2
			x += p 
			r -= Ap
			feed_dict[self.Minv.x] = r
			z = self.sess.run(self.Minv(),feed_dict = feed_dict)
			rz = np.dot(r,z)
			self.final_norm = math.sqrt(rz)
			if(self.parameters["print_level"] >= 0):
				print( self.reason[self.reason_id])
				print( "Converged in ", self.iter, " iterations with final norm ", self.final_norm)
			return x, False

		# Loop until convergence
		self.iter = 1
		while True:
			# Calculate alpha
			alpha = rz_0/pAp

			# Update x
			on_boundary,x = self.update_x(x,alpha,p)
			# Update r

			r -= alpha*Ap
			# Apply preconditioner z = M^{-1}r
			feed_dict[self.Minv.x] = r
			z = self.sess.run(self.Minv(),feed_dict = feed_dict)

			# Calculate rz
			rz = np.dot(r,z)
			# print(self.iter,rz)
			# Check convergence
			converged = (rz < tol)
			if converged:
				self.converged = True
				self.reason_id = 1
				self.final_norm = math.sqrt(rz)
				if(self.parameters["print_level"] >= 0):
					print( self.reason[self.reason_id])
					print( "Converged in ", self.iter, " iterations with final norm ", self.final_norm)
				break
			self.iter += 1
			if self.iter > self.parameters["max_iter"]:
				self.converged = False
				self.reason_id = 0
				self.final_norm = math.sqrt(rz)
				if(self.parameters["print_level"] >= 0):
					print( self.reason[self.reason_id])
					print( "Not Converged. Final residual norm ", self.final_norm)
				break
			beta = rz / rz_0
			p = z + beta*p
			# Check if the direction is negative, and prepare for next iteration.
			feed_dict[self.problem.dw] = p
			Ap = self.sess.run(self.Aop,feed_dict = feed_dict)
			pAp = np.dot(p,Ap)
			negative_direction = (pAp <= 0.0)

			if negative_direction:
				self.converged = True
				self.reason_id = 2
				self.final_norm = math.sqrt(rz)
				if(self.parameters["print_level"] >= 0):
					print( self.reason[self.reason_id])
					print( "Converged in ", self.iter, " iterations with final norm ", self.final_norm)
				break
			
			rz_0 = rz

		return x, on_boundary	









class CGSolver_scipy(ABC):
	"""
	This class implements a wrapper for the scipy CG solver
	"""
	reason = ["Maximum Number of Iterations Reached",
			  "Relative/Absolute residual less than tol",
			  "Reached a negative direction",
			  "Reached trust region boundary"
			  ]
	def __init__(self,problem,regularization,sess = None,Aop = None,preconditioner = None,parameters = ParametersCGSolver()):
		"""
		The constructor for this class takes
			-problem: hessianlearn.problem.Problem
			-regularization: hessianlearn.problem.Regularization
			-sees: tf.Session()
			-Aop: matrix vector product callable
			-preconditioner: hessianlearn.problem.Preconditioner (not currently even used)
			-parameters: solver hyperparameters
		"""
		self.sess = sess
		self.problem = problem
		self.regularization = regularization
		self.parameters = parameters
		if Aop is None:
			self.Aop = self.problem.Hdw + self.regularization.Hdw
		else:
			# be careful to note what the operator requires be passed into feed_dict 
			self.Aop = Aop
		# # Define preconditioner 
		# if preconditioner is None:
		# 	self.Minv = IdentityPreconditioner(problem,self.problem.dtype)
		# else:
		# 	self.Minv = preconditioner






	def solve(self,b,feed_dict = None,x_0 = None):
		r"""
		Solve Ax=b by the mines method
		as defined in Iterative Methods Ed. 2 by Youssef Saad p 140
			-b: right hand side
			-feed_dict: data dictionary for 
			-x_0: initial guess
		"""
		assert self.sess is not None
		assert feed_dict is not None

		self.iter = 0
		self.converged = False
		self.reason_id = 0
		x = np.zeros_like(b)

		feed_dict[self.problem.dw] = x
		Ax_0 = self.sess.run(self.Aop,feed_dict = feed_dict)
		# Calculate initial residual r = Ax_0 -b
		r = b - Ax_0
		# Calculate tolerance for Eisenstat Walker conditions
		rr_0 = np.dot(r,r)
		rtol2 = rr_0 * self.parameters["rel_tolerance"] * self.parameters["rel_tolerance"]
		atol2 = self.parameters["abs_tolerance"] * self.parameters["abs_tolerance"]
		tol = max(rtol2, atol2)
		import scipy
		from scipy.sparse.linalg import LinearOperator

		def Ap(p):
			feed_dict[self.problem.dw] = p
			return self.sess.run(self.Aop,feed_dict = feed_dict)

		n = self.problem.dimension

		A = LinearOperator((n,n), matvec=Ap)

		# self.iter += self.parameters["max_iter"]

		def update_iters(rk):
			self.iter +=1

		return scipy.sparse.linalg.cg(A, b, tol=tol, maxiter=self.parameters["max_iter"],callback = update_iters)








		