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
# along with stein variational inference methods class project.  If not, see <http://www.gnu.org/licenses/>.
#
# Author: Tom O'Leary-Roseberry
# Contact: tom.olearyroseberry@utexas.edu

from __future__ import absolute_import, division, print_function
import numpy as np

# from ..utilities.mpiFunctions import *




def ArmijoLineSearch(w_dir,w_dir_inner_g,cost_at_candidate, initial_cost, c_armijo = 1e-4 ,alpha =1.0, max_backtracking_iter = 10,comm = None):
	# Armijo Line Search
	line_search, line_search_iter = ( True, 0 )
	while line_search and (line_search_iter <max_backtracking_iter):
		line_search_iter   += 1
		cost_new   = cost_at_candidate(alpha*w_dir)
		# print('cost_new', cost_new, 'sufficient descent:', initial_cost + alpha*c_armijo*w_dir_inner_g )
		armijo_condition = (cost_new < initial_cost + alpha*c_armijo*w_dir_inner_g)
		if armijo_condition:
			line_search   	= False
		else:
			alpha          *= 0.5
	return alpha, line_search, line_search_iter


class TrustRegion(object):

	def __init__(self,delta_0 = 1.0,delta_hat = 1.0,eta = 0.05):
		# delta_hat: maximum trust region radius
		# delta_0: initial trust region radius
		# eta: threshold for reduction acceptance (rho<eta means we should reject the step)
		self.delta_hat = delta_hat
		self.radius = delta_0
		self.eta = eta
		
	def evaluate_step(self,actual_reduction = None,predicted_reduction = None,on_boundary = False):
		rho = actual_reduction/predicted_reduction
		if rho < 0.25:
			self.radius *= 0.5
		elif rho > 0.75 and on_boundary:
			self.radius *= 2.
			# self.delta *= max(2,self.delta_hat)
		if rho > self.eta:
			accept_step = True
		else:
			accept_step	 = False

		return accept_step



	