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

def rayleigh_quotients(H,U):
	# H is a callable:
	# U is a matrix of size (n,rank)
	rqs = np.zeros(U.shape[1])
	for i,u_i in enumerate(U.T):
		rqs[i] = np.dot(u_i,H(u_i))
	return rqs


